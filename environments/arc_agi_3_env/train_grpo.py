"""Self-contained GRPO on the local arcengine maze with Qwen3-0.6B + LoRA.

No external trainer (prime-rl is git-blocked here) — just torch/transformers/peft +
the vendored env's local backend + tgaer shaping. GRPO core = group-relative
advantage (REINFORCE with a per-group baseline) on a LoRA policy.

Env vars: GAME (complex_maze), LEVELS ("0,1,2,3,4"), GROUP (6), MAX_TURNS (30),
STEPS (60), LR (1e-5), MODEL (Qwen/Qwen3-0.6B), SMOKE (0/1 = rollouts only).
"""

import os
import random
import statistics
import time

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

import arc_agi_3_env as _pkg  # noqa: F401 — runs the loader, registers the impl module

_impl = sys.modules["_arc_agi_3_env_impl"]  # the real module (package __init__ shadows it)
LocalArcEngineBackend = _impl.LocalArcEngineBackend
_parse_action = _impl._parse_action

from tgaer.envs.arc_agi3 import shaping

GAME = os.environ.get("GAME", "complex_maze")
LEVELS = [int(x) for x in os.environ.get("LEVELS", "0,1,2,3,4").split(",")]
GROUP = int(os.environ.get("GROUP", "6"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "30"))
STEPS = int(os.environ.get("STEPS", "60"))
LR = float(os.environ.get("LR", "1e-5"))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
SMOKE = os.environ.get("SMOKE", "0") == "1"
DEV = "cuda"

SYSTEM = (
    "You are playing a grid maze. Reply with EXACTLY one XML action tag and nothing else. "
    "Use <action>RESET</action> to start, then <action>ACTION1</action>/ACTION2/ACTION3/ACTION4 "
    "to move. Goal: reach state WIN."
)

print(f"[grpo] {MODEL} on {GAME} levels={LEVELS} group={GROUP} steps={STEPS} smoke={SMOKE}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV)
model = get_peft_model(
    model,
    LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"),
)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def _render(prompt: str, assistant: str | None = None) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return tok.apply_chat_template(
        msgs, add_generation_prompt=(assistant is None), enable_thinking=False, tokenize=False
    )


@torch.no_grad()
def rollout(level: int):
    be = LocalArcEngineBackend(GAME, level)
    res = be.step("RESET")
    turns, st, parsed_ok = [], {}, 0
    for t in range(MAX_TURNS):
        enc = tok(_render(res.observation), return_tensors="pt").to(DEV)
        out = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=1.0, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        parsed = _parse_action(text)
        if parsed:
            action, data = parsed
            parsed_ok += 1
        else:
            action, data = random.choice(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]), {}
        # train on what the model ACTUALLY generated (coherent credit assignment)
        turns.append((res.observation, text))
        try:
            res = be.step(action, data)
        except Exception:
            break
        shaping.observe(st, res.observation)
        if res.game_state in ("WIN", "GAME_OVER"):
            break
    st["game_state"] = res.game_state
    win = 1.0 if res.game_state == "WIN" else 0.0
    parse_rate = parsed_ok / max(len(turns), 1)
    # format reward gives a weak model an immediately-learnable signal (emit valid actions)
    reward = (
        shaping.win_reward(st)
        + 0.2 * shaping.novelty_reward(st)
        + 0.1 * shaping.anti_stall_penalty(st)
        + 0.3 * parse_rate
    )
    return turns, reward, win, parse_rate


def logprob_of(prompt: str, assistant: str) -> torch.Tensor:
    plen = tok(_render(prompt), return_tensors="pt").input_ids.shape[1]
    fenc = tok(_render(prompt, assistant), return_tensors="pt").to(DEV)
    logits = model(**fenc).logits[0, :-1]
    targets = fenc.input_ids[0, 1:]
    lp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return lp[plen - 1:].mean()  # mean logprob over assistant (action) tokens


def main():
    for step in range(1, STEPS + 1):
        level = LEVELS[step % len(LEVELS)]
        roll = [rollout(level) for _ in range(GROUP)]
        rewards = [r for _, r, _, _ in roll]
        wins = [w for _, _, w, _ in roll]
        prate = statistics.mean([p for _, _, _, p in roll])
        mean_r, std_r = statistics.mean(rewards), (statistics.pstdev(rewards) or 1.0)
        if SMOKE:
            print(f"[smoke step{step}] L{level} rewards={[round(r,3) for r in rewards]} "
                  f"wins={sum(wins)} parse_rate={prate:.2f}", flush=True)
            if step >= 2:
                break
            continue
        # GRPO update — backprop PER TURN (accumulate grads) so we never retain
        # many big-vocab logit graphs at once (that OOMs a 40GB card).
        opt.zero_grad()
        active = [(turns, (r - mean_r) / (std_r + 1e-6)) for (turns, r, _, _) in roll if turns]
        active = [(t, a) for (t, a) in active if abs(a) > 1e-8]
        lv, n_terms = 0.0, 0
        for (turns, adv) in active:
            scale = 1.0 / (len(active) * len(turns))
            for (p, a) in turns:
                loss = -adv * logprob_of(p, a) * scale
                loss.backward()
                lv += loss.item()
                n_terms += 1
        if n_terms:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[step {step}] L{level} mean_reward={mean_r:.4f} wins={sum(wins):.0f}/{GROUP} "
              f"parse_rate={prate:.2f} std={std_r:.4f} loss={lv:.4f} gpu={mem:.1f}G "
              f"t={time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
