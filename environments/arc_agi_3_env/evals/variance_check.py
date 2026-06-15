"""Offline check that the shaped reward has cross-rollout variance on a solvable
local maze — the go/no-go before any RL training (a flat reward has no gradient).

Drives N rollouts of one game via verifiers + Gemini (OpenAI-compat, no GPU) and
prints the per-rollout total reward + each shaped component (win / novelty /
anti-stall). Configure via env vars; needs a py3.12 venv with verifiers +
arcengine + tgaer on the path:

    PYTHONPATH=<env_dir>:<repo>/src GEMINI_API_KEY=... \
      python environments/arc_agi_3_env/evals/variance_check.py

Env vars: GAME_FAMILY (simple_maze), LEVEL_INDEX (0), MAX_TURNS (40),
ROLLOUTS (8), MODEL (gemini-2.5-flash).
"""

import os
import statistics

import shaped_reward as sh
from dotenv import load_dotenv
from verifiers.clients import ClientConfig

GAME_FAMILY = os.environ.get("GAME_FAMILY", "simple_maze")
LEVEL_INDEX = int(os.environ.get("LEVEL_INDEX", "0"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "40"))
ROLLOUTS = int(os.environ.get("ROLLOUTS", "8"))
MODEL = os.environ.get("MODEL", "gemini-2.5-flash")


def _get(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def main() -> None:
    load_dotenv()  # ARC/GEMINI keys from the nearest .env
    assert os.environ.get("GEMINI_API_KEY"), "GEMINI_API_KEY not set"
    env = sh.load_environment(
        game_family=GAME_FAMILY, level_index=LEVEL_INDEX, max_turns=MAX_TURNS
    )
    client = ClientConfig(
        client_type="openai_chat_completions",
        api_key_var="GEMINI_API_KEY",
        api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    res = env.evaluate_sync(
        client=client,
        model=MODEL,
        num_examples=1,
        rollouts_per_example=ROLLOUTS,
        sampling_args={"max_tokens": 512, "temperature": 0.8},
    )
    out = (res if isinstance(res, dict) else vars(res))["outputs"]
    rewards = [float(_get(o, "reward")) for o in out if _get(o, "reward") is not None]

    print(
        f"=== {GAME_FAMILY} L{LEVEL_INDEX} · {MODEL} · {MAX_TURNS} turns · {ROLLOUTS} rollouts ==="
    )
    print("total reward:", [round(r, 4) for r in rewards])
    for comp in ("win_reward", "novelty_reward", "anti_stall_penalty", "num_turns"):
        if _get(out[0], comp) is not None:
            print(f"  {comp}: {[round(float(_get(o, comp)), 4) for o in out]}")
    if rewards:
        sd = statistics.pstdev(rewards)
        print(
            f"mean={statistics.mean(rewards):.4f} stdev={sd:.4f} "
            f"min={min(rewards):.4f} max={max(rewards):.4f} wins={sum(_get(o, 'win_reward') for o in out):.0f}"
        )
        print("VARIANCE PRESENT" if sd > 1e-6 else "FLAT (no gradient)")


if __name__ == "__main__":
    main()
