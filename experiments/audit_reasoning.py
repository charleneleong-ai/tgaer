"""Audit why the ARC-AGI-3 LLM agent scores 0: play a few live steps and dump,
per step, what the model perceives (grid stats), its full reasoning, the action
it picks, and what the env returns. Not committed — a diagnostic probe."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from litellm import completion

from tgaer.agents.arc_agi3_llm import ArcAgi3LLMAgent
from tgaer.envs.arc_agi3.arc_agi3_client import ArcAgi3Client
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment

load_dotenv()

GAME = "ls20-9607627b"
MODEL = "openai/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
API_BASE = "http://localhost:8000/v1"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def grid_stats(frame: list) -> str:
    grid = (frame or [[]])[-1]
    cells = [c for row in grid for c in row]
    if not cells:
        return "empty"
    nonzero = sum(1 for c in cells if c != 0)
    colours = sorted(set(cells))
    return f"{len(cells)} cells, {nonzero} non-zero ({100*nonzero//len(cells)}%), colours={colours}"


def main() -> None:
    agent = ArcAgi3LLMAgent(model=MODEL, api_base=API_BASE, max_tokens=2048)
    client = ArcAgi3Client(api_key=os.environ["ARC_API_KEY"])
    client.open_scorecard()  # RESET requires an open scorecard (card_id)
    env = ArcAgi3Environment(client, GAME, max_actions=80)
    obs = env.reset()
    print(f"=== AUDIT {GAME}, {N} steps ===\n")
    prev_grid = None
    for step in range(1, N + 1):
        available = obs.get("available_actions") or [1]
        prompt = agent._build_prompt(obs, available)
        resp = completion(
            model=MODEL,
            api_base=API_BASE,
            api_key="local",
            messages=[
                {"role": "system", "content": prompt[:1]},  # system unused here
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        msg = resp.choices[0].message
        reasoning = (getattr(msg, "reasoning_content", "") or "").strip()
        content = (msg.content or "").strip()
        cur_grid = (obs.get("frame") or [[]])[-1]
        changed = "CHANGED" if cur_grid != prev_grid else "SAME as last step"
        prev_grid = cur_grid

        action = agent._parse(content, available) if content else agent._fallback(available)
        trans = env.step(action)
        obs = trans.state

        print(f"--- STEP {step} ---")
        print(f"perceives: {grid_stats(obs.get('frame'))}  | grid {changed}")
        print(f"available: {available}")
        print(f"REASONING: {reasoning[:700]}")
        print(f"picked: id={action.id} (x={action.x},y={action.y})  -> reward={trans.reward}  "
              f"levels={trans.info.get('levels_completed')}  state={trans.info.get('state')}")
        print()
        if trans.done:
            print("episode ended")
            break


if __name__ == "__main__":
    main()
