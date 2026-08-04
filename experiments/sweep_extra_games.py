"""Sweep agents across extra local games (pw14, tl01).

These games aren't served by the hosted API - they run locally via arcengine.
Tests both the planner and LLM REPL agents.

Usage:
    PYTHONPATH=src ARC_API_KEY=... OPENAI_API_KEY=... uv run python experiments/sweep_extra_games.py
"""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

from tgaer.agents.arc_agi3_planner import PlannerArcAgi3Agent
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.envs.arc_agi3.arc_agi3_local import LocalArcTransport
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

EXTRA_GAMES = [
    ("tl01-f96b1cdb", "Tutorial"),
    ("pw14-c1586e55", "PushWorld"),
    ("custom_kd01-test", "Custom Key-Door"),
    ("custom_maze01-test", "Custom Maze"),
]

OUT = "experiments/extra_games_results.jsonl"


def main() -> None:
    transport = LocalArcTransport()
    print(f"[sweep] {len(EXTRA_GAMES)} extra games", flush=True)

    total_levels = 0
    start_time = time.time()
    
    with open(OUT, "w") as fh:
        try:
            for i, (gid, name) in enumerate(EXTRA_GAMES, 1):
                game_start = time.time()
                try:
                    env = ArcAgi3Environment(transport, gid, max_actions=80)
                    agent = PlannerArcAgi3Agent()
                    result = evaluate_arc_agi3_agent(
                        agent, env, {"guards": [], "max_steps": 80}
                    )
                    row = {
                        "game": gid,
                        "name": name,
                        "score": result.score,
                        **result.details,
                    }
                    total_levels += int(result.details.get("levels_completed", 0) or 0)
                except Exception as exc:
                    row = {
                        "game": gid,
                        "name": name,
                        "score": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                
                game_time = time.time() - game_start
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                
                print(
                    f"[sweep] {i}/{len(EXTRA_GAMES)} {gid} ({name}) "
                    f"score={row.get('score')} "
                    f"levels={row.get('levels_completed')} "
                    f"err={row.get('error')} total_levels={total_levels} "
                    f"time={game_time:.1f}s",
                    flush=True,
                )
        finally:
            transport.close()
            elapsed = time.time() - start_time
            print(
                f"[sweep] DONE total_levels={total_levels} elapsed={elapsed:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
