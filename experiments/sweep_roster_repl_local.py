"""Sweep REPL agent locally with Qwen3.5-4B via llama-server.

Usage:
    PYTHONPATH=src uv run python experiments/sweep_roster_repl_local.py
"""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

import requests

from tgaer.agents.arc_agi3_repl import ArcAgi3ReplAgent
from tgaer.envs.arc_agi3.arc_agi3_client import BASE_URL, ArcAgi3Client
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

OUT = "experiments/roster_results_repl_local.jsonl"

# Local llama-server config
LOCAL_MODEL = "openai/qwen/qwen3.5-4b"
LOCAL_API_BASE = "http://127.0.0.1:8080/v1"
LOCAL_API_KEY = "none"


def main() -> None:
    key = os.environ["ARC_API_KEY"]
    games = [
        g["game_id"]
        for g in requests.get(
            f"{BASE_URL}/api/games", headers={"X-API-Key": key}
        ).json()
    ]
    print(f"[sweep] {len(games)} games", flush=True)

    client = ArcAgi3Client(api_key=key)
    card = client.open_scorecard()
    print(f"[sweep] scorecard {card}", flush=True)

    total_levels = 0
    start_time = time.time()
    
    with open(OUT, "w") as fh:
        try:
            for i, gid in enumerate(games, 1):
                game_start = time.time()
                try:
                    env = ArcAgi3Environment(client, gid, max_actions=80)
                    agent = ArcAgi3ReplAgent(
                        model=LOCAL_MODEL,
                        api_base=LOCAL_API_BASE,
                        api_key=LOCAL_API_KEY,
                        vision=False,  # llama-server doesn't support images
                        max_tool_steps=3,  # Fewer steps for speed
                        max_tokens=500,
                        temperature=0.6,
                    )
                    result = evaluate_arc_agi3_agent(
                        agent, env, {"guards": [], "max_steps": 80}
                    )
                    row = {"game": gid, "score": result.score, **result.details}
                    total_levels += int(result.details.get("levels_completed", 0) or 0)
                except Exception as exc:
                    row = {
                        "game": gid,
                        "score": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                
                game_time = time.time() - game_start
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (len(games) - i)
                
                print(
                    f"[sweep] {i}/{len(games)} {gid} score={row.get('score')} "
                    f"levels={row.get('levels_completed')} state={row.get('state')} "
                    f"err={row.get('error')} total_levels={total_levels} "
                    f"time={game_time:.1f}s ETA={eta:.0f}s",
                    flush=True,
                )
        finally:
            summary = client.close_scorecard()
            elapsed = time.time() - start_time
            print(
                f"[sweep] DONE total_levels={total_levels} elapsed={elapsed:.0f}s "
                f"scorecard={summary}",
                flush=True,
            )


if __name__ == "__main__":
    main()
