"""Score the `planner` agent across the full ARC-AGI-3 roster (GET /api/games)
under one scorecard — the competition-style aggregate. Writes one JSONL row per
game as it finishes so progress is readable mid-run.

The planner is LS20-tuned (key/door cell-value heuristic), so most non-LS20
games are expected to score 0; the point is to measure where the key->door
grammar transfers and the roster-wide total vs the 0-everywhere LLM baselines.
"""

from __future__ import annotations

import json
import os
import time

import requests

from tgaer.agents.arc_agi3_planner import PlannerArcAgi3Agent
from tgaer.envs.arc_agi3.arc_agi3_client import BASE_URL, ArcAgi3Client
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

OUT = "/workspace/tgaer/experiments/roster_results.jsonl"


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
    with open(OUT, "w") as fh:
        try:
            for i, gid in enumerate(games, 1):
                try:
                    env = ArcAgi3Environment(client, gid, max_actions=80)
                    result = evaluate_arc_agi3_agent(
                        PlannerArcAgi3Agent(), env, {"guards": [], "max_steps": 80}
                    )
                    row = {"game": gid, "score": result.score, **result.details}
                    total_levels += int(result.details.get("levels_completed", 0) or 0)
                except Exception as exc:  # one bad game must not abort the roster
                    row = {
                        "game": gid,
                        "score": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"[sweep] {i}/{len(games)} {gid} score={row.get('score')} "
                    f"levels={row.get('levels_completed')} state={row.get('state')} "
                    f"err={row.get('error')} total_levels={total_levels}",
                    flush=True,
                )
        finally:
            summary = client.close_scorecard()
            print(
                f"[sweep] DONE total_levels={total_levels} scorecard={summary}",
                flush=True,
            )


if __name__ == "__main__":
    main()
