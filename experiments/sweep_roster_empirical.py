"""Score the `empirical` agent across the full ARC-AGI-3 roster (GET /api/games)
under one scorecard — the competition-style aggregate. Writes one JSONL row per
game as it finishes so progress is readable mid-run.

Unlike the LS20-tuned `planner` sweep (``sweep_roster.py``), the empirical agent
derives avatar/key/door from the action stream with no per-game palette, so the
headline metric is **non-zero on any game beyond LS20** — the transfer the
hardcoded planner and the per-frame VL scientist both fail. A fresh agent is
built per game (each game is a new episode → detector resets); the VL cold-start
is queried once per game at frame 0.
"""

from __future__ import annotations

import json
import os

import requests

from tgaer.agents.arc_agi3_empirical import EmpiricalPlannerAgent
from tgaer.envs.arc_agi3.arc_agi3_client import BASE_URL, ArcAgi3Client
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

OUT = "/workspace/tgaer/experiments/roster_results_empirical.jsonl"


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
                        EmpiricalPlannerAgent(), env, {"guards": [], "max_steps": 80}
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
