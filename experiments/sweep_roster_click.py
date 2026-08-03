"""Score the `click` agent across ARC-AGI-3 games that offer ACTION6.

The click agent uses CLICK_DEFAULT semantics (verb="click") and emits
ACTION6 coordinate-clicks at detected key/door centroids. This sweep
tests all ACTION6-capable games from the roster.

Usage:
    PYTHONPATH=src ARC_API_KEY=... uv run python experiments/sweep_roster_click.py
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()

import requests

from tgaer.agents.arc_agi3_planner import PlannerArcAgi3Agent
from tgaer.agents.arc_agi3_grid import CLICK_DEFAULT
from tgaer.envs.arc_agi3.arc_agi3_client import BASE_URL, ArcAgi3Client
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

OUT = "experiments/roster_results_click.jsonl"


def main() -> None:
    key = os.environ["ARC_API_KEY"]
    games = [
        g["game_id"]
        for g in requests.get(
            f"{BASE_URL}/api/games", headers={"X-API-Key": key}
        ).json()
    ]
    print(f"[sweep] {len(games)} total games", flush=True)

    client = ArcAgi3Client(api_key=key)
    card = client.open_scorecard()
    print(f"[sweep] scorecard {card}", flush=True)

    # First pass: identify ACTION6-capable games
    click_games = []
    for gid in games:
        try:
            f = client.reset(gid)
            if 6 in f.available_actions:
                click_games.append(gid)
        except Exception as exc:
            print(f"[sweep] {gid} scan error: {exc}", flush=True)

    print(f"[sweep] {len(click_games)} ACTION6-capable games", flush=True)

    total_levels = 0
    with open(OUT, "w") as fh:
        try:
            for i, gid in enumerate(click_games, 1):
                try:
                    env = ArcAgi3Environment(client, gid, max_actions=80)
                    agent = PlannerArcAgi3Agent(semantics=CLICK_DEFAULT)
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
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"[sweep] {i}/{len(click_games)} {gid} score={row.get('score')} "
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
