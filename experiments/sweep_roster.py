"""Score an agent (AGENT=explorer|planner) across the full ARC-AGI-3 roster
(GET /api/games) under one scorecard — the competition-style aggregate. Writes
one JSONL row per game to roster_results_<agent>.jsonl as it finishes so progress
is readable mid-run.

The `planner` baseline is LS20-tuned (key/door cell-value heuristic) and scores
on 1/25 games; the `explorer` is game-agnostic frontier exploration. Run both
head-to-head to measure where the explorer lifts levels the planner can't touch.
"""

from __future__ import annotations

import json
import os

import requests

from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
from tgaer.agents.arc_agi3_planner import PlannerArcAgi3Agent
from tgaer.envs.arc_agi3.arc_agi3_client import BASE_URL, ArcAgi3Client
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

# AGENT selects the controller under test: the game-agnostic explorer (default)
# or the LS20-tuned planner baseline for head-to-head comparison.
AGENTS = {"explorer": ExplorerArcAgi3Agent, "planner": PlannerArcAgi3Agent}
AGENT_NAME = os.environ.get("AGENT", "explorer")
AGENT = AGENTS[AGENT_NAME]
OUT = f"/workspace/tgaer/experiments/roster_results_{AGENT_NAME}.jsonl"

# tgaer's env default is 80, but the real ARC-AGI-3 budget is far larger (preview
# winners used ~4k interactions). Systematic exploration is budget-hungry, so the
# explorer needs a higher cap to be measured fairly — override via MAX_ACTIONS.
MAX_ACTIONS = int(os.environ.get("MAX_ACTIONS", "80"))


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
                    env = ArcAgi3Environment(
                        client,
                        gid,
                        max_actions=MAX_ACTIONS,
                        # let the explorer learn fatal transitions and respawn
                        # within budget instead of forfeiting on first death
                        reset_on_game_over=AGENT is ExplorerArcAgi3Agent,
                    )
                    result = evaluate_arc_agi3_agent(
                        AGENT(),
                        env,
                        {"guards": [], "max_steps": MAX_ACTIONS},
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
