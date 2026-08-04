"""Sweep the REPL agent across the full ARC-AGI-3 roster.

The REPL agent uses an LLM in a Python REPL to explore and solve games.
This sweep tests it against all25 games to compare with other agents.

Usage:
    PYTHONPATH=src ARC_API_KEY=... OPENAI_API_KEY=... uv run python experiments/sweep_roster_repl.py
    WANDB_ENABLED=1 PYTHONPATH=src ARC_API_KEY=... OPENAI_API_KEY=... uv run python experiments/sweep_roster_repl.py
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
from tgaer.evaluation.wandb_logger import WandbRunLogger

OUT = "experiments/roster_results_repl.jsonl"


def main() -> None:
    key = os.environ["ARC_API_KEY"]
    games = [
        g["game_id"]
        for g in requests.get(
            f"{BASE_URL}/api/games", headers={"X-API-Key": key}
        ).json()
    ]
    print(f"[sweep] {len(games)} games", flush=True)

    # Wandb setup
    wandb_run = None
    logger = None
    if os.environ.get("WANDB_ENABLED"):
        import wandb

        wandb_run = wandb.init(
            project="tgaer-arc-agi3",
            name=f"repl-sweep-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "agent": "repl",
                "model": "openai/gpt-4o-mini",
                "vision": True,
                "max_steps": 80,
            },
        )
        logger = WandbRunLogger(
            project="tgaer-arc-agi3",
            run_name=wandb_run.name,
            log_images=True,
            image_every=5,
        )
        print(f"[sweep] wandb: {logger.url}", flush=True)

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
                        model="openai/gpt-4o-mini",
                        vision=True,
                    )
                    result = evaluate_arc_agi3_agent(
                        agent, env, {"guards": [], "max_steps": 80}, logger=logger
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

                if wandb_run:
                    wandb_run.log(
                        {
                            "game_idx": i,
                            "game": gid,
                            "score": row.get("score", 0.0),
                            "levels_completed": row.get("levels_completed", 0),
                            "total_levels": total_levels,
                            "game_time_s": game_time,
                        }
                    )
        finally:
            summary = client.close_scorecard()
            elapsed = time.time() - start_time
            print(
                f"[sweep] DONE total_levels={total_levels} elapsed={elapsed:.0f}s "
                f"scorecard={summary}",
                flush=True,
            )
            if wandb_run:
                wandb_run.summary.update(
                    {
                        "total_levels": total_levels,
                        "total_games": len(games),
                        "elapsed_s": elapsed,
                        "scorecard": summary,
                    }
                )
                wandb_run.finish()


if __name__ == "__main__":
    main()
