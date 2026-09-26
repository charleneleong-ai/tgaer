#!/usr/bin/env python3
"""Are the games we never clear budget-limited, or out of reachable states?

A derived question rather than a tuned one. The games are deterministic (one
ambiguous transition in ~2600) and their per-level state spaces are small — tu93
156 states, lp85 431, ls20 562 — so exhaustive `(state, action)` coverage is a
few hundred to a few thousand pairs, inside the 16000 actions a run affords.

If a game that never clears has **no untested pairs left**, it has exhausted
everything reachable and more budget cannot help it: the winning action is not
reachable from any state exploration found, which is a different problem. If it
has many left, exploration is the bottleneck and budget or traversal can pay.

That distinction decides whether raising `MAX_ACTIONS` reaches the 19 games that
score nothing, and it is a structural fact about the games rather than anything
fitted to them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import typer
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
)

from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    load_agent_class,
    play,
    require_starter,
)

os.chdir(REPO)

SUITE = sorted(
    p.name
    for p in (REPO / "environment_files").iterdir()
    if p.is_dir() and len(p.name) == 4
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    games: str = typer.Option(",".join(SUITE), help="Comma-separated game ids."),
    max_steps: int = typer.Option(6000),
    seed: int = typer.Option(0),
) -> None:
    """Report, per game, how much of the reachable space is still untested."""
    require_starter()
    logger.info(
        "{:6} {:>7} {:>8} {:>9} {:>9} {:>8}",
        "game",
        "levels",
        "states",
        "untested",
        "exhausted",
        "verdict",
    )
    for game_id in games.split(","):
        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(REPO / "environment_files"),
        )
        hold: dict[str, Any] = {}

        def hook(step: int, observation: Any, env: Any, actor: Any) -> None:
            hold["actor"] = actor  # the graph is read once the run is over

        try:
            row = play(
                load_agent_class(None, "explorer"),
                game_id,
                arc,
                None,
                max_steps,
                seed=seed,
                on_step=hook,
            )
        except Exception as exc:
            logger.error("{}: {}: {}", game_id, type(exc).__name__, exc)
            continue
        actor = hold.get("actor")
        if actor is None:
            logger.warning("{}: no steps recorded", game_id)
            continue
        graph = actor._graph
        states = len(graph._untested)
        left = sum(len(v) for v in graph._untested.values())
        done = sum(1 for v in graph._untested.values() if not v)
        levels = int(row.get("levels_completed", 0))
        # A game with nothing untested has run out of reachable space; budget
        # cannot reach it and the winning action is not where it looked.
        verdict = "COVERAGE" if left == 0 else "budget"
        logger.info(
            "{:6} {:7} {:8} {:9} {:8.0%} {:>9}",
            game_id,
            levels,
            states,
            left,
            done / max(states, 1),
            verdict if not levels else "scores",
        )


if __name__ == "__main__":
    app()
