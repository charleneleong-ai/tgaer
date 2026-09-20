#!/usr/bin/env python3
"""M0's planner across the whole roster, with no game-specific knowledge.

`m0_plan.py` cleared lp85 L2 in 8 real actions against the explorer's 364, but
it used a heuristic built from *that game's* sprite tags. Nothing about that
transfers. This runs the same forked-game search on every game with the one
thing that is always available — "did the level counter go up" — and no distance
signal at all, to measure how much of the roster offline search reaches
unaided.

Expect most games to fail. M0 already showed uninformed BFS stalls around depth
6 on lp85 because branching is ~6 with almost no state merging; the point here
is to find which games are *small enough* that it does not, because those are
the ones a generic planner could take without solving the heuristic problem
first.

Both action kinds are covered: simple ids come from `available_actions`, clicks
from a stride-3 probe of the board (`m1_codegen.button_cells`).

Caveat that applies to everything here: the scored Kaggle run is online against
a gateway with no local game object, so forking is a local-only upper bound.
See docs/arc-agi3-world-model.md.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import typer
from loguru import logger

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
)
import m1_codegen as M  # noqa: E402

from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)
os.environ.setdefault("ARC_TGAER_REPO", str(REPO))

app = typer.Typer(add_completion=False)

SUITE = sorted(
    p.name
    for p in (REPO / "environment_files").iterdir()
    if p.is_dir() and len(p.name) == 4
)
# The explorer's per-game cost at the shipping budget, for comparison.
EXPLORER_ACTIONS = 6000


def actions_for(game: object, fd: object, base: np.ndarray) -> list:
    """Every distinct action: simple ids plus one click per distinct effect."""
    simple = [
        int(a)
        for a in (getattr(fd, "available_actions", None) or [])
        if int(a) != M.CLICK_ID
    ]
    return list(simple) + M.button_cells(game, base)


def plan_level(
    game: object,
    base: np.ndarray,
    start_levels: int,
    cells: list,
    budget: int,
    seconds: float,
) -> tuple[list | None, int]:
    """Breadth-first to the next level-up over forked games. No heuristic.

    Capped on wall clock as well as expansions: every game that succeeds does so
    within ~2 minutes, while a hopeless one burns the whole budget, and on a
    board with large grids a single expansion is slow enough that 15k of them
    ran past 15 minutes on lf52 alone.
    """
    deadline = time.monotonic() + seconds
    queue: deque = deque([(copy.deepcopy(game), [])])
    seen = {base.tobytes()}
    expansions = 0
    while queue and expansions < budget and time.monotonic() < deadline:
        node, path = queue.popleft()
        for action in cells:
            child = copy.deepcopy(node)
            out = M.act(child, action)
            expansions += 1
            if int(getattr(out, "levels_completed", start_levels)) > start_levels:
                return path + [action], expansions
            if "GAME_OVER" in str(getattr(out, "state", "")):
                continue
            key = M.grid(out).tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + [action]))
    return None, expansions


@app.command()
def main(
    games: str = typer.Option(",".join(SUITE), help="Comma-separated game ids."),
    budget: int = typer.Option(20000, help="Expansions per level before giving up."),
    levels: int = typer.Option(3, help="Levels to attempt per game."),
    per_game: float = typer.Option(150.0, help="Seconds per level before giving up."),
) -> None:
    """Report, per game, how far uninformed offline planning gets."""
    require_starter()
    solved = 0
    for game_id in games.split(","):
        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(REPO / "environment_files"),
        )
        env = arc.make(game_id)
        if env is None:
            logger.warning("{}: no environment", game_id)
            continue
        fd = env.reset()
        game = env._game
        t0 = time.monotonic()
        try:
            cells = actions_for(game, fd, M.grid(fd))
        except Exception as exc:
            logger.error("{}: probing raised {}", game_id, type(exc).__name__)
            continue
        if not cells:
            logger.warning("{}: no actions found", game_id)
            continue

        spent = cleared = 0
        try:
            for _ in range(levels):
                done = int(getattr(fd, "levels_completed", 0))
                plan, expansions = plan_level(
                    game, M.grid(fd), done, cells, budget, per_game
                )
                if plan is None:
                    break
                for action in plan:
                    fd = M.act(game, action)
                spent += len(plan)
                cleared += 1
        except Exception as exc:
            # The games raise on their own account under exhaustive probing —
            # lf52 hits `list.remove(x): x not in list` inside its own step code.
            # One game's bug must not end the sweep.
            logger.error("{}: game raised {}: {}", game_id, type(exc).__name__, exc)
        dt = time.monotonic() - t0
        if cleared:
            solved += 1
            logger.success(
                "{}: {} level(s) in {} real actions ({:.0f}s) — explorer spends "
                "up to {}",
                game_id,
                cleared,
                spent,
                dt,
                EXPLORER_ACTIONS,
            )
        else:
            logger.info(
                "{}: no level reached, {} actions available, {:.0f}s",
                game_id,
                len(cells),
                dt,
            )
    logger.info("{}/{} games reached at least one level", solved, len(games.split(",")))


if __name__ == "__main__":
    app()
