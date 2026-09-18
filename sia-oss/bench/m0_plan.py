#!/usr/bin/env python3
"""M0, second pass: goal-directed search through a perfect lp85 simulator.

Uninformed BFS over a perfect simulator fails on lp85 L2 — 40k expansions and
26k distinct states still only reached depth 6, because branching is ~6 with
almost no state merging. That is the M0 result that matters: obtaining the world
model was trivial (deepcopy), and it is *not sufficient*. Search needs a
distance-to-goal signal.

lp85's win condition supplies one. `khartslnwa` requires every `bghvgbtwcb`
sprite to sit on a `goal` and every `fdgmtkfrxl` on a `goal-o`, so the count of
misplaced blocks is an admissible-ish heuristic. This runs greedy best-first on
it (ties broken by depth), which trades shortest-path guarantees for reach.

Explorer baseline: L1 9, L2 364, L3 100, L4 428 actions.
"""

from __future__ import annotations

import copy
import heapq
import os
import sys
import time
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
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)
os.environ.setdefault("ARC_TGAER_REPO", str(REPO))

from arcengine import ActionInput, GameAction  # noqa: E402

app = typer.Typer(add_completion=False)

# lp85 renders sprites as 3x3 blocks, so a stride-3 probe lattice cannot miss one.
SCALE = 3
# The explorer's per-level cost on the 25-game scored run, for comparison.
EXPLORER = {1: 9, 2: 364, 3: 100, 4: 428}


def grid(fd) -> np.ndarray:
    return np.asarray(fd.frame[-1], dtype=np.int16)


def click(g, rc: tuple[int, int]):
    r, c = rc
    return g.perform_action(
        ActionInput(id=GameAction.ACTION6, data={"x": int(c), "y": int(r)}), raw=True
    )


PAIRS = (("bghvgbtwcb", "goal"), ("fdgmtkfrxl", "goal-o"))


def misplaced(g) -> int:
    """Summed distance from each block to its nearest goal of the right kind.

    `Lp85.khartslnwa` only asks whether every `bghvgbtwcb` sits on a `goal` and
    every `fdgmtkfrxl` on a `goal-o`, which as a heuristic takes about three
    distinct values on a two-block level and gives greedy search no gradient to
    follow. Distance keeps the win condition (0 exactly when it holds) and adds
    the gradient back.
    """
    lvl = g.current_level
    total = 0
    for tag, goal_tag in PAIRS:
        goals = [(t.x, t.y) for t in lvl.get_sprites_by_tag(goal_tag)]
        if not goals:
            continue
        for s in lvl.get_sprites_by_tag(tag):
            if lvl.get_sprite_at(s.x + 1, s.y + 1, goal_tag) is not None:
                continue
            total += min(abs(s.x - gx) + abs(s.y - gy) for gx, gy in goals)
    return total


def button_cells(game, base: np.ndarray) -> list[tuple[int, int]]:
    seen: dict[bytes, tuple[int, int]] = {}
    for r in range(0, 64, SCALE):
        for c in range(0, 64, SCALE):
            g2 = grid(click(copy.deepcopy(game), (r, c)))
            if np.array_equal(g2, base):
                continue
            seen.setdefault(g2.tobytes(), (r, c))
    return list(seen.values())


def solve(
    game, base: np.ndarray, start_levels: int, max_expansions: int
) -> tuple[list[tuple[int, int]] | None, int, float]:
    """Greedy best-first on ``misplaced``; returns (plan, expansions, seconds)."""
    t0 = time.monotonic()
    cells = button_cells(game, base)
    h0 = misplaced(game)
    logger.info("{} buttons, distance-to-goal {} at start", len(cells), h0)

    counter = 0
    heap = [(h0, 0, counter, copy.deepcopy(game), [])]
    seen: set[bytes] = {base.tobytes()}
    expansions = 0
    best = h0
    while heap and expansions < max_expansions:
        _, depth, _, node, path = heapq.heappop(heap)
        for rc in cells:
            child = copy.deepcopy(node)
            out = click(child, rc)
            expansions += 1
            if int(getattr(out, "levels_completed", start_levels)) > start_levels:
                return path + [rc], expansions, time.monotonic() - t0
            if "GAME_OVER" in str(getattr(out, "state", "")):
                continue
            key = grid(out).tobytes()
            if key in seen:
                continue
            seen.add(key)
            h = misplaced(child)
            best = min(best, h)
            counter += 1
            heapq.heappush(heap, (h, depth + 1, counter, child, path + [rc]))
        if expansions % 10000 < len(cells):
            logger.debug(
                "{} expansions, {} states, best h={}, depth {}, {:.0f}s",
                expansions,
                len(seen),
                best,
                depth,
                time.monotonic() - t0,
            )
    return None, expansions, time.monotonic() - t0


@app.command()
def main(
    levels: int = typer.Option(4, help="How many lp85 levels to plan through."),
    max_expansions: int = typer.Option(
        60000, help="Offline expansion budget per level."
    ),
) -> None:
    """Plan lp85 offline and report real actions spent against the explorer."""
    require_starter()
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    env = arc.make("lp85")
    fd = env.reset()
    game = env._game

    total = 0
    for lvl in range(1, levels + 1):
        done = int(getattr(fd, "levels_completed", 0))
        logger.info("level {}", lvl)
        plan, expansions, dt = solve(game, grid(fd), done, max_expansions)
        if plan is None:
            logger.error("no plan after {} expansions ({:.0f}s)", expansions, dt)
            break
        for rc in plan:
            fd = click(game, rc)
        total += len(plan)
        ref = EXPLORER[lvl]
        logger.success(
            "PLAN {} actions | explorer {} | {:.1f}x fewer | {} expansions, {:.0f}s",
            len(plan),
            ref,
            ref / len(plan),
            expansions,
            dt,
        )

    ref_total = sum(EXPLORER[i] for i in range(1, levels + 1))
    logger.info("total real actions {} (explorer {})", total, ref_total)


if __name__ == "__main__":
    app()
