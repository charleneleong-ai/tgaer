#!/usr/bin/env python3
"""Does a model-written `distance` drive search as well as a privileged one?

M0 cleared lp85 L1 in 5 real actions and 9 expansions using a heuristic built
from the game's own sprite tags (`bghvgbtwcb`, `goal`) — information no agent has
in the kernel. This swaps in the `distance` a model wrote from recorded play and
changes nothing else.

Deliberately keeps expanding against a `deepcopy` of the real game rather than
the generated `simulate`, which scores 0% on held-out transitions: the point is
to measure the heuristic on its own, since M0 showed the heuristic — not the
world model — is what gates search.
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
import m1_codegen as M  # noqa: E402

from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)
os.environ.setdefault("ARC_TGAER_REPO", str(REPO))

app = typer.Typer(add_completion=False)

# What M0's privileged heuristic achieved, for comparison.
M0_ACTIONS, M0_EXPANSIONS = 5, 9


@app.command()
def main(
    generated: Path = typer.Argument(..., help="File defining `distance`."),
    max_expansions: int = typer.Option(60000),
) -> None:
    """Greedy search on the generated heuristic; report real actions to clear L1."""
    namespace: dict[str, object] = {"np": np}
    exec(compile(generated.read_text(), str(generated), "exec"), namespace)  # noqa: S102
    distance = namespace["distance"]

    require_starter()
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    env = arc.make("lp85")
    fd = env.reset()
    game = env._game
    base = M.grid(fd)
    bg = M.background(base)
    cells = M.button_cells(game, base)

    def score(arr: np.ndarray) -> float:
        return float(distance(M.objects(arr, bg)))

    t0 = time.monotonic()
    start = score(base)
    logger.info("{} buttons | start distance {}", len(cells), start)

    counter, expansions, best = 0, 0, start
    heap = [(start, 0, counter, copy.deepcopy(game), 0)]
    seen = {base.tobytes()}
    while heap and expansions < max_expansions:
        _, depth, _, node, taken = heapq.heappop(heap)
        for rc in cells:
            child = copy.deepcopy(node)
            out = M.click(child, rc)
            expansions += 1
            if int(getattr(out, "levels_completed", 0)) > 0:
                logger.success(
                    "solved in {} real actions, {} expansions, {:.0f}s "
                    "(M0's privileged heuristic: {} actions, {} expansions)",
                    taken + 1,
                    expansions,
                    time.monotonic() - t0,
                    M0_ACTIONS,
                    M0_EXPANSIONS,
                )
                return
            if "GAME_OVER" in str(getattr(out, "state", "")):
                continue
            arr = M.grid(out)
            if arr.tobytes() in seen:
                continue
            seen.add(arr.tobytes())
            value = score(arr)
            best = min(best, value)
            counter += 1
            heapq.heappush(heap, (value, depth + 1, counter, child, taken + 1))
    logger.error("no solution in {} expansions, best distance {}", expansions, best)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
