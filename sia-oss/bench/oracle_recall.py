#!/usr/bin/env python3
"""Can the shipped agent even propose the oracle's winning move?

`oracle_labels.py` records, for every level the offline oracle solves, the board
as it stood and the action that won from it. This replays those boards through
the agent's own `proposals` and asks where the winning action lands in its
ranking — absent, or at what rank.

That splits the two failure modes apart. Absent is a *coverage* failure: no
amount of reranking helps, the candidate generator has to change. Present but
deep is a *ranking* failure, which is what a learned prior fixes. The metric is
deterministic and instant, unlike RHAE (sd ~= 0.030pp per `ab.py`), so it can be
iterated against directly.

Exact-cell matching understates clicks badly — a button spans many cells, so a
proposed centroid can be a different cell that does the identical thing (lp85
and s5i5 score 0/5 and 0/13 exact, 5/5 and 13/13 by effect). `--effect` forks
the game and counts a candidate as a hit when it reproduces the winning frame,
which is the honest question; it costs a fork per candidate, so exact-cell stays
available as the instant version.

Read it as an upper bound on the agent in isolation: these boards lie on the
oracle's trajectory, and goal values the agent would have learned by playing are
not replayed, so cold-start ranking is what is measured.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
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

import m1_codegen as M  # noqa: E402

from tgaer.agents.arc_agi3_explorer import (  # noqa: E402
    click_targets,
    field_box,
    proposals,
)
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)

CLICK_ID = 6
LABELS = REPO / "sia-oss" / "bench" / "oracle"
CUTOFFS = (1, 4, 12)

app = typer.Typer(add_completion=False)


def rank_of(arr: np.ndarray, available: list[int], target: tuple) -> int:
    """Where the winning action sits in the agent's ranking; -1 if absent."""
    prims = proposals(arr, available, box=field_box(arr))
    return prims.index(target) if target in prims else -1


def effect_ranks(game_id: str, data: Any) -> list[int]:
    """Rank of the first proposed click reproducing the winning frame; -1 if none.

    Replays the oracle's plan against a live game so each candidate can be
    forked and compared by effect rather than by cell identity.
    """
    require_starter()
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    env = arc.make(game_id)
    if env is None:
        raise RuntimeError("env-unavailable")
    fd = env.reset()
    game = env._game
    out: list[int] = []
    for i in range(len(data["kind"])):
        kind = int(data["kind"][i])
        win = (
            (int(data["row"][i]), int(data["col"][i]))
            if kind == CLICK_ID
            else int(kind)
        )
        arr = M.grid(fd)
        if kind == CLICK_ID:
            goal = M.grid(M.act(copy.deepcopy(game), win))
            rank = -1
            for j, t in enumerate(click_targets(arr, box=field_box(arr))):
                if np.array_equal(M.grid(M.act(copy.deepcopy(game), t)), goal):
                    rank = j
                    break
            out.append(rank)
        else:
            mask = int(data["avail"][i])
            available = [a for a in range(16) if mask >> a & 1]
            out.append(rank_of(arr, available, ("act", kind)))
        fd = M.act(game, win)
    return out


def game_ranks(path: Path) -> list[int]:
    data = np.load(path)
    out: list[int] = []
    for i in range(len(data["kind"])):
        kind = int(data["kind"][i])
        mask = int(data["avail"][i])
        available = [a for a in range(16) if mask >> a & 1]
        target = (
            ("click", int(data["row"][i]), int(data["col"][i]))
            if kind == CLICK_ID
            else ("act", kind)
        )
        out.append(rank_of(data["frames"][i].astype(np.int16), available, target))
    return out


@app.command()
def main(
    labels: Path = typer.Option(LABELS, help="Directory of per-game npz labels."),
    effect: bool = typer.Option(
        True, help="Count a click as covered when it reproduces the winning frame."
    ),
) -> None:
    """Report per-game coverage and rank of the oracle's winning actions."""
    files = sorted(p for p in labels.glob("*.npz"))
    if not files:
        raise typer.BadParameter(f"no labels in {labels}; run oracle_labels.py first")

    everything: list[int] = []
    header = "  ".join(f"@{k}" for k in CUTOFFS)
    logger.info("{:6} {:>7} {:>8}  {}", "game", "labels", "covered", header)
    for path in files:
        ranks = effect_ranks(path.stem, np.load(path)) if effect else game_ranks(path)
        everything.extend(ranks)
        found = [r for r in ranks if r >= 0]
        at = "  ".join(
            f"{sum(1 for r in found if r < k) / len(ranks):.2f}" for k in CUTOFFS
        )
        logger.info(
            "{:6} {:7} {:7.0%}  {}   median rank {}",
            path.stem,
            len(ranks),
            len(found) / len(ranks),
            at,
            int(np.median(found)) if found else "-",
        )

    found = [r for r in everything if r >= 0]
    logger.success(
        "{}/{} winning actions are proposable at all ({:.0%})",
        len(found),
        len(everything),
        len(found) / max(len(everything), 1),
    )
    for k in CUTOFFS:
        hit = sum(1 for r in found if r < k)
        logger.info(
            "recall@{}: {}/{} ({:.0%})",
            k,
            hit,
            len(everything),
            hit / max(len(everything), 1),
        )


if __name__ == "__main__":
    app()
