#!/usr/bin/env python3
"""Can a ranker learned from the oracle's labels beat the shipped proposal order?

`oracle_recall.py` says the winning move is already proposable 97% of the time
but sits at rank 0 only 18% of the time, and `_choose` plays `untested[0]` in
proposal order — so reordering `proposals` is the whole intervention. This fits
a ranker on the labelled decisions and scores it the only way that means
anything for a submission: **leave-one-game-out**, because the scored kernel
plays games this never saw.

In-sample numbers are deliberately not reported. A ranker fitted and scored on
the same 11 games says nothing about transfer, which is the entire question.

Features are what the agent can compute from the frame it is looking at —
component geometry and colour statistics for clicks, the available-action set
for simple actions, plus the existing proposal rank so the model can learn to
fall back on it.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
)
import m1_codegen as M  # noqa: E402

from tgaer.agents.arc_agi3_explorer import field_box, proposals  # noqa: E402
from tgaer.agents.arc_agi3_grid import components  # noqa: E402
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)

CLICK_ID = 6
LABELS = REPO / "sia-oss" / "bench" / "oracle"
N_FEATURES = 12

app = typer.Typer(add_completion=False)


def cell_index(arr: np.ndarray) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Cell -> (component size, bounding-box area, colour count on the board)."""
    index: dict[tuple[int, int], tuple[int, int, int]] = {}
    counts = {int(v): int((arr == v).sum()) for v in np.unique(arr)}
    for v in np.unique(arr).tolist():
        for comp in components(arr, (v,)):
            rows, cols = comp[:, 0], comp[:, 1]
            bbox = (int(rows.max() - rows.min()) + 1) * (
                int(cols.max() - cols.min()) + 1
            )
            for r, c in comp.tolist():
                index[(r, c)] = (len(comp), bbox, counts[v])
    return index


def features(
    prim: tuple,
    rank: int,
    arr: np.ndarray,
    available: list[int],
    index: dict[tuple[int, int], tuple[int, int, int]],
) -> list[float]:
    """Frame-derived, game-agnostic description of one candidate."""
    n = arr.shape[0]
    area = float(n * n)
    is_click = float(prim[0] == "click")
    f = [is_click, rank / 12.0, 1.0 / max(len(available), 1)]
    if prim[0] == "click":
        r, c = int(prim[1]), int(prim[2])
        size, bbox, colour_n = index.get((r, c), (1, 1, 1))
        f += [
            size / area,
            size / max(bbox, 1),  # compactness
            colour_n / area,
            abs(r - n / 2) / n,
            abs(c - n / 2) / n,
            min(r, n - 1 - r) / n,  # distance to nearest edge
            min(c, n - 1 - c) / n,
            0.0,
            0.0,
        ]
    else:
        a = int(prim[1])
        f += [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            a / 16.0,
            float(available.index(a)) / max(len(available), 1),
        ]
    return f


def build(game_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate-level rows for one game: (features, positive, decision index)."""
    data = np.load(LABELS / f"{game_id}.npz")
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    env = arc.make(game_id)
    if env is None:
        raise RuntimeError("env-unavailable")
    fd = env.reset()
    game = env._game

    X: list[list[float]] = []
    y: list[int] = []
    group: list[int] = []
    for i in range(len(data["kind"])):
        arr = M.grid(fd)
        kind = int(data["kind"][i])
        win = (
            (int(data["row"][i]), int(data["col"][i]))
            if kind == CLICK_ID
            else int(kind)
        )
        mask = int(data["avail"][i])
        available = [a for a in range(16) if mask >> a & 1]
        goal = M.grid(M.act(copy.deepcopy(game), win))
        index = cell_index(arr)
        prims = proposals(arr, available, box=field_box(arr))
        for rank, prim in enumerate(prims):
            move = (int(prim[1]), int(prim[2])) if prim[0] == "click" else int(prim[1])
            same = np.array_equal(M.grid(M.act(copy.deepcopy(game), move)), goal)
            X.append(features(prim, rank, arr, available, index))
            y.append(int(same))
            group.append(i)
        fd = M.act(game, win)
    return np.asarray(X, float), np.asarray(y, int), np.asarray(group, int)


def top1(scores: np.ndarray, y: np.ndarray, group: np.ndarray) -> tuple[int, int]:
    """Decisions where the highest-scored candidate is a winning one."""
    hit = seen = 0
    for g in np.unique(group):
        m = group == g
        if not y[m].any():  # uncoverable decision; no ranking can win it
            continue
        seen += 1
        hit += int(y[m][int(np.argmax(scores[m]))] == 1)
    return hit, seen


@app.command()
def main(
    games: str = typer.Option("", help="Comma-separated subset; default all labelled."),
) -> None:
    """Fit per fold on every other game, score on the held-out one."""
    require_starter()
    ids = games.split(",") if games else sorted(p.stem for p in LABELS.glob("*.npz"))
    logger.info("building candidate rows for {} games", len(ids))
    per_game = {}
    for gid in ids:
        try:
            per_game[gid] = build(gid)
        except Exception as exc:
            logger.error("{}: {}: {}", gid, type(exc).__name__, exc)
    if len(per_game) < 2:
        raise typer.Exit(1)

    base_hit = base_n = mod_hit = mod_n = 0
    logger.info(
        "{:6} {:>9} {:>12} {:>12}", "held", "decisions", "baseline@1", "model@1"
    )
    for held, (Xh, yh, gh) in per_game.items():
        rest = [v for k, v in per_game.items() if k != held]
        Xtr = np.vstack([r[0] for r in rest])
        ytr = np.concatenate([r[1] for r in rest])
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(Xtr, ytr)
        # Baseline is the shipped order: rank 0 wins, so score by -rank.
        b_hit, n = top1(-Xh[:, 1], yh, gh)
        m_hit, _ = top1(model.decision_function(Xh), yh, gh)
        base_hit += b_hit
        mod_hit += m_hit
        base_n += n
        mod_n += n
        logger.info(
            "{:6} {:9} {:11.0%} {:11.0%}", held, n, b_hit / max(n, 1), m_hit / max(n, 1)
        )
    verdict = logger.success if mod_hit > base_hit else logger.error
    verdict(
        "leave-one-game-out recall@1: baseline {}/{} ({:.0%}) -> model {}/{} ({:.0%})",
        base_hit,
        base_n,
        base_hit / base_n,
        mod_hit,
        mod_n,
        mod_hit / mod_n,
    )


if __name__ == "__main__":
    app()
