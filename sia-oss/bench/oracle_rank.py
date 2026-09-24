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


def state_features(arr: np.ndarray) -> list[float]:
    """A compact description of the board itself.

    `features` below describes the *action*: for a click it reads the cell under
    it, for a simple action it carries only the id. So within one state every
    simple action looks alike apart from its id, and a model can learn "action 2
    is usually good" but never "in this state, action 2". These let it condition
    on the board.
    """
    n = float(arr.size)
    hist = np.bincount(arr.ravel().astype(np.int64), minlength=16)[:16] / n
    bg = int(np.bincount(arr.ravel().astype(np.int64)).argmax())
    fg = np.argwhere(arr != bg)
    if len(fg):
        pos = [
            float(fg[:, 0].mean()) / arr.shape[0],
            float(fg[:, 1].mean()) / arr.shape[1],
            float(fg[:, 0].std()) / arr.shape[0],
            float(fg[:, 1].std()) / arr.shape[1],
        ]
    else:
        pos = [0.0, 0.0, 0.0, 0.0]
    return [*hist.tolist(), len(fg) / n, *pos]


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


def build(game_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Candidate rows for one game: (features, positive, decision, level)."""
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
    level: list[int] = []
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
            X.append(features(prim, rank, arr, available, index) + state_features(arr))
            y.append(int(same))
            group.append(i)
            level.append(int(data["level"][i]))
        fd = M.act(game, win)
    return (
        np.asarray(X, float),
        np.asarray(y, int),
        np.asarray(group, int),
        np.asarray(level, int),
    )


def top1(scores: np.ndarray, y: np.ndarray, group: np.ndarray) -> tuple[int, int]:
    """Decisions where the highest-scored candidate is a winning one."""
    rng = np.random.default_rng(0)
    hit = seen = 0
    for g in np.unique(group):
        m = group == g
        if not y[m].any():  # uncoverable decision; no ranking can win it
            continue
        seen += 1
        # Break ties at random: candidate order is the proposal order, so
        # argmax's first-index bias would silently score a constant predictor
        # as though it had learned the shipped ranking.
        s_g = scores[m]
        best = np.flatnonzero(s_g == s_g.max())
        hit += int(y[m][int(rng.choice(best))] == 1)
    return hit, seen


def chance(y: np.ndarray, group: np.ndarray) -> float:
    """Recall@1 from picking uniformly among a decision's candidates.

    The floor every arm has to clear. Without it a ranker that has learned
    nothing still reports a respectable number wherever candidate lists are
    short.
    """
    rates = []
    for g in np.unique(group):
        m = group == g
        if not y[m].any():
            continue
        rates.append(float(y[m].sum()) / int(m.sum()))
    return float(np.mean(rates)) if rates else 0.0


def within_game(per_game: dict[str, tuple]) -> None:
    """Train on a game's earlier levels and score its last — the 6.71% agent's
    setting, on labels that never came from the agent's own wandering.

    Leave-one-*game*-out cannot work for simple actions: an id means something
    different in every game. Within one game it can, which is why that agent
    retrains per episode rather than shipping a prior.
    """
    hit = seen = base = 0
    floors: list[tuple[float, int]] = []
    logger.info(
        "{:6} {:>6} {:>9} {:>12} {:>12} {:>9}",
        "game",
        "levels",
        "decisions",
        "baseline@1",
        "model@1",
        "chance",
    )
    for game_id, (X, y, g, lv) in per_game.items():
        levels = sorted(set(lv.tolist()))
        if len(levels) < 2:
            continue
        last = levels[-1]
        tr, te = lv != last, lv == last
        if not y[tr].any() or not y[te].any():
            continue
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X[tr], y[tr])
        m_hit, n = top1(model.decision_function(X[te]), y[te], g[te])
        b_hit, _ = top1(-X[te][:, 1], y[te], g[te])
        floor = chance(y[te], g[te])
        hit += m_hit
        base += b_hit
        seen += n
        floors.append((floor, n))
        logger.info(
            "{:6} {:6} {:9} {:11.0%} {:11.0%} {:9.0%}",
            game_id,
            len(levels),
            n,
            b_hit / max(n, 1),
            m_hit / max(n, 1),
            floor,
        )
    if not seen:
        logger.warning("no game has two labelled levels to split")
        return
    pooled = sum(f * n for f, n in floors) / max(sum(n for _, n in floors), 1)
    line = logger.success if hit > base else logger.error
    line(
        "within-game held-out level: baseline {}/{} ({:.0%}) -> model {}/{} "
        "({:.0%}), against a {:.0%} chance floor",
        base,
        seen,
        base / seen,
        hit,
        seen,
        hit / seen,
        pooled,
    )


@app.command()
def main(
    games: str = typer.Option("", help="Comma-separated subset; default all labelled."),
    within: bool = typer.Option(
        True, help="Also split within each game, train on earlier levels."
    ),
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
    floors: list[tuple[float, int]] = []
    logger.info(
        "{:6} {:>9} {:>12} {:>12} {:>9}",
        "held",
        "decisions",
        "baseline@1",
        "model@1",
        "chance",
    )
    for held, (Xh, yh, gh, _lv) in per_game.items():
        rest = [v for k, v in per_game.items() if k != held]
        Xtr = np.vstack([r[0] for r in rest])
        ytr = np.concatenate([r[1] for r in rest])
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(Xtr, ytr)
        # Baseline is the shipped order: rank 0 wins, so score by -rank.
        b_hit, n = top1(-Xh[:, 1], yh, gh)
        m_hit, _ = top1(model.decision_function(Xh), yh, gh)
        floor = chance(yh, gh)
        base_hit += b_hit
        mod_hit += m_hit
        base_n += n
        mod_n += n
        logger.info(
            "{:6} {:9} {:11.0%} {:11.0%} {:9.0%}",
            held,
            n,
            b_hit / max(n, 1),
            m_hit / max(n, 1),
            floor,
        )
        floors.append((floor, n))
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
    if within:
        within_game(per_game)


if __name__ == "__main__":
    app()
