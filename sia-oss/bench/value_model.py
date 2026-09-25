#!/usr/bin/env python3
"""Can a value model learned from earlier levels rank actions on a later one?

`oracle_rank.py` fitted a ranker across games and it did not transfer: an action
id means something different in every game. The 6.71% Preview agent avoids that
by training **per game, in-episode** — it back-labels a cleared level with
distance-to-win over its own state graph and retrains, then plays the next level
with it. So the honest offline test is leave-one-*level*-out inside one game.

This replays the explorer, rebuilds the transition graph it walked, back-labels
each cleared level by shortest distance to the winning state, trains on the
earlier levels and scores the held-out one. Baseline is the shipped proposal
order, which is what `_choose` consumes today (recall@1 = 18% on the oracle
labels).

Only actions actually taken carry a distance — the same information the agent
has online, so nothing here is privileged.

**Known flaw, read the numbers with it.** `_choose` plays `untested[0]`, so
proposal rank *is* visit order: rank 0 is tried on the first visit to a state,
rank k on the k-th. The episode meanwhile moves toward the win, so a later visit
is genuinely closer to it. Rank is therefore anti-correlated with
distance-to-win by construction — which is why lp85 scores 0/215 for *both* arms
against a 32% chance floor. Rank is unusable here, as a feature and as a
baseline; compare against chance instead. tu93 avoids the worst of it and reads
model 34% against 27% chance (1.8 sd, n=102) — suggestive, not established.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
)
import oracle_rank as R  # noqa: E402

from tgaer.agents.arc_agi3_explorer import (  # noqa: E402
    frame_signature,
    proposals,
)
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    load_agent_class,
    play,
    require_starter,
)

os.chdir(REPO)

app = typer.Typer(add_completion=False)
# Games the explorer clears more than one level of, so there is an earlier level
# to learn from and a later one to be scored on.
MULTI_LEVEL = "lp85,tu93,ar25,m0r0"


def rollout(game_id: str, max_steps: int, seed: int) -> list[dict[str, Any]]:
    """Replay the explorer, recording the board, action and level of every step."""
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    steps: list[dict[str, Any]] = []

    def hook(step: int, observation: Any, env: Any, actor: Any) -> None:
        frame = (observation or {}).get("frame") or []
        if not frame:
            return
        arr = np.asarray(frame[-1], dtype=np.int16)
        trace = getattr(actor, "trace", {}) or {}
        # The agent keys on the *settled* board inside its own field box — the
        # chrome mask that flattens self-animating cells. Grouping by the raw
        # board measures a state space the agent never sees.
        settled = actor._settled(arr)
        field = actor._field(arr)
        steps.append(
            {
                "arr": arr,
                "settled": settled,
                "sig": frame_signature(settled, field),
                "prim": trace.get("prim"),
                "level": int((observation or {}).get("levels_completed", 0)),
                "available": list((observation or {}).get("available_actions") or []),
                # The agent's own ordering, with its learned goal values and its
                # seeded salt. Recomputing with the defaults gives a different
                # list, and any action missing from it collapses to one rank.
                "order": proposals(
                    arr,
                    list((observation or {}).get("available_actions") or []),
                    actor._goal_values,
                    box=field,
                    salt=actor._salt,
                ),
            }
        )

    play(
        load_agent_class(None, "explorer"),
        game_id,
        arc,
        None,
        max_steps,
        seed=seed,
        on_step=hook,
    )
    return steps


def back_label(seg: list[dict[str, Any]]) -> dict[int, float]:
    """Shortest distance to the level-winning state, per step index in ``seg``.

    Edges come from the transitions actually walked. The winning state is the
    last board of the segment, which is the one the level-up was observed from.
    """
    adj: dict[Any, list[tuple[int, Any]]] = {}
    for i in range(len(seg) - 1):
        adj.setdefault(seg[i]["sig"], []).append((i, seg[i + 1]["sig"]))
    goal = seg[-1]["sig"]

    # Reverse BFS over signatures, then map back to the step that reaches them.
    rev: dict[Any, list[Any]] = {}
    for src, edges in adj.items():
        for _, dst in edges:
            rev.setdefault(dst, []).append(src)
    dist_sig: dict[Any, float] = {goal: 0.0}
    q = deque([goal])
    while q:
        node = q.popleft()
        for src in rev.get(node, []):
            if src not in dist_sig:
                dist_sig[src] = dist_sig[node] + 1
                q.append(src)

    out: dict[int, float] = {}
    for i in range(len(seg) - 1):
        nxt = dist_sig.get(seg[i + 1]["sig"])
        if nxt is not None:
            out[i] = nxt + 1  # cost of taking this action, then the rest
    return out


def state_features(arr: np.ndarray) -> list[float]:
    """A compact description of the board itself.

    `oracle_rank.features` describes the *action* — for a click it reads the
    cell under it, for a simple action it carries only the id. So within one
    state every simple action looks alike apart from its id, and a model can
    learn "action 2 is usually good" but never "in this state, action 2". These
    let it condition on the board.
    """
    n = float(arr.size)
    hist = np.bincount(arr.ravel().astype(np.int64), minlength=16)[:16] / n
    bg = int(np.bincount(arr.ravel().astype(np.int64)).argmax())
    fg = np.argwhere(arr != bg)
    if len(fg):
        centre = [
            float(fg[:, 0].mean()) / arr.shape[0],
            float(fg[:, 1].mean()) / arr.shape[1],
        ]
        spread = [
            float(fg[:, 0].std()) / arr.shape[0],
            float(fg[:, 1].std()) / arr.shape[1],
        ]
    else:
        centre, spread = [0.0, 0.0], [0.0, 0.0]
    return [*hist.tolist(), len(fg) / n, *centre, *spread]


def rows(seg: list[dict[str, Any]], labels: dict[int, float]) -> tuple[np.ndarray, ...]:
    """(features, distance, state-group, proposal-rank) per labelled step."""
    X: list[list[float]] = []
    y: list[float] = []
    g: list[int] = []
    ranks: list[float] = []
    by_sig: dict[Any, int] = {}
    # One row per distinct (state, action). Frontier routing re-walks known
    # edges, so the same pair recurs many times — counting each visit inflates
    # the group and has both arms ranking duplicates of one another.
    seen_pair: set[tuple[Any, Any]] = set()
    for i, d in labels.items():
        step = seg[i]
        prim = step["prim"]
        if not prim:
            continue
        pair = (step["sig"], tuple(prim))
        if pair in seen_pair:
            continue
        seen_pair.add(pair)
        arr = step["arr"]
        prim = tuple(prim)
        # Where the agent's own ordering put this action. It is both a feature
        # and the baseline the model has to beat, so it must be the real rank.
        order = step["order"]
        if prim not in order:
            continue  # not rankable by the order the agent actually used
        rank = order.index(prim)
        # Rank is poison here and is kept only to be reported, never learned
        # from: `_choose` plays untested[0], so rank IS visit order, and the
        # episode moves toward the win — a later visit is genuinely closer, so
        # rank is anti-correlated with distance by construction.
        feat = R.features(prim, 0, arr, step["available"], R.cell_index(arr))
        X.append(feat + state_features(arr))
        y.append(d)
        ranks.append(float(rank))
        g.append(by_sig.setdefault(step["sig"], len(by_sig)))
    return (
        np.asarray(X, float),
        np.asarray(y, float),
        np.asarray(g, int),
        np.asarray(ranks, float),
    )


def top1(
    pred: np.ndarray, true: np.ndarray, group: np.ndarray, seed: int = 0
) -> tuple[int, int]:
    """States where the best-predicted action is genuinely the closest to the win.

    Ties are broken at random, not by position. Rows arrive in the order the
    agent first tried each action, and that order is anti-correlated with
    distance-to-win, so `argmin`'s first-index bias scores a model that predicts
    a constant at *zero* rather than at chance — lp85 read 0/215 against a 32%
    floor for exactly that reason.
    """
    rng = np.random.default_rng(seed)
    hit = seen = 0
    for gid in np.unique(group):
        m = group == gid
        if m.sum() < 2:  # nothing to rank
            continue
        seen += 1
        p = pred[m]
        best = np.flatnonzero(p == p.min())
        pick = int(rng.choice(best))
        hit += int(true[m][pick] == true[m].min())
    return hit, seen


@app.command()
def main(
    games: str = typer.Option(MULTI_LEVEL, help="Comma-separated game ids."),
    max_steps: int = typer.Option(2500),
    seed: int = typer.Option(0),
) -> None:
    """Train on a game's earlier levels, score it on its last cleared one."""
    require_starter()
    tot_hit = tot_seen = base_hit = 0
    for game_id in games.split(","):
        try:
            steps = rollout(game_id, max_steps, seed)
        except Exception as exc:
            logger.error("{}: {}: {}", game_id, type(exc).__name__, exc)
            continue
        levels = sorted({s["level"] for s in steps})
        segs = [[s for s in steps if s["level"] == lv] for lv in levels]
        # The last segment is the one that never cleared, so it has no winning
        # state to label; drop it.
        segs = [s for s in segs[:-1] if len(s) > 2]
        if len(segs) < 2:
            logger.warning(
                "{}: only {} cleared level(s), cannot split", game_id, len(segs)
            )
            continue

        built = [rows(s, back_label(s)) for s in segs]
        Xtr = np.vstack([b[0] for b in built[:-1]])
        ytr = np.concatenate([b[1] for b in built[:-1]])
        Xte, yte, gte, rte = built[-1]
        if len(Xte) == 0 or len(Xtr) == 0:
            logger.warning("{}: no labelled rows", game_id)
            continue

        model = GradientBoostingRegressor(random_state=0)
        model.fit(Xtr, ytr)
        hit, seen = top1(model.predict(Xte), yte, gte)
        # Rank cannot be a baseline here (see rows()); chance is the floor.
        bhit, _ = top1(rte, yte, gte)
        tot_hit += hit
        tot_seen += seen
        base_hit += bhit
        sizes = np.array([(gte == k).sum() for k in np.unique(gte)])
        chance = float(np.mean(1.0 / sizes[sizes >= 2])) if (sizes >= 2).any() else 0.0
        logger.info(
            "{:6} train {:5} rows | held-out: model {}/{} ({:3.0%})  "
            "baseline {:3.0%}  chance {:3.0%}",
            game_id,
            len(Xtr),
            hit,
            seen,
            hit / max(seen, 1),
            bhit / max(seen, 1),
            chance,
        )
    if tot_seen:
        logger.success(
            "held-out levels: model {}/{} ({:.0%}) vs baseline {}/{} ({:.0%})",
            tot_hit,
            tot_seen,
            tot_hit / tot_seen,
            base_hit,
            tot_seen,
            base_hit / tot_seen,
        )


if __name__ == "__main__":
    app()
