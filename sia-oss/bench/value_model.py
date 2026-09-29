#!/usr/bin/env python3
"""Can a value model learned from earlier levels rank actions on a later one?

`oracle_rank.py` fitted a ranker across games and it did not transfer: an action
id means something different in every game. The 6.71% Preview agent avoids that
by training **per game, in-episode** — it back-labels a cleared level with
distance-to-win over its own state graph and retrains, then plays the next level
with it. So the honest offline test is leave-one-*level*-out inside one game.

This replays the explorer, rebuilds the transition graph it walked, back-labels
each cleared level by shortest distance to the winning state, and scores every
cleared level from a model of the levels before it — the order they become
available online. Only actions actually taken carry a distance, so nothing here
is privileged.

**Read it against chance, per fold.** Proposal rank is not a baseline: `_choose`
plays `untested[0]`, so rank is visit order, and the episode moves toward the win,
which anti-correlates rank with distance by construction. The floor is a constant
prediction's expected hits, and significance is a sign test over (game, seed)
episodes — states in one level share a board, and levels in one episode share a
trajectory and nested training sets, so neither is an independent trial.
"""

from __future__ import annotations

import gzip
import hashlib
import inspect
import itertools
import math
import os
import pickle
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import typer
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

import arc_runner  # noqa: E402

EXPLORER = REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"
arc_runner._load_local("tgaer.agents.arc_agi3_explorer", str(EXPLORER))
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
CACHE = REPO / "sia-oss" / "bench" / "runs" / "value_model_cache"
HARNESS = REPO / "src" / "tgaer" / "evaluation" / "arc_agi3_score_local.py"


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


def fingerprint(game_id: str) -> str:
    """Everything a trajectory depends on besides its seed and budget.

    The agent, the harness that plays it, the recorder that writes each step, the
    game's own files and the arc_agi package — change any and a cached replay is
    stale.
    """
    h = hashlib.sha256()
    game = REPO / "environment_files" / game_id
    for path in [EXPLORER, HARNESS, *sorted(p for p in game.rglob("*") if p.is_file())]:
        h.update(path.read_bytes())
    h.update(inspect.getsource(rollout).encode())
    h.update(str(getattr(arc_agi, "__version__", "")).encode())
    return h.hexdigest()[:12]


def cached_rollout(game_id: str, seed: int, max_steps: int) -> list[dict[str, Any]]:
    """``rollout``, persisted: the explorer is deterministic per seed, and the
    rollout is nearly all of this bench's cost."""
    path = CACHE / f"{game_id}_s{seed}_{max_steps}_{fingerprint(game_id)}.pkl.gz"
    if path.is_file():
        return pickle.loads(gzip.decompress(path.read_bytes()))
    steps = rollout(game_id, max_steps, seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(pickle.dumps(steps)))
    tmp.replace(path)  # a killed run must not leave a truncated cache behind
    return steps


def cleared_segments(steps: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The trajectory of each level that cleared, in order.

    The last level was still in progress when the rollout ended, so it has no
    winning state to back-label. Segments of two steps or fewer are too short
    to rank.
    """
    levels = sorted({s["level"] for s in steps})
    segs = [[s for s in steps if s["level"] == lv] for lv in levels]
    return [s for s in segs[:-1] if len(s) > 2]


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


class Table(NamedTuple):
    """One cleared level's labelled rows."""

    X: np.ndarray
    y: np.ndarray
    g: np.ndarray
    grids: np.ndarray
    prims: list[tuple]


def rows(seg: list[dict[str, Any]], labels: dict[int, float]) -> Table:
    """Features, distance, state group, board and action per labelled step."""
    X: list[list[float]] = []
    y: list[float] = []
    g: list[int] = []
    grids: list[np.ndarray] = []
    prims: list[tuple] = []
    by_sig: dict[Any, int] = {}
    # One row per distinct (state, action). Frontier routing re-walks known
    # edges, so the same pair recurs many times — counting each visit inflates
    # the group and has both arms ranking duplicates of one another.
    seen_pair: set[tuple[Any, Any]] = set()
    for i, d in labels.items():
        step = seg[i]
        if not step["prim"]:
            continue
        prim = tuple(step["prim"])
        pair = (step["sig"], prim)
        if pair in seen_pair:
            continue
        seen_pair.add(pair)
        arr = step["arr"]
        if prim not in step["order"]:
            continue  # kept so the row set matches earlier runs of this bench
        feat = R.features(prim, 0, arr, step["available"], R.cell_index(arr))
        X.append(feat + state_features(arr))
        y.append(d)
        g.append(by_sig.setdefault(step["sig"], len(by_sig)))
        # The board the agent keys on: its chrome mask flattens self-animating
        # cells, which tick with time and so would leak distance-to-win.
        grids.append(step["settled"])
        prims.append(prim)
    return Table(
        np.asarray(X, float),
        np.asarray(y, float),
        np.asarray(g, int),
        np.asarray(grids),
        prims,
    )


def top1(pred: np.ndarray, true: np.ndarray, group: np.ndarray) -> tuple[float, int]:
    """Expected hits from playing the best-predicted action, over rankable states.

    A hit is any action tied for the least distance. Where the prediction ties,
    the pick is uniform over its tied actions, so a state pays the share of them
    that are optimal — and a constant prediction scores chance by construction.
    """
    hit, seen = 0.0, 0
    for gid in np.unique(group):
        m = group == gid
        if m.sum() < 2:
            continue
        p, t = pred[m], true[m]
        hit += float((t[p == p.min()] == t.min()).mean())
        seen += 1
    return hit, seen


def sign_test(deltas: list[float]) -> float:
    """One-sided p that the model beats chance. Zero deltas carry no direction."""
    signs = [d for d in deltas if d != 0]
    wins = sum(d > 0 for d in signs)
    return sum(
        math.comb(len(signs), k) for k in range(wins, len(signs) + 1)
    ) / 2 ** len(signs)


Model = Callable[[list[Table], Table], np.ndarray]


def gbr(train: list[Table], test: Table) -> np.ndarray:
    """Gradient boosting over action features and 21 global board scalars."""
    model = GradientBoostingRegressor(random_state=0).fit(
        np.vstack([t.X for t in train]), np.concatenate([t.y for t in train])
    )
    return model.predict(test.X)


def conv(train: list[Table], test: Table) -> np.ndarray:
    """A small conv net over the board itself; see ``grid_encoder``."""
    # torch is in the optional bench group, so only this model may need it.
    import grid_encoder

    return grid_encoder.fit_predict(
        np.concatenate([t.grids for t in train]),
        [p for t in train for p in t.prims],
        np.concatenate([t.y for t in train]),
        test.grids,
        test.prims,
    )


MODELS: dict[str, Model] = {"gbr": gbr, "conv": conv}


class Fold(NamedTuple):
    """One held-out level, scored by every model — and by chance — on the same states."""

    level: int
    states: int
    hits: dict[str, float]


def score_fold(
    level: int, train: list[Table], test: Table, models: dict[str, Model]
) -> Fold | None:
    """Every model's expected hits on the held-out level, or None if nothing ranks."""
    train = [t for t in train if len(t.y)]
    if not train:
        return None
    chance, seen = top1(np.zeros_like(test.y), test.y, test.g)
    if not seen:
        return None
    hits = {n: top1(m(train, test), test.y, test.g)[0] for n, m in models.items()}
    return Fold(level, seen, {"chance": chance, **hits})


def score_episode(
    game_id: str, seed: int, max_steps: int, models: dict[str, Model]
) -> list[Fold]:
    """Every cleared level after the first, scored from the levels before it only.

    Online a level is back-labelled once it clears, so later ones cannot inform it.
    """
    steps = cached_rollout(game_id, seed, max_steps)
    built = [rows(s, back_label(s)) for s in cleared_segments(steps)]
    return [
        fold
        for k in range(1, len(built))
        if (fold := score_fold(k, built[:k], built[k], models)) is not None
    ]


def distinct(
    episodes: dict[tuple[str, int], list[Any]],
) -> dict[tuple[str, int], list[Any]]:
    """Episodes with repeated trajectories dropped, keeping the first of each.

    The explorer's seed barely varies play — all five tu93 seeds are identical —
    and a copy is not independent evidence. Scoring is deterministic, so equal
    fold results mean equal trajectories; ``repr`` is an exact hashable image.
    """
    first: dict[str, tuple[str, int]] = {}
    for key, folds in episodes.items():
        first.setdefault(repr(folds), key)
    return {key: episodes[key] for key in first.values()}


def report(episodes: dict[tuple[str, int], list[Fold]], names: list[str]) -> None:
    """Every scorer's hit rate, then each pair head to head over distinct episodes."""
    total = len(episodes)
    episodes = distinct(episodes)
    folds = [f for scored in episodes.values() for f in scored]
    states = sum(f.states for f in folds)
    per = {
        n: [sum(f.hits[n] for f in scored) for scored in episodes.values()]
        for n in names
    }
    logger.success(
        "{} distinct of {} episodes over {} games, {} folds, {} states | {}",
        len(episodes),
        total,
        len({g for g, _ in episodes}),
        len(folds),
        states,
        "  ".join(f"{n} {sum(per[n]) / states:.1%}" for n in names),
    )
    for a, b in itertools.combinations(names, 2):
        d = [y - x for x, y in zip(per[a], per[b])]
        logger.success(
            "{} vs {}: {} episodes better, {} worse, sign test p = {:.4f}",
            b,
            a,
            sum(x > 0 for x in d),
            sum(x < 0 for x in d),
            sign_test(d),
        )


@app.command()
def main(
    games: str = typer.Option("", help="Comma-separated game ids; default all 25."),
    max_steps: int = typer.Option(6000),
    seeds: int = typer.Option(5, help="Independent episodes per game."),
    model: str = typer.Option("gbr", help="Comma-separated, from: gbr, conv."),
) -> None:
    """Score every cleared level from the levels before it, over several seeds."""
    require_starter()
    if not games:
        # measure lists environment_files at import, which CI does not have.
        from measure import SUITE

        games = ",".join(SUITE)
    models = {n: MODELS[n] for n in model.split(",")}
    episodes: dict[tuple[str, int], list[Fold]] = {}
    for game_id in games.split(","):
        for seed in range(seeds):
            try:
                scored = score_episode(game_id, seed, max_steps, models)
            except Exception as exc:
                logger.error(
                    "{} seed {}: {}: {}", game_id, seed, type(exc).__name__, exc
                )
                continue
            for f in scored:
                logger.info(
                    "{:6} seed {} level {}: {} over {} states",
                    game_id,
                    seed,
                    f.level,
                    "  ".join(f"{n} {h / f.states:3.0%}" for n, h in f.hits.items()),
                    f.states,
                )
            if scored:
                episodes[(game_id, seed)] = scored
    if episodes:
        report(episodes, ["chance", *models])
    else:
        logger.warning("no scorable episodes")


if __name__ == "__main__":
    app()
