#!/usr/bin/env python3
"""Does a coarse effect model predict what an action does, better than what ships?

Measurement only: it fits nothing into the agent and changes nothing under `src/`.
Each recorded transition is predicted *before* it is learned from (prequential), so
every number is out-of-sample without needing a split.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

# (level_advanced, avatar_delta, appeared, vanished, churn_bucket)
Effect = tuple[bool, tuple[int, int] | None, frozenset[int], frozenset[int], int]
# (signature, action, colour, size_bucket)
Ctx = tuple[Any, int, int | None, int | None]
Key = tuple[Any, ...]

CLICK_ACTION = 6  # the API's click is action id 6; not a colour, so not roster-fitted


def bucket(n: int) -> int:
    """0 is reserved for "none"; everything else is a log2 band."""
    return 0 if n <= 0 else 1 + int(math.log2(n))


def centroid(arr: np.ndarray, value: int) -> tuple[float, float] | None:
    cells = np.argwhere(arr == value)
    if not len(cells):
        return None
    return float(cells[:, 0].mean()), float(cells[:, 1].mean())


def effect_signature(
    prev: np.ndarray, cur: np.ndarray, level_advanced: bool, avatar: int | None
) -> Effect:
    """A coarse, hashable description of what one action did.

    Callers pass *settled* frames: chrome must be masked before it gets here, since
    this cannot enforce its own caller.
    """
    before = {int(v) for v in np.unique(prev)}
    after = {int(v) for v in np.unique(cur)}
    delta: tuple[int, int] | None = None
    if avatar is not None:
        p, c = centroid(prev, avatar), centroid(cur, avatar)
        if p is not None and c is not None:
            delta = (int(round(c[0] - p[0])), int(round(c[1] - p[1])))
    changed = int(np.count_nonzero(prev != cur)) if prev.shape == cur.shape else -1
    return (
        bool(level_advanced),
        delta,
        frozenset(after - before),
        frozenset(before - after),
        bucket(changed),
    )


def class_keys(ctx: Ctx) -> list[Key]:
    """Most specific first. Back-off is what lets a rule fire somewhere new."""
    _, action, colour, size = ctx
    keys: list[Key] = []
    if colour is not None and size is not None:
        keys.append((action, colour, size))
    if colour is not None:
        keys.append((action, colour))
    keys.append((action,))
    return keys


def state_keys(ctx: Ctx) -> list[Key]:
    """Per state, no generalisation — the control arm."""
    sig, action, colour, size = ctx
    return [(sig, action, colour, size)]


def marginal_keys(ctx: Ctx) -> list[Key]:
    """One bucket for the whole game: predicting the commonest effect."""
    return [()]


class EffectModel:
    """Counts effects per key and predicts the mode, backing off when unseen."""

    def __init__(self, keys: Callable[[Ctx], list[Key]]) -> None:
        self._keys = keys
        self._counts: dict[Key, Counter[Effect]] = {}

    def predict(self, ctx: Ctx) -> tuple[Effect | None, int]:
        for level, key in enumerate(self._keys(ctx)):
            counter = self._counts.get(key)
            if counter:
                return counter.most_common(1)[0][0], level
        return None, -1

    def observe(self, ctx: Ctx, effect: Effect) -> None:
        for key in self._keys(ctx):
            self._counts.setdefault(key, Counter())[effect] += 1

    def seen(self, ctx: Ctx) -> bool:
        keys = self._keys(ctx)
        return bool(keys) and keys[0] in self._counts


def inert_view(effect: Effect) -> tuple[bool]:
    """What `_inert` knows: whether anything changed at all."""
    return (effect[4] == 0,)


class Prequential:
    """Predict with the model as it stands, score, then learn. No split, no leakage."""

    def __init__(self) -> None:
        self._arms: dict[str, EffectModel] = {
            "class": EffectModel(class_keys),
            "state": EffectModel(state_keys),
            "marginal": EffectModel(marginal_keys),
        }
        self._inert = EffectModel(class_keys)
        self._hits: Counter[str] = Counter()
        self._levels: Counter[int] = Counter()
        self._first_hits = 0
        self._first_n = 0
        self.n = 0

    def step(self, ctx: Ctx, effect: Effect) -> None:
        fresh = not self._arms["class"].seen(ctx)
        for name, model in self._arms.items():
            got, level = model.predict(ctx)
            if got == effect:
                self._hits[name] += 1
                if name == "class" and fresh:
                    self._first_hits += 1
            if name == "class":
                self._levels[level] += 1
        got_inert, _ = self._inert.predict(ctx)
        if got_inert is not None and inert_view(got_inert) == inert_view(effect):
            self._hits["inert"] += 1
        if fresh:
            self._first_n += 1

        for model in self._arms.values():
            model.observe(ctx, effect)
        self._inert.observe(ctx, effect)
        self.n += 1

    def report(self) -> dict[str, float]:
        n = max(self.n, 1)
        out = {k: self._hits[k] / n for k in ("class", "state", "inert", "marginal")}
        out["first_sighting"] = self._first_hits / max(self._first_n, 1)
        for level in (0, 1, 2):
            out[f"backoff_{level}"] = self._levels[level] / n
        return out


def component_size(arr: np.ndarray, row: int, col: int) -> int:
    """Cells in the 4-connected same-colour blob containing (row, col).

    Local rather than `arc_agi3_grid.components`, so this module imports nothing
    from the harness: that import pulls the vendored agents directory onto
    sys.path, whose own tests/ shadows this repo's.
    """
    colour = arr[row, col]
    stack = [(row, col)]
    seen = {(row, col)}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen:
                continue
            if (
                0 <= nr < arr.shape[0]
                and 0 <= nc < arr.shape[1]
                and arr[nr, nc] == colour
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen)


def is_transition(obs: dict[str, Any]) -> bool:
    """A respawn frame is a fresh level start, not a successor of the last action."""
    return not obs.get("terminal")


def context_of(prim: tuple[Any, ...], arr: np.ndarray, sig: Any) -> Ctx:
    """Key parts for one primitive: colour and size only when a click resolves.

    A click primitive is ``("click", row, col)`` — three elements, matching
    `to_arc`, which reads prim[1] as row and prim[2] as col.
    """
    if prim[0] != "click":
        return (sig, int(prim[1]), None, None)
    row, col = int(prim[1]), int(prim[2])
    if not (0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]):
        return (sig, CLICK_ACTION, None, None)
    return (
        sig,
        CLICK_ACTION,
        int(arr[row, col]),
        bucket(component_size(arr, row, col)),
    )


SUITE = "tu93,s5i5,ar25,sp80,ls20,m0r0,lp85,g50t"


def main(games: str = SUITE, max_steps: int = 600, seed: int = 0) -> None:
    """Score the effect model prequentially along the explorer's own trajectory.

    Every harness import lives here: at module level they would run during test
    collection and shadow this repo's tests/ with the vendored one.
    """
    import arc_runner
    from loguru import logger

    arc_runner._load_local(
        "tgaer.agents.arc_agi3_explorer",
        str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
    )
    from tgaer.agents.arc_agi3_explorer import frame_signature
    from tgaer.evaluation.arc_agi3_score_local import (
        OperationMode,
        arc_agi,
        load_agent_class,
        play,
        require_starter,
    )

    require_starter()
    logger.info(
        "{:6} {:>6} {:>7} {:>7} {:>7} {:>9} {:>6}",
        "game",
        "n",
        "class",
        "state",
        "inert",
        "marginal",
        "first",
    )
    for game_id in games.split(","):
        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(REPO / "environment_files"),
        )
        pre = Prequential()
        prev: dict[str, Any] = {}

        def hook(
            step: int, obs: Any, env: Any, actor: Any, _p: dict[str, Any] = prev
        ) -> None:
            obs = obs or {}
            frame = obs.get("frame") or []
            if not frame:
                return
            arr = np.asarray(frame[-1], dtype=np.int16)
            inner = getattr(actor, "_explorer", actor)
            levels = int(obs.get("levels_completed", 0))
            settled = inner._settled(arr)
            # Effects are measured on settled frames: chrome must not read as churn.
            # Contexts key on the raw board, because the colour under a click is what
            # the agent actually clicked.
            if _p and is_transition(obs):
                eff = effect_signature(
                    _p["settled"], settled, levels > _p["levels"], inner._det.avatar
                )
                pre.step(_p["ctx"], eff)
            prim = (getattr(actor, "trace", {}) or {}).get("prim")
            if prim is None:
                _p.clear()
                return
            sig = frame_signature(settled, inner._field(arr))
            _p.update(
                settled=settled, levels=levels, ctx=context_of(tuple(prim), arr, sig)
            )

        try:
            play(
                load_agent_class(None, "explorer"),
                game_id,
                arc,
                None,
                max_steps,
                seed=seed,
                on_step=hook,
            )
        except Exception as exc:  # a broken game must not abort the sweep
            logger.error("{}: {}: {}", game_id, type(exc).__name__, exc)
            continue
        r = pre.report()
        logger.info(
            "{:6} {:>6} {:>6.1%} {:>7.1%} {:>7.1%} {:>9.1%} {:>6.1%}",
            game_id,
            pre.n,
            r["class"],
            r["state"],
            r["inert"],
            r["marginal"],
            r["first_sighting"],
        )
        logger.info(
            "        back-off: specific {:.0%}  colour {:.0%}  action {:.0%}",
            r["backoff_0"],
            r["backoff_1"],
            r["backoff_2"],
        )


if __name__ == "__main__":
    import typer

    typer.run(main)
