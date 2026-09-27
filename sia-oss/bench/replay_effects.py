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
