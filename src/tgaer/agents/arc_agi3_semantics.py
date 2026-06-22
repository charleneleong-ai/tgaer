from __future__ import annotations

from collections import Counter

import numpy as np

from tgaer.agents.arc_agi3_grid import cells, components, field_box, in_field

Delta = tuple[int, int]


class EmpiricalSemantics:
    """Derive avatar/key/door roles from the action→observation stream.

    The avatar is the value whose motion is *controllable* — a consistent,
    action-specific translation; a distractor that moves the same way regardless
    of action is not controllable and is never pinned.
    """

    def __init__(self, avatar_confirm: int = 2) -> None:
        self._avatar_confirm = avatar_confirm
        self._avatar: int | None = None
        # value -> action -> Counter of observed integer Δ tuples
        self._deltas: dict[int, dict[int, Counter[Delta]]] = {}
        self._keys: set[int] = set()
        self._door: int | None = None
        self._levels: int = 0

    @property
    def avatar(self) -> int | None:
        return self._avatar

    @property
    def keys(self) -> tuple[int, ...]:
        return tuple(sorted(self._keys))

    @property
    def door(self) -> int | None:
        return self._door

    def observe(
        self, prev: np.ndarray | None, action: int | None, cur: np.ndarray, levels: int
    ) -> None:
        if prev is not None and action is not None:
            self._observe_motion(prev, action, cur)
            if self._avatar is not None:
                if levels > self._levels:
                    self._observe_door(prev)
                else:
                    self._observe_key(prev, cur)
        self._levels = levels

    def _observe_motion(self, prev: np.ndarray, action: int, cur: np.ndarray) -> None:
        if self._avatar is not None:
            return
        shared = set(np.unique(prev)).intersection(np.unique(cur))
        for v in shared:
            p = _single_in_field_centroid(prev, int(v))
            c = _single_in_field_centroid(cur, int(v))
            if p is None or c is None:
                continue
            delta = (int(round(c[0] - p[0])), int(round(c[1] - p[1])))
            self._deltas.setdefault(int(v), {}).setdefault(action, Counter())[delta] += 1
            if self._is_controllable(int(v)):
                self._avatar = int(v)
                return

    def _is_controllable(self, v: int) -> bool:
        per_action = self._deltas[v]
        # consistency: some action has the SAME non-zero Δ at least avatar_confirm times
        consistent = any(
            d != (0, 0) and n >= self._avatar_confirm
            for counter in per_action.values()
            for d, n in counter.items()
        )
        # controllability: the per-action majority Δ takes ≥2 distinct values
        majorities = {counter.most_common(1)[0][0] for counter in per_action.values()}
        return consistent and len(majorities) >= 2

    def _avatar_cell(self, arr: np.ndarray) -> np.ndarray | None:
        rc = cells(arr, self._avatar) if self._avatar is not None else np.empty((0, 2))
        return rc.mean(0) if len(rc) else None

    def _observe_key(self, prev: np.ndarray, cur: np.ndarray) -> None:
        av = self._avatar_cell(prev)
        if av is None:
            return
        gone = set(np.unique(prev)) - set(np.unique(cur))
        for v in gone:
            if v in (3, 4, self._avatar):  # floor / wall / avatar are not keys
                continue
            near = any(abs(r - av[0]) + abs(c - av[1]) <= 1 for r, c in cells(prev, int(v)))
            if near:
                self._keys.add(int(v))

    def _observe_door(self, prev: np.ndarray) -> None:
        av = self._avatar_cell(prev)
        if av is None:
            return
        for r, c in np.argwhere(prev != 3):  # non-floor cells adjacent to avatar
            v = int(prev[r, c])
            if v in (4, self._avatar) or v in self._keys:
                continue
            if abs(r - av[0]) + abs(c - av[1]) <= 1:
                self._door = v
                return


def _single_in_field_centroid(arr: np.ndarray, v: int) -> np.ndarray | None:
    box = field_box(arr)
    comps = [c for c in components(arr, (v,)) if in_field(c.mean(0), box)]
    return comps[0].mean(0) if len(comps) == 1 else None
