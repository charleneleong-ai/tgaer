from __future__ import annotations

from collections import Counter

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    Semantics,
    avatar_is_sprite,
    cells,
    components,
    field_box,
    in_field,
)

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

    def move_lattice(self) -> dict[int, np.ndarray]:
        """The avatar's learned move map — per-action majority Δ, in the form the
        ``Planner`` consumes. Empty until the avatar is pinned; an action whose
        majority outcome is a refused (0,0) move drops out."""
        if self._avatar is None:
            return {}
        lattice: dict[int, np.ndarray] = {}
        for action, counter in self._deltas.get(self._avatar, {}).items():
            dr, dc = counter.most_common(1)[0][0]
            if dr or dc:
                lattice[action] = np.array([dr, dc])
        return lattice

    def observe(
        self, prev: np.ndarray | None, action: int | None, cur: np.ndarray, levels: int
    ) -> None:
        if prev is not None and action is not None:
            self._observe_motion(prev, action, cur)
            if self._avatar is not None:
                if levels > self._levels:
                    self._observe_door(prev, cur)
                else:
                    self._observe_key(prev, cur)
        self._levels = levels

    def _observe_motion(self, prev: np.ndarray, action: int, cur: np.ndarray) -> None:
        if self._avatar is not None:
            return
        shared = set(np.unique(prev)).intersection(np.unique(cur))
        # Accumulate deltas for all shared values this step.
        candidates: list[int] = []
        for v in shared:
            p = _single_in_field_centroid(prev, int(v))
            c = _single_in_field_centroid(cur, int(v))
            if p is None or c is None:
                continue
            delta = (int(round(c[0] - p[0])), int(round(c[1] - p[1])))
            self._deltas.setdefault(int(v), {}).setdefault(action, Counter())[
                delta
            ] += 1
            if self._is_controllable(int(v)):
                candidates.append(int(v))
        if not candidates:
            return
        # Prefer the most sprite-like candidate; fall back to smallest footprint.
        sprites = [v for v in candidates if avatar_is_sprite(cur, v)]
        pool = sprites if sprites else candidates
        self._avatar = min(pool, key=lambda v: len(cells(cur, v)))

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

    def _avatar_cells(self, arr: np.ndarray) -> np.ndarray:
        """All (row, col) positions of the avatar in arr; empty if avatar unknown."""
        if self._avatar is None:
            return np.empty((0, 2), dtype=int)
        return cells(arr, self._avatar)

    def _any_adjacent(self, av_cells: np.ndarray, r: int, c: int) -> bool:
        """True when (r, c) is Manhattan-distance ≤ 1 from ANY avatar cell."""
        return bool(
            np.any(np.abs(av_cells[:, 0] - r) + np.abs(av_cells[:, 1] - c) <= 1)
        )

    def _observe_key(self, prev: np.ndarray, cur: np.ndarray) -> None:
        av = self._avatar_cells(prev)
        if not len(av):
            return
        gone = set(np.unique(prev)) - set(np.unique(cur))
        for v in gone:
            if v in (3, 4, self._avatar):  # floor / wall / avatar are not keys
                continue
            near = any(self._any_adjacent(av, r, c) for r, c in cells(prev, int(v)))
            if near:
                self._keys.add(int(v))

    def _observe_door(self, prev: np.ndarray, cur: np.ndarray) -> None:
        av = self._avatar_cells(prev)
        if not len(av):
            return
        gone = set(np.unique(prev)) - set(np.unique(cur))
        for r, c in np.argwhere(prev != 3):  # non-floor cells adjacent to avatar
            v = int(prev[r, c])
            if v in (4, self._avatar) or v in self._keys:
                continue
            if v not in gone:  # must have vanished (not just a static decoration)
                continue
            if self._any_adjacent(av, r, c):
                self._door = v
                return

    def semantics(self, cold_start: Semantics) -> Semantics:
        return Semantics(
            avatar=self._avatar if self._avatar is not None else cold_start.avatar,
            keys=self.keys if self._keys else cold_start.keys,
            door=self._door if self._door is not None else cold_start.door,
            walls=cold_start.walls,
            verb=cold_start.verb,
        )


def _single_in_field_centroid(arr: np.ndarray, v: int) -> np.ndarray | None:
    box = field_box(arr)
    comps = [c for c in components(arr, (v,)) if in_field(c.mean(0), box)]
    return comps[0].mean(0) if len(comps) == 1 else None
