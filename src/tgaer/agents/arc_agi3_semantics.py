from __future__ import annotations

from collections import Counter

import numpy as np

from tgaer.agents.arc_agi3_grid import components, field_box, in_field

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

    @property
    def avatar(self) -> int | None:
        return self._avatar

    def observe(
        self, prev: np.ndarray | None, action: int | None, cur: np.ndarray, levels: int
    ) -> None:
        if prev is not None and action is not None:
            self._observe_motion(prev, action, cur)

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


def _single_in_field_centroid(arr: np.ndarray, v: int) -> np.ndarray | None:
    box = field_box(arr)
    comps = [c for c in components(arr, (v,)) if in_field(c.mean(0), box)]
    return comps[0].mean(0) if len(comps) == 1 else None
