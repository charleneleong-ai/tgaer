"""Pure-geometry helpers and BFS planner for ARC-AGI-3 grid games.

This module holds everything that is stateless and game-topology-agnostic:
cell finders, connected-component analysis, the play-field bounding box, and
the rigid-footprint BFS planner. The ``Semantics`` dataclass + ``LS20_DEFAULT``
encode which cell values are avatar / keys / door / walls for a given game
family, so a future VL "scientist" can supply per-game semantics without
touching the planner logic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

GREEN = 3
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
Box = tuple[np.ndarray, np.ndarray]  # (top-left, bottom-right) of the play-field


@dataclass(frozen=True)
class Semantics:
    avatar: int
    keys: tuple[int, ...]
    door: int
    walls: tuple[int, ...]
    verb: str  # "navigate" | "press"


LS20_DEFAULT = Semantics(avatar=12, keys=(0, 1), door=9, walls=(4, 11), verb="navigate")


def cells(arr: np.ndarray, v: int) -> np.ndarray:
    return np.argwhere(arr == v)


def components(arr: np.ndarray, values: tuple[int, ...]) -> list[np.ndarray]:
    mask = np.isin(arr, values)
    seen = np.zeros_like(mask, bool)
    out = []
    for r0, c0 in np.argwhere(mask):
        if seen[r0, c0]:
            continue
        q, comp = deque([(r0, c0)]), []
        seen[r0, c0] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in NBRS:
                r2, c2 = r + dr, c + dc
                if (
                    0 <= r2 < arr.shape[0]
                    and 0 <= c2 < arr.shape[1]
                    and mask[r2, c2]
                    and not seen[r2, c2]
                ):
                    seen[r2, c2] = True
                    q.append((r2, c2))
        out.append(np.array(comp))
    return out


def field_box(arr: np.ndarray) -> Box:
    g = cells(arr, GREEN)
    return (g.min(0), g.max(0)) if len(g) else (np.zeros(2), np.array(arr.shape))


def in_field(centroid: np.ndarray, box: Box, pad: int = 4) -> bool:
    lo, hi = box
    return bool((centroid >= lo - pad).all() and (centroid <= hi + pad).all())


def find_role(arr: np.ndarray, values: tuple[int, ...], box: Box) -> list[np.ndarray]:
    """Component centroids of `values` that fall inside the play-field `box`."""
    return [c.mean(0) for c in components(arr, values) if in_field(c.mean(0), box)]


class Planner:
    """BFS over the avatar's rigid footprint on its learned move lattice. The
    avatar is a multi-cell block, so a state is its top-left corner and a move is
    legal only if the destination centre cell is non-wall and not known-blocked
    (the game adjudicates the rest). Arrival = footprint covers the goal cell."""

    def __init__(
        self,
        arr: np.ndarray,
        footprint: np.ndarray,
        delta: dict[int, np.ndarray],
        walls: tuple[int, ...],
    ) -> None:
        self.arr, self.fp, self.delta, self.walls = arr, footprint, delta, walls
        self.coff = tuple(int(round(x)) for x in footprint.mean(0))
        self.blocked: set[tuple[int, int]] = set()

    def _ok(self, tl) -> bool:
        if tl in self.blocked:
            return False
        r, c = tl[0] + self.coff[0], tl[1] + self.coff[1]
        return (
            0 <= r < self.arr.shape[0]
            and 0 <= c < self.arr.shape[1]
            and self.arr[r, c] not in self.walls
        )

    def _cover(self, tl, g) -> int:
        return int(
            min(abs(tl[0] + dr - g[0]) + abs(tl[1] + dc - g[1]) for dr, dc in self.fp)
        )

    def path(self, tl0, goal) -> list[int] | None:
        s = (int(tl0[0]), int(tl0[1]))
        g = (int(round(goal[0])), int(round(goal[1])))
        steps = {
            a: (int(round(d[0])), int(round(d[1])))
            for a, d in self.delta.items()
            if int(round(d[0])) or int(round(d[1]))
        }
        prev, seen, q = {}, {s}, deque([s])
        best, bestd = s, self._cover(s, g)
        while q:
            node = q.popleft()
            if self._cover(node, g) <= 1:
                return self._trace(prev, s, node)
            for aid, (dr, dc) in steps.items():
                nxt = (node[0] + dr, node[1] + dc)
                if nxt not in seen and self._ok(nxt):
                    seen.add(nxt)
                    prev[nxt] = (node, aid)
                    q.append(nxt)
                    if (d := self._cover(nxt, g)) < bestd:
                        best, bestd = nxt, d
        return self._trace(prev, s, best) if best != s else None

    @staticmethod
    def _trace(prev, start, node) -> list[int]:
        acts = []
        while node != start:
            node, aid = prev[node]
            acts.append(aid)
        return acts[::-1]
