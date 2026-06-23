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

from tgaer.envs.arc_agi3.arc_agi3_api import COMPLEX_ACTION_ID, GRID_SIZE, ArcAction

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


def to_action(action_id: int) -> ArcAction:
    """Wrap an action-id int into an ``ArcAction``.

    ACTION6 (coordinate-click) requires x/y; a centre click is a harmless no-op
    in keyboard games. Every other id maps to a plain ``ArcAction``.
    """
    if action_id == COMPLEX_ACTION_ID:
        return ArcAction(id=action_id, x=GRID_SIZE // 2, y=GRID_SIZE // 2)
    return ArcAction(id=action_id)


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


def avatar_is_sprite(
    arr: np.ndarray, avatar: int, max_field_frac: float = 0.03
) -> bool:
    """True when `avatar` is a single in-field component small relative to the field.

    A player sprite is one compact piece; wall/floor structure is multi-part or large.
    """
    box = field_box(arr)
    comps = [c for c in components(arr, (avatar,)) if in_field(c.mean(0), box)]
    if len(comps) != 1:
        return False
    lo, hi = box
    field_area = float((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1))
    return len(comps[0]) <= max_field_frac * field_area


_MOVES = (1, 2, 3, 4)  # directional action ids — constant across LS20 games


class KeyDoorController:
    """Per-step navigation state and logic for LS20-family key→door games.

    Owns the online-learned move map (``delta``), wall memory (``blocked``),
    two-phase key→door goal tracking, and bootstrap probing. Parameterised by
    a ``Semantics`` so a VL scientist can supply per-game role values without
    touching this class.

    ``step`` returns an **action id** (int); the agent wraps it via ``to_action``.
    """

    def __init__(self) -> None:
        self.delta: dict[int, np.ndarray] = {}
        self.blocked: set[tuple[int, int]] = set()
        self.probed: set[int] = set()
        self.phase = "key"
        self.last_key_goal: np.ndarray | None = None
        self._prev_action: int | None = None
        self._prev_tl: np.ndarray | None = None
        self._progressed = False

    def on_new_level(self) -> None:
        """Reset goal state for a fresh level; keep the learned move map."""
        self.phase = "key"
        self.last_key_goal = None
        self.blocked = set()

    def made_progress(self) -> bool:
        """True if the last ``step`` advanced the avatar closer to the goal."""
        return self._progressed

    def learn(self, arr: np.ndarray, sem: Semantics) -> None:
        """Update delta + blocked from the last move."""
        if (
            self._prev_action is None
            or self._prev_tl is None
            or self._prev_action not in _MOVES
        ):
            return
        av = cells(arr, sem.avatar)
        if not len(av):
            return
        now = av.min(0)
        if (now != self._prev_tl).any():
            self.delta[self._prev_action] = now - self._prev_tl
        elif self._prev_action in self.delta:
            refused = self._prev_tl + self.delta[self._prev_action]
            self.blocked.add(tuple(refused.astype(int).tolist()))

    def _remember(self, arr: np.ndarray, action: int, sem: Semantics) -> None:
        av = cells(arr, sem.avatar)
        self._prev_action = action
        self._prev_tl = av.min(0) if len(av) else None
        self.probed.add(action)

    def _keys(self, arr: np.ndarray, sem: Semantics) -> list[np.ndarray]:
        return find_role(arr, sem.keys, field_box(arr))

    def _door(self, arr: np.ndarray, sem: Semantics) -> np.ndarray | None:
        ds = find_role(arr, (sem.door,), field_box(arr))
        return ds[0] if ds else None

    def _plan(
        self,
        arr: np.ndarray,
        fp: np.ndarray,
        tl: np.ndarray,
        goal: np.ndarray,
        sem: Semantics,
    ) -> list[int] | None:
        for walls in (sem.walls, (11,)):  # green-only first, yellow-passable fallback
            planner = Planner(arr, fp, self.delta, walls)
            planner.blocked = self.blocked
            if path := planner.path(tl, goal):
                return path
        return None

    @staticmethod
    def _fallback(avail: list[int], move_avail: list[int]) -> int:
        if move_avail:
            return move_avail[0]
        keyboard = [a for a in avail if a != COMPLEX_ACTION_ID]
        return keyboard[0] if keyboard else COMPLEX_ACTION_ID

    @staticmethod
    def _interaction(avail: list[int]) -> int:
        for a in (5, 7):
            if a in avail:
                return a
        return COMPLEX_ACTION_ID

    @staticmethod
    def _adjacent(av: np.ndarray, target: np.ndarray) -> bool:
        """True when the avatar footprint is adjacent to (covers within 1 of) ``target``."""
        fp = (av - av.min(0)).astype(int)
        tl = av.min(0)
        g = (int(round(target[0])), int(round(target[1])))
        return (
            min(abs(tl[0] + dr - g[0]) + abs(tl[1] + dc - g[1]) for dr, dc in fp) <= 1
        )

    def step(self, arr: np.ndarray, sem: Semantics, avail: list[int]) -> int:
        """Choose and return the next action id for ``navigate`` or ``press`` verbs."""
        self._progressed = False
        move_avail = [a for a in avail if a in _MOVES]

        # Bootstrap: probe each directional action once to learn its move vector.
        unprobed = [a for a in move_avail if a not in self.probed]
        if unprobed and len(self.delta) < len(move_avail):
            action = unprobed[0]
            self._remember(arr, action, sem)
            return action

        av = cells(arr, sem.avatar)
        if not len(av) or not self.delta:
            action = self._fallback(avail, move_avail)
            self._remember(arr, action, sem)
            return action

        ks = self._keys(arr, sem)
        d = self._door(arr, sem)
        if ks:
            centre = av.mean(0)
            self.last_key_goal = min(ks, key=lambda c: abs(c - centre).sum())

        fp = (av - av.min(0)).astype(int)
        tl = av.min(0)

        if sem.verb == "press":
            target = self.last_key_goal if ks and self.last_key_goal is not None else d
            if target is not None and self._adjacent(av, target):
                self._progressed = True
                return self._interaction(avail)

        if self.phase == "key" and self.last_key_goal is not None:
            path = self._plan(arr, fp, tl, self.last_key_goal, sem)
            if path:
                self._progressed = True
                action = path[0]
                self._remember(arr, action, sem)
                return action
            self.phase = "door"  # reached the key — commit to the door this step

        if d is not None:
            path = self._plan(arr, fp, tl, d, sem)
            if path:
                self._progressed = True
                action = path[0]
                self._remember(arr, action, sem)
                return action

        action = self._fallback(avail, move_avail)
        self._remember(arr, action, sem)
        return action


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
