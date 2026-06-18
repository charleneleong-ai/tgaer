"""In-episode planning agent for ARC-AGI-3 keyboard games (the ``planner`` kind).

Frozen LLMs score 0/7 on these games because they reason per-step about a 64x64
board. This agent instead exploits the confirmed structure of the LS20-family
("Locksmith") games: each level is won by NAVIGATING the avatar onto the key
(a small marker that is consumed on contact) and then onto the door (a glyph in
a box) — pure movement, within a shared move budget. See the LS20 win-mechanic
notes; this is the controller half of the controller+scientist hybrid, with the
scientist's per-level semantics (which cell-values are key/door) currently
supplied by a cheap heuristic rather than a VL model.

Reactive and reset-free, so it drops straight into the standard single-episode
eval loop. It learns online from the observation stream:
  * the action->direction map (which action id translates the avatar, and by
    how much) — inferred by watching the avatar move, never assumed;
  * walls — any planned move the GAME refuses (avatar doesn't translate) is
    recorded as blocked and routed around. The game is the wall oracle, so the
    planner can never alias a wall the way a static green-only model would.

Goal per step is two-phase: collect the nearest key while any remain, then head
for the door. Paths are shortest-path over the avatar's rigid footprint on its
learned move lattice; identification is by connected-component analysis inside
the green play-field (the bottom-left HUD box sits outside it and is ignored).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    Planner,
    cells,
    components,
    field_box,
    find_role,
)
from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import (
    COMPLEX_ACTION_ID,
    GRID_SIZE,
    ArcAction,
)

AVATAR, DOOR_V = 12, 9
KEY_VS = (0, 1)  # key markers consumed on contact (black/blue)
WALL = (4, 11)  # yellow background + darkblue move-budget bar
MOVES = (1, 2, 3, 4)  # only directional actions navigate; 5/7 unused here

# Back-compat aliases so existing callers of _cells / _components still work.
_cells = cells
_components = components


def _keys(arr: np.ndarray) -> list[np.ndarray]:
    return find_role(arr, KEY_VS, field_box(arr))


def _door(arr: np.ndarray) -> np.ndarray | None:
    ds = find_role(arr, (DOOR_V,), field_box(arr))
    return ds[0] if ds else None


def _to_action(action_id: int) -> ArcAction:
    # This is a keyboard-game navigator; if only the coordinate action is
    # offered (a non-keyboard game) it must still send valid x/y or the API
    # rejects it — a centre click is a harmless no-op there.
    if action_id == COMPLEX_ACTION_ID:
        return ArcAction(id=action_id, x=GRID_SIZE // 2, y=GRID_SIZE // 2)
    return ArcAction(id=action_id)


class PlannerArcAgi3Agent(Agent):
    def __init__(self, seed: int = 0, **_: Any) -> None:
        self.delta: dict[int, np.ndarray] = {}  # learned action -> (dr,dc)
        self.blocked: set[tuple[int, int]] = set()
        self.probed: set[int] = set()
        self.phase = "key"  # sticky: "key" then "door"
        self.last_key_goal: np.ndarray | None = None
        self._levels = 0
        self._prev_action: int | None = None
        self._prev_tl: np.ndarray | None = None
        self.last_reply: str | None = None

    def _on_new_level(self, levels: int) -> None:
        """Each level is a fresh key->door board; reset goal state (but keep the
        learned move-map, which is constant across levels)."""
        if levels != self._levels:
            self._levels = levels
            self.phase = "key"
            self.last_key_goal = None
            self.blocked = set()

    def _grid(self, observation: Any) -> np.ndarray | None:
        frame = (observation or {}).get("frame") or []
        return np.asarray(frame[-1]) if frame else None

    def _learn(self, arr: np.ndarray) -> None:
        """Update the action->direction map and the wall set from the last move."""
        if (
            self._prev_action is None
            or self._prev_tl is None
            or self._prev_action not in MOVES
        ):
            return
        av = _cells(arr, AVATAR)
        if not len(av):
            return
        now = av.min(0)
        if (now != self._prev_tl).any():
            self.delta[self._prev_action] = now - self._prev_tl
        elif self._prev_action in self.delta:  # refused move = wall
            refused = self._prev_tl + self.delta[self._prev_action]
            self.blocked.add(tuple(refused.astype(int).tolist()))

    def _remember(self, arr: np.ndarray, action: int) -> None:
        av = _cells(arr, AVATAR)
        self._prev_action = action
        self._prev_tl = av.min(0) if len(av) else None
        self.probed.add(action)

    def act(self, observation: Any) -> ArcAction:
        avail = list((observation or {}).get("available_actions") or [1])
        arr = self._grid(observation)
        if arr is None:
            return _to_action(avail[0])
        self._on_new_level((observation or {}).get("levels_completed", self._levels))
        self._learn(arr)

        move_avail = [a for a in avail if a in MOVES]
        # Bootstrap: probe each directional action once to learn its vector.
        unprobed = [a for a in move_avail if a not in self.probed]
        if unprobed and len(self.delta) < len(move_avail):
            return self._emit(arr, unprobed[0], "probe")

        av = _cells(arr, AVATAR)
        if not len(av) or not self.delta:
            return self._emit(arr, self._fallback(avail, move_avail), "fallback")

        ks = _keys(arr)
        d = _door(arr)
        if ks:
            centre = av.mean(0)
            self.last_key_goal = min(ks, key=lambda c: abs(c - centre).sum())

        # Phase order: drive onto the key, then the door. The avatar is a rigid
        # block on a 5-cell lattice, so it may only reach the key's edge (cover
        # <=1) — an empty plan there means "as close as movement allows", i.e.
        # the key is reached. Commit to the door THAT step rather than stepping
        # off (which livelocks on the key tile).
        fp = (av - av.min(0)).astype(int)
        if self.phase == "key" and self.last_key_goal is not None:
            path = self._plan(arr, fp, av.min(0), self.last_key_goal)
            if path:
                return self._emit(
                    arr, path[0], f"key->{tuple(int(x) for x in self.last_key_goal)}"
                )
            self.phase = "door"  # reached the key — switch to the door

        if d is not None:
            path = self._plan(arr, fp, av.min(0), d)
            if path:
                return self._emit(arr, path[0], f"door->{tuple(int(x) for x in d)}")
        return self._emit(
            arr, self._fallback(avail, move_avail), f"{self.phase}:no-path"
        )

    @staticmethod
    def _fallback(avail: list[int], move_avail: list[int]) -> int:
        if move_avail:
            return move_avail[0]
        keyboard = [a for a in avail if a != COMPLEX_ACTION_ID]
        return keyboard[0] if keyboard else COMPLEX_ACTION_ID

    def _plan(
        self, arr: np.ndarray, fp: np.ndarray, tl: np.ndarray, goal: np.ndarray
    ) -> list[int] | None:
        for walls in (WALL, (11,)):  # green-only first, yellow-passable fallback
            planner = Planner(arr, fp, self.delta, walls)
            planner.blocked = self.blocked
            if path := planner.path(tl, goal):
                return path
        return None

    def _emit(self, arr: np.ndarray, action: int, why: str) -> ArcAction:
        self.last_reply = f"[planner] act={action} {why}"
        self._remember(arr, action)
        return _to_action(action)
