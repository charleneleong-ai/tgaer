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
    KeyDoorController,
    LS20_DEFAULT,
    cells,
    components,
    field_box,
    find_role,
    to_action,
)
from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction

# Back-compat aliases so existing callers of _cells / _components still work.
_cells = cells
_components = components

AVATAR, DOOR_V = 12, 9
KEY_VS = (0, 1)  # key markers consumed on contact (black/blue)
WALL = (4, 11)  # yellow background + darkblue move-budget bar


def _keys(arr: np.ndarray) -> list[np.ndarray]:
    return find_role(arr, KEY_VS, field_box(arr))


def _door(arr: np.ndarray) -> np.ndarray | None:
    ds = find_role(arr, (DOOR_V,), field_box(arr))
    return ds[0] if ds else None


class PlannerArcAgi3Agent(Agent):
    def __init__(self, seed: int = 0, **_: Any) -> None:
        self._ctl = KeyDoorController()
        self._levels = 0
        self.last_reply: str | None = None

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        frame = obs.get("frame") or []
        if not frame:
            return to_action((obs.get("available_actions") or [1])[0])
        arr = np.asarray(frame[-1])
        levels = obs.get("levels_completed", self._levels)
        if levels != self._levels:
            self._levels = levels
            self._ctl.on_new_level()
        self._ctl.learn(arr, LS20_DEFAULT)
        aid = self._ctl.step(arr, LS20_DEFAULT, obs.get("available_actions") or [1])
        self.last_reply = f"[planner] act={aid}"
        return to_action(aid)

    # Proxy properties so Task 1's tests can still read a.delta / a.phase
    # without knowing about _ctl (both forms work; the brief updates the tests
    # to use a._ctl.* in this commit, so these proxies are belt-and-suspenders).
    @property
    def delta(self) -> dict[int, np.ndarray]:
        return self._ctl.delta

    @property
    def phase(self) -> str:
        return self._ctl.phase

    @phase.setter
    def phase(self, v: str) -> None:
        self._ctl.phase = v
