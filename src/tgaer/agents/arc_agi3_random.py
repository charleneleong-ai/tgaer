from __future__ import annotations

import random
from typing import Any

from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction

GRID_SIZE = 64  # ARC-AGI-3 grids are fixed 64x64 (colour indices 0-15)
COMPLEX_ACTION_ID = 6  # ACTION6 carries (x, y) coordinates


class RandomArcAgi3Agent(Agent):
    """Seeded random baseline: picks a uniformly random action from the
    observation's ``available_actions`` each step (random ``x``/``y`` for the
    coordinate action). Establishes a floor to measure guarded agents against.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: Any) -> ArcAction:
        available = (observation or {}).get("available_actions") or [1]
        action_id = self._rng.choice(available)
        if action_id == COMPLEX_ACTION_ID:
            return ArcAction(
                id=action_id,
                x=self._rng.randrange(GRID_SIZE),
                y=self._rng.randrange(GRID_SIZE),
            )
        return ArcAction(id=action_id)
