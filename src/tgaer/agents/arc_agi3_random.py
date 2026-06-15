from __future__ import annotations

import random
from typing import Any

from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction, random_action


class RandomArcAgi3Agent(Agent):
    """Seeded random baseline: picks a uniformly random action from the
    observation's ``available_actions`` each step (random ``x``/``y`` for the
    coordinate action). Establishes a floor to measure guarded agents against.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: Any) -> ArcAction:
        available = (observation or {}).get("available_actions") or [1]
        return random_action(available, self._rng)
