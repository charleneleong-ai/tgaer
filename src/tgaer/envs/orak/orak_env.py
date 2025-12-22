from __future__ import annotations

from typing import Any, Optional

from tgaer.core.env_base import Environment, Transition
from tgaer.envs.orak.orak_api import OrakClient, OrakStepResult


class OrakEnvironment(Environment):
    """Stub ORAK environment wrapping OrakClient."""

    def __init__(self, client: OrakClient, mission_id: str) -> None:
        self._client = client
        self._mission_id = mission_id
        self._state: Any = None

    def reset(self, seed: Optional[int] = None) -> Any:
        self._state = self._client.start_mission(self._mission_id, seed=seed)
        return self._state

    def step(self, action: Any) -> Transition:
        result: OrakStepResult = self._client.apply_action(self._mission_id, action)
        self._state = result.state
        return Transition(
            state=result.state,
            action=action,
            reward=result.reward,
            done=result.done,
            info=result.info,
        )

    def render(self) -> Any:
        return self._client.render_state(self._mission_id, self._state)

    def task_id(self):
        return self._mission_id
