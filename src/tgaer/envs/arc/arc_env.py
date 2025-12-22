from __future__ import annotations

from typing import Any, Optional

from tgaer.core.env_base import Environment, Transition


class ArcEnvironment(Environment):
    """Stub ARC environment wrapping a single ARC task object."""

    def __init__(self, task: Any) -> None:
        self._task = task
        self._state = None

    def reset(self, seed: Optional[int] = None) -> Any:
        self._state = getattr(self._task, "initial_state", lambda seed=None: None)(seed=seed)
        return self._state

    def step(self, action: Any) -> Transition:
        evaluate = getattr(self._task, "evaluate", lambda a: (0.0, True, {}))
        reward, done, info = evaluate(action)
        return Transition(state=None, action=action, reward=reward, done=done, info=info)

    def render(self) -> Any:
        visualize = getattr(self._task, "visualize", lambda: self._state)
        return visualize()

    def task_id(self):
        return getattr(self._task, "task_id", None)
