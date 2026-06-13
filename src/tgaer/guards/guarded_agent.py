from __future__ import annotations

from typing import Any

from tgaer.core.agent_base import Agent
from tgaer.guards.base import Guard


class GuardedAgent(Agent):
    """Wraps an inner :class:`Agent`, running a set of trajectory guards each
    step and injecting any fired hints into the observation before delegating
    to ``inner.act``.

    Guards whose ``can_run(env)`` is False are dropped at construction, so the
    hot path stays branch-free. ``hint_count`` records how many steps fired at
    least one hint — a cheap eval signal for how often the agent looped.
    """

    def __init__(self, inner: Agent, guards: list[Guard], *, env: Any = None) -> None:
        self._inner = inner
        self._guards = [g for g in guards if env is None or g.can_run(env)]
        self.hint_count = 0

    def act(self, observation: Any) -> Any:
        for guard in self._guards:
            guard.observe(observation)
        hints = [h for guard in self._guards if (h := guard.hint())]
        if hints:
            self.hint_count += 1
        action = self._inner.act(_augment(observation, hints) if hints else observation)
        for guard in self._guards:
            guard.record_action(action)
        return action

    def update_context(self, feedback: dict[str, Any]) -> None:
        self._inner.update_context(feedback)

    def reset(self) -> None:
        self.hint_count = 0
        for guard in self._guards:
            guard.reset()


def _augment(observation: Any, hints: list[str]) -> Any:
    if isinstance(observation, dict):
        return {**observation, "guard_hints": hints}
    return f"{observation}\n\n" + "\n".join(hints)
