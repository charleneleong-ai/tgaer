from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Guard(ABC):
    """Env-agnostic trajectory guard.

    A guard watches the (observation, action) stream of a single episode and
    emits a planner ``hint`` when it detects a degenerate loop. The three
    lifecycle hooks are called once per step by :class:`GuardedAgent`:
    ``observe`` (pre-act, sees the observation), ``hint`` (pure read), and
    ``record_action`` (post-act, sees the chosen action). Subclasses override
    only the hooks they need.
    """

    def can_run(self, env: Any) -> bool:
        """Whether this guard applies to ``env``. Default: always.

        Trajectory guards are already harmless on single-shot envs (their
        windows never fill in a one-step episode); override only to opt out
        explicitly.
        """
        return True

    def observe(self, observation: Any) -> None:
        return

    @abstractmethod
    def hint(self) -> str | None: ...

    def record_action(self, action: Any) -> None:
        return

    def reset(self) -> None:
        return
