from __future__ import annotations

from collections import deque
from typing import Any

from tgaer.guards.base import Guard

FUTILE_ACTION_WINDOW = 3
REPEATED_PLAN_WINDOW = 4


class FutileActionGuard(Guard):
    """No-op detector: fires when the last ``window`` observations are
    identical, meaning recent actions produced no observable change (walking
    into a wall, picking a rejected move, ...).

    Ported from orak MACLA's ``_detect_futile_action``.
    """

    def __init__(self, window: int = FUTILE_ACTION_WINDOW) -> None:
        self._window = window
        self._hashes: deque[int] = deque(maxlen=window)

    def observe(self, observation: Any) -> None:
        self._hashes.append(hash(_freeze(observation)))

    def hint(self) -> str | None:
        if not _window_is_uniform(self._hashes, self._window):
            return None
        return (
            f"[Futile-action notice] Your last {self._window} actions produced "
            f"no observable change in the game state — the action you keep "
            f"choosing is being rejected by the environment. Pick a clearly "
            f"different action this step."
        )

    def reset(self) -> None:
        self._hashes.clear()


class RepeatedPlanGuard(Guard):
    """Action-side sibling of :class:`FutileActionGuard`: fires when the last
    ``window`` chosen actions are identical, regardless of whether the
    observation is stable. Catches envs where the observation ticks
    continuously but the agent loops on the same rejected plan.

    Ported from orak MACLA's ``_detect_repeated_plan``.
    """

    def __init__(self, window: int = REPEATED_PLAN_WINDOW) -> None:
        self._window = window
        self._plans: deque[Any] = deque(maxlen=window)

    def record_action(self, action: Any) -> None:
        self._plans.append(_freeze(action))

    def hint(self) -> str | None:
        if not _window_is_uniform(self._plans, self._window):
            return None
        return (
            f"[Repeated-plan notice] You have chosen the same action plan "
            f"{self._window} steps in a row and the goal isn't advancing — the "
            f"environment is rejecting it or it's a no-op. Pick a structurally "
            f"different plan this step (different action, target, or sub-goal)."
        )

    def reset(self) -> None:
        self._plans.clear()


def _window_is_uniform(window: deque[Any], size: int) -> bool:
    """True once ``window`` is full (``size`` entries) and every entry is equal."""
    return len(window) == size and len(set(window)) == 1


def _freeze(value: Any) -> Any:
    """Hashable projection of an observation/action (dicts and lists aren't
    hashable; their repr is a stable stand-in for equality comparison)."""
    return (
        value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
    )
