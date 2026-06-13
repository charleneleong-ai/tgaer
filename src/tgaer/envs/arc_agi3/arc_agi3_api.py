from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

Grid = list[list[int]]

# ARC-AGI-3 game states that end an episode (vs NOT_STARTED / NOT_FINISHED).
TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})


@dataclass
class ArcFrame:
    """One observation returned by the ARC-AGI-3 API (RESET / ACTION* response)."""

    game_id: str
    guid: str
    frame: list[Grid]  # list of 64x64 grids, colour indices 0-15
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]


@dataclass
class ArcAction:
    """An agent action: a simple action (id 1-5,7) or ACTION6 with x/y coords."""

    id: int
    x: int | None = None
    y: int | None = None


class ArcTransport(Protocol):
    """Transport the env depends on — the seam between env logic and the HTTP API."""

    def reset(self, game_id: str) -> ArcFrame: ...

    def act(
        self,
        game_id: str,
        guid: str,
        action_id: int,
        x: int | None = None,
        y: int | None = None,
    ) -> ArcFrame: ...
