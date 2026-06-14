from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

Grid = list[list[int]]

# ARC-AGI-3 game states that end an episode (vs NOT_STARTED / NOT_FINISHED).
TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})

GRID_SIZE = 64  # ARC-AGI-3 grids are 64x64, colour indices 0-15
COMPLEX_ACTION_ID = 6  # ACTION6 carries (x, y) coordinates in [0, GRID_SIZE)


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


def random_action(available: list[int], rng: random.Random) -> ArcAction:
    """Pick a uniformly random action from ``available``; ACTION6 gets random
    in-grid coordinates. Shared by the random baseline and the LLM agent's
    fallback path so action construction lives in one place."""
    action_id = rng.choice(available)
    if action_id == COMPLEX_ACTION_ID:
        return ArcAction(
            id=action_id, x=rng.randrange(GRID_SIZE), y=rng.randrange(GRID_SIZE)
        )
    return ArcAction(id=action_id)


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
