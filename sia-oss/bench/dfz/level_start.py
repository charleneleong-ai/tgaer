"""Start a TAAF game at a recorded level by replaying the actions that reached it."""

from __future__ import annotations

from dataclasses import dataclass, field

from arcengine import GameAction
from taaf.game import GameState, RunSession
from taaf.game_api import GameAPI


@dataclass
class LevelStartGameAPI(GameAPI):
    """A GameAPI whose start replays `prefix` — (action name, data) pairs — before handing over."""

    prefix: tuple[tuple[str, dict[str, int]], ...] = field(default=(), kw_only=True)

    def _start_game(self, session: RunSession) -> GameState:
        state = super()._start_game(session)
        for name, data in self.prefix:
            resp = self.env.step(GameAction[name], data=dict(data))
            if resp is None or not resp.frame:
                raise RuntimeError(
                    f"prefix replay of {self.env_name} stopped at {name}"
                )
            state = GameState(raw=resp)
        return state
