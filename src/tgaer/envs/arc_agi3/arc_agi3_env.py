from __future__ import annotations

from tgaer.core.env_base import Environment, Transition
from tgaer.envs.arc_agi3.arc_agi3_api import (
    TERMINAL_STATES,
    ArcAction,
    ArcFrame,
    ArcTransport,
)


class ArcAgi3Environment(Environment):
    """Multi-step interactive ARC-AGI-3 environment.

    Drives one game via an injected transport. Reward is the per-step delta in
    ``levels_completed``; the episode ends on WIN/GAME_OVER or after
    ``max_actions`` steps (the ARC-AGI-3 cap is 80).
    """

    DEFAULT_MAX_ACTIONS = 80

    def __init__(
        self,
        transport: ArcTransport,
        game_id: str,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        reset_on_game_over: bool = False,
    ) -> None:
        self._transport = transport
        self._game_id = game_id
        self._max_actions = max_actions
        self._reset_on_game_over = reset_on_game_over
        self._frame: ArcFrame | None = None
        self._steps = 0

    def reset(self, seed: int | None = None) -> dict:
        self._frame = self._transport.reset(self._game_id)
        self._steps = 0
        return self._obs(self._frame)

    def step(self, action: ArcAction) -> Transition:
        prev_levels = self._frame.levels_completed
        frame = self._transport.act(
            self._game_id, self._frame.guid, action.id, action.x, action.y
        )
        self._steps += 1
        reward = float(frame.levels_completed - prev_levels)
        # On death, optionally respawn (within budget) so the agent can learn the
        # fatal transition and keep playing rather than forfeiting the episode.
        respawned = (
            frame.state == "GAME_OVER"
            and self._reset_on_game_over
            and self._steps < self._max_actions
        )
        if respawned:
            frame = self._transport.reset(self._game_id)
        self._frame = frame
        done = frame.state in TERMINAL_STATES or self._steps >= self._max_actions
        obs = self._obs(frame)
        if respawned:
            obs["terminal"] = True  # signal the agent: the prior action was fatal
        return Transition(
            state=obs,
            action=action,
            reward=reward,
            done=done,
            info={
                "state": frame.state,
                "levels_completed": frame.levels_completed,
                "win_levels": frame.win_levels,
                "available_actions": frame.available_actions,
                "guid": frame.guid,
                "steps": self._steps,
            },
        )

    def render(self) -> dict:
        frame = self._frame
        return {
            "game_id": self._game_id,
            "state": frame.state if frame else None,
            "levels_completed": frame.levels_completed if frame else None,
            "steps": self._steps,
        }

    def task_id(self) -> str:
        return self._game_id

    @staticmethod
    def _obs(frame: ArcFrame) -> dict:
        return {
            "frame": frame.frame,
            "available_actions": frame.available_actions,
            "levels_completed": frame.levels_completed,
            "state": frame.state,
        }
