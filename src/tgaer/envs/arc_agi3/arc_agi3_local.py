"""Local transport adapter for arcengine/arc-agi games.

Wraps the arc_agi.Arcade API to implement the ArcTransport protocol,
allowing our agents to play local games (pw14, tl01, etc.) that aren't
served by the hosted API.
"""

from __future__ import annotations

import uuid
from typing import Any

from arc_agi import Arcade, OperationMode

from tgaer.envs.arc_agi3.arc_agi3_api import ArcFrame


class LocalArcTransport:
    """Transport that plays local arcengine games via the arc_agi.Arcade wrapper."""

    def __init__(self, environments_dir: str = "environment_files") -> None:
        self._arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=environments_dir,
        )
        self._envs: dict[str, Any] = {}

    def _ensure_env(self, game_id: str) -> Any:
        if game_id not in self._envs:
            self._envs[game_id] = self._arc.make(game_id, render_mode=None)
        return self._envs[game_id]

    def reset(self, game_id: str) -> ArcFrame:
        env = self._ensure_env(game_id)
        obs = env.reset()
        # obs is FrameDataRaw which has .frame (list of ndarrays), not .grid
        frame = obs.frame if hasattr(obs, "frame") else []
        return ArcFrame(
            game_id=game_id,
            guid=str(uuid.uuid4()),
            frame=frame,
            state=obs.state.name if hasattr(obs, "state") else str(obs.state),
            levels_completed=obs.levels_completed,
            win_levels=obs.win_levels,
            available_actions=obs.available_actions,
        )

    _ACTION_MAP = {
        0: "RESET",
        1: "ACTION1",
        2: "ACTION2",
        3: "ACTION3",
        4: "ACTION4",
        5: "ACTION5",
        6: "ACTION6",
        7: "ACTION7",
    }

    def act(
        self,
        game_id: str,
        guid: str,
        action_id: int,
        x: int | None = None,
        y: int | None = None,
    ) -> ArcFrame:
        env = self._ensure_env(game_id)
        from arcengine import GameAction

        action_name = self._ACTION_MAP.get(action_id, f"ACTION{action_id}")
        action = GameAction[action_name]
        data = {"x": x, "y": y} if action_id == 6 and x is not None and y is not None else None
        obs = env.step(action, data=data)
        # obs is FrameDataRaw which has .frame (list of ndarrays), not .grid
        frame = obs.frame if hasattr(obs, "frame") else []
        return ArcFrame(
            game_id=game_id,
            guid=guid,
            frame=frame,
            state=obs.state.name if hasattr(obs, "state") else str(obs.state),
            levels_completed=obs.levels_completed,
            win_levels=obs.win_levels,
            available_actions=obs.available_actions,
        )

    def close(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs.clear()
