from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction


class RecordingAgent(Agent):
    """Wraps an agent, appending every ``(obs, action)`` it processes to a JSONL.

    Capture happens at the agent boundary so the recorded obs is exactly what
    ``inner.act`` saw — including the env's post-transport ``terminal`` flag and
    respawn frame, which a transport-level recorder would miss."""

    def __init__(self, inner: Agent, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("")  # truncate any prior capture

    def act(self, observation: Any) -> Any:
        action = self._inner.act(observation)
        rec = {"obs": observation, "action": _action_dict(action)}
        with self._path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return action

    def reset(self) -> None:
        if hasattr(self._inner, "reset"):
            self._inner.reset()


def _action_dict(a: ArcAction) -> dict:
    return {"id": a.id, "x": a.x, "y": a.y}
