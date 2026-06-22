from __future__ import annotations

from typing import Any

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    KeyDoorController,
    LS20_DEFAULT,
    to_action,
)
from tgaer.agents.arc_agi3_scientist import Scientist
from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics
from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction


class EmpiricalPlannerAgent(Agent):
    """Key→door controller whose semantics are detected from the action stream.

    A VL ``Scientist`` supplies a per-episode cold-start labelling (frame 0);
    ``EmpiricalSemantics`` then pins avatar/key/door from evidence and overrides
    the cold-start. With no VL the cold-start is ``LS20_DEFAULT``.
    """

    def __init__(
        self,
        seed: int = 0,
        model: str = "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        api_base: str | None = "http://localhost:8000/v1",
        scientist: Scientist | None = None,
        **_: Any,
    ) -> None:
        self._sci = scientist or (Scientist(model=model, api_base=api_base) if api_base else None)
        self._ctl = KeyDoorController()
        self._det = EmpiricalSemantics()
        self._cold = LS20_DEFAULT
        self._started = False
        self._levels = -1
        self._prev: np.ndarray | None = None
        self._prev_action: int | None = None
        self.last_reply: str | None = None

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        frame = obs.get("frame") or []
        if not frame:
            return to_action((obs.get("available_actions") or [1])[0])
        arr = np.asarray(frame[-1])
        levels = obs.get("levels_completed", self._levels)

        if not self._started:
            self._started = True
            if self._sci is not None:
                self._cold = self._sci.infer(frame) or LS20_DEFAULT
        if levels != self._levels:
            self._levels = levels
            self._ctl.on_new_level()

        self._det.observe(self._prev, self._prev_action, arr, levels)
        sem = self._det.semantics(self._cold)
        self._ctl.learn(arr, sem)
        aid = self._ctl.step(arr, sem, obs.get("available_actions") or [1])

        self._prev, self._prev_action = arr, aid
        self.last_reply = f"[empirical] act={aid} avatar={sem.avatar}"
        return to_action(aid)
