from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    LS20_DEFAULT,
    KeyDoorController,
    Semantics,
    avatar_is_sprite,
    to_action,
)
from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction
from tgaer.envs.arc_agi3.rendering import grid_to_png_data_url

_JSON_RE = re.compile(r"\{[^{}]*\}")
_VERBS = ("navigate", "press")
_SYSTEM = (
    "You are the perception module for an ARC-AGI-3 grid game. You are shown the "
    "board as an image and the list of palette indices present. Identify the roles: "
    "which index is the AVATAR (the piece the player moves), which are KEY markers "
    "(collected on contact), which is the DOOR/goal, and which are WALLS (impassable). "
    "Also decide the interaction VERB: 'navigate' if the avatar wins by moving onto "
    "the key then the door, or 'press' if it must trigger an action on a target. "
    "Reply with ONE JSON object on the final line: "
    '{"avatar": <int>, "keys": [<int>...], "door": <int>, "walls": [<int>...], '
    '"verb": "navigate"|"press"}.'
)


class Scientist:
    def __init__(
        self,
        model: str = "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        api_base: str | None = "http://localhost:8000/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = "local" if api_base else None
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.last_reply: str = ""

    def infer(self, frame) -> Semantics | None:
        try:
            arr = np.asarray((frame or [None])[-1])
            present = sorted(int(v) for v in np.unique(arr))
            prompt = f"Palette indices present on the board: {present}."
            self.last_reply = self._complete(prompt, grid_to_png_data_url(frame))
            return self._parse(self.last_reply, arr)
        except Exception:
            return None

    def _parse(self, raw: str, arr: np.ndarray) -> Semantics | None:
        present = set(int(v) for v in np.unique(arr))
        matches = _JSON_RE.findall(raw)
        if not matches:
            return None
        try:
            d = json.loads(matches[-1])
            verb = d.get("verb")
            if verb not in _VERBS:
                return None
            avatar, door = int(d["avatar"]), int(d["door"])
            keys = tuple(int(k) for k in d.get("keys", []))
            walls = tuple(int(w) for w in d.get("walls", []))
            # Hallucination guard (any-present): avatar and door must each be visible;
            # keys/walls need ≥1 index present (a multi-index role may show only one
            # of its indices on a given frame, so requiring ALL is too strict).
            if (
                avatar not in present
                or door not in present
                or not keys
                or not any(k in present for k in keys)
                or not walls
                or not any(w in present for w in walls)
            ):
                return None
            # Geometry guard: the any-present check only proves the indices exist,
            # not that the avatar role is correct. A confidently-wrong avatar (a
            # wall/floor index) is structural rather than a sprite — reject it so the
            # agent keeps its trusted heuristic instead of navigating a wall index.
            if not avatar_is_sprite(arr, avatar):
                return None
            return Semantics(avatar, keys, door, walls, verb)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _complete(self, prompt: str, image_url: str | None) -> str:
        # local import: litellm is a heavy optional dep, and tests stub _complete so it never loads
        from litellm import completion

        extra = (
            {"api_base": self._api_base, "api_key": self._api_key}
            if self._api_base
            else {}
        )
        content: Any = prompt
        if image_url:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        resp = completion(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **extra,
        )
        return resp.choices[0].message.content or ""


class ScientistPlannerAgent(Agent):
    """Planner agent that queries a VL scientist once per level for semantics.

    The scientist is injected via ``scientist=`` for tests; defaults to a real
    ``Scientist`` talking to a local vLLM/SGLang endpoint. If ``infer`` returns
    ``None`` the agent degrades gracefully to ``LS20_DEFAULT``.
    """

    def __init__(
        self,
        seed: int = 0,
        model: str = "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        api_base: str | None = "http://localhost:8000/v1",
        stall_limit: int = 8,
        scientist: Scientist | None = None,
        **_: Any,
    ) -> None:
        self._sci = scientist or Scientist(model=model, api_base=api_base)
        self._ctl = KeyDoorController()
        self._stall_limit = stall_limit
        self._levels = -1
        self._sem = LS20_DEFAULT
        self._stall = 0
        self.last_reply: str | None = None

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        frame = obs.get("frame") or []
        if not frame:
            return to_action((obs.get("available_actions") or [1])[0])
        arr = np.asarray(frame[-1])
        levels = obs.get("levels_completed", self._levels)
        if levels != self._levels:
            self._levels = levels
            self._ctl.on_new_level()
            self._stall = 0
            self._sem = self._sci.infer(frame) or LS20_DEFAULT
        self._ctl.learn(arr, self._sem)
        action = self._ctl.step(arr, self._sem, obs.get("available_actions") or [1])
        self._stall = 0 if self._ctl.made_progress() else self._stall + 1
        if self._stall >= self._stall_limit:  # re-query once on a stall
            self._sem = self._sci.infer(frame) or self._sem
            self._stall = 0
        self.last_reply = f"[scientist] act={action.id} verb={self._sem.verb}"
        return action
