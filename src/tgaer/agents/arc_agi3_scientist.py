from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from tgaer.agents.arc_agi3_grid import Semantics
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
            present = set(int(v) for v in np.unique(arr))
            prompt = f"Palette indices present on the board: {sorted(present)}."
            self.last_reply = self._complete(prompt, grid_to_png_data_url(frame))
            return self._parse(self.last_reply, present)
        except Exception:
            return None

    def _parse(self, raw: str, present: set[int]) -> Semantics | None:
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
