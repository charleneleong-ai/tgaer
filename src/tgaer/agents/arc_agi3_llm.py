from __future__ import annotations

import json
import random
import re
from typing import Any

from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import (
    COMPLEX_ACTION_ID,
    GRID_SIZE,
    ArcAction,
    random_action,
)

_SYSTEM = (
    "You are playing ARC-AGI-3, an interactive grid puzzle. Each turn you see the "
    "current 64x64 grid (colour indices 0-15) and the action ids available to you. "
    "Pick ONE action that makes progress toward completing levels. Actions other than "
    f"{COMPLEX_ACTION_ID} are simple moves; action {COMPLEX_ACTION_ID} acts on a single "
    "cell and needs x,y coordinates in [0,63]. "
    'Reply with ONLY a JSON object: {"id": <available action id>, "x": <int|null>, '
    '"y": <int|null>}.'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ArcAgi3LLMAgent(Agent):
    """LLM agent for the interactive ARC-AGI-3 env (Gemini via litellm by default).

    Reasons over the current frame + available actions and returns one ``ArcAction``
    per step. Any failure — network error, unparseable reply, or an action the game
    didn't offer — falls back to a random *available* action so a single bad turn
    never aborts the episode.
    """

    def __init__(
        self,
        seed: int = 0,
        model: str = "gemini/gemini-3.1-flash-lite",
        temperature: float = 0.3,
        max_tokens: int = 256,
        max_history: int = 8,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_history = max_history
        # Set for an OpenAI-compatible local server (e.g. vLLM on :8000); leave
        # None to use the provider keyed off the model prefix (gemini/, etc.).
        self._api_base = api_base
        self._api_key = (
            api_key if api_key is not None else ("local" if api_base else None)
        )
        self._history: list[str] = []
        # Last turn's model trace, surfaced for logging/audit (empty on fallback).
        self.last_reasoning: str = ""
        self.last_reply: str = ""

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        available = obs.get("available_actions") or [1]
        self.last_reasoning = ""
        self.last_reply = ""
        try:
            self.last_reply = self._complete(self._build_prompt(obs, available))
            action = self._parse(self.last_reply, available)
        except Exception:
            action = self._fallback(available)
        tag = f"{action.id}"
        if action.id == COMPLEX_ACTION_ID:
            tag += f"@({action.x},{action.y})"
        self._history = [*self._history, tag][-self._max_history :]
        return action

    def _complete(self, prompt: str) -> str:
        from litellm import completion

        extra = {}
        if self._api_base:
            extra = {"api_base": self._api_base, "api_key": self._api_key}
        resp = completion(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **extra,
        )
        msg = resp.choices[0].message
        # vLLM/Gemini expose the <think> trace separately when a reasoning
        # parser is active; keep it for logging/audit.
        self.last_reasoning = getattr(msg, "reasoning_content", "") or ""
        return msg.content or ""

    def _build_prompt(self, obs: dict, available: list[int]) -> str:
        frames = obs.get("frame") or []
        grid = frames[-1] if frames else []
        grid_txt = "\n".join("".join(format(c, "x") for c in row) for row in grid)
        history = ", ".join(self._history) or "(none)"
        return (
            f"State: {obs.get('state')}  Levels completed: {obs.get('levels_completed')}\n"
            f"Available action ids: {available}\n"
            f"Recent actions: {history}\n"
            f"Current grid (hex colour per cell):\n{grid_txt}\n"
            "Choose the next action as JSON."
        )

    def _parse(self, raw: str, available: list[int]) -> ArcAction:
        match = _JSON_RE.search(raw)
        if not match:
            raise ValueError("no JSON object in reply")
        data = json.loads(match.group(0))
        action_id = int(data["id"])
        if action_id not in available:
            raise ValueError(f"action {action_id} not in available {available}")
        if action_id == COMPLEX_ACTION_ID:
            return ArcAction(
                id=action_id, x=self._coord(data.get("x")), y=self._coord(data.get("y"))
            )
        return ArcAction(id=action_id)

    def _coord(self, value: Any) -> int:
        try:
            return max(0, min(GRID_SIZE - 1, int(value)))
        except (TypeError, ValueError):
            return self._rng.randrange(GRID_SIZE)

    def _fallback(self, available: list[int]) -> ArcAction:
        return random_action(available, self._rng)
