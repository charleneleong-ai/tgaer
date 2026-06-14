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
    "board as a grid of integer cell values. Each character is ONE cell's value, a "
    "palette index 0-15 written 0-9 then a-f (a=10 ... f=15) — these are category "
    "labels, NOT hex colour codes. Rows are 0-indexed top-to-bottom (label on the "
    "left); columns 0-indexed left-to-right. You also get feedback on what your last "
    "action changed. Infer the rules from how the board responds, then pick ONE action "
    "that makes progress toward completing levels. Actions other than "
    f"{COMPLEX_ACTION_ID} are simple moves; action {COMPLEX_ACTION_ID} acts on a single "
    "cell at (x, y) with x=column and y=row in [0,63]. "
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
        vision: bool = False,
    ) -> None:
        self._rng = random.Random(seed)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_history = max_history
        # When True, also send the board rendered as an image (VL models).
        self._vision = vision
        # Set for an OpenAI-compatible local server (e.g. vLLM on :8000); leave
        # None to use the provider keyed off the model prefix (gemini/, etc.).
        self._api_base = api_base
        self._api_key = (
            api_key if api_key is not None else ("local" if api_base else None)
        )
        self._history: list[str] = []
        self._prev_grid: list[list[int]] | None = None  # last frame, for diff feedback
        # Last turn's model trace, surfaced for logging/audit (empty on fallback).
        self.last_reasoning: str = ""
        self.last_reply: str = ""

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        available = obs.get("available_actions") or [1]
        self.last_reasoning = ""
        self.last_reply = ""
        try:
            image_url = self._image_of(obs) if self._vision else None
            self.last_reply = self._complete(self._build_prompt(obs, available), image_url)
            action = self._parse(self.last_reply, available)
        except Exception:
            action = self._fallback(available)
        tag = f"{action.id}"
        if action.id == COMPLEX_ACTION_ID:
            tag += f"@({action.x},{action.y})"
        self._history = [*self._history, tag][-self._max_history :]
        self._prev_grid = self._grid_of(obs)  # compare against this next step
        return action

    def _image_of(self, obs: dict) -> str | None:
        from tgaer.envs.arc_agi3.rendering import grid_to_png_data_url

        return grid_to_png_data_url(obs.get("frame"))

    def _complete(self, prompt: str, image_url: str | None = None) -> str:
        from litellm import completion

        extra = {}
        if self._api_base:
            extra = {"api_base": self._api_base, "api_key": self._api_key}
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
        msg = resp.choices[0].message
        # vLLM/Gemini expose the <think> trace separately when a reasoning
        # parser is active; keep it for logging/audit.
        self.last_reasoning = getattr(msg, "reasoning_content", "") or ""
        return msg.content or ""

    @staticmethod
    def _grid_of(obs: dict) -> list[list[int]]:
        frames = obs.get("frame") or []
        return frames[-1] if frames else []

    def _build_prompt(self, obs: dict, available: list[int]) -> str:
        grid = self._grid_of(obs)
        history = ", ".join(self._history) or "(none)"
        return (
            f"State: {obs.get('state')}  Levels completed: {obs.get('levels_completed')}\n"
            f"Available action ids: {available}\n"
            f"Recent actions: {history}\n"
            f"{self._diff_feedback(grid)}\n"
            f"Board ({len(grid)} rows x {len(grid[0]) if grid else 0} cols), "
            "each char = one cell's value 0-f:\n"
            f"{self._render_grid(grid)}\n"
            "Choose the next action as JSON."
        )

    @staticmethod
    def _render_grid(grid: list[list[int]]) -> str:
        # Row index on the left so the model can read coordinates unambiguously.
        return "\n".join(
            f"{r:2d}|" + "".join(format(c, "x") for c in row)
            for r, row in enumerate(grid)
        )

    def _diff_feedback(self, grid: list[list[int]]) -> str:
        prev = self._prev_grid
        if prev is None:
            return "Feedback: first move (no prior action)."
        last = self._history[-1] if self._history else "?"
        changed = [
            (r, c)
            for r in range(min(len(grid), len(prev)))
            for c in range(min(len(grid[r]), len(prev[r])))
            if grid[r][c] != prev[r][c]
        ]
        if not changed:
            return f"Feedback: your last action ({last}) changed NOTHING — try something different."
        sample = ", ".join(f"({r},{c})" for r, c in changed[:8])
        return f"Feedback: your last action ({last}) changed {len(changed)} cells, e.g. {sample}."

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
