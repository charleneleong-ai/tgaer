"""LLM REPL agent for ARC-AGI-3 — Duck-harness-style.

The agent works in a Python REPL where game observations are encoded as
variables. It can inspect variables, evaluate helper functions, and take
actions. The model drives; the harness is minimal.

Key variables available to the model:
- current_frame: latest board state (64x64 grid)
- previous_frame: previous board state
- history: list of (action, frame) pairs
- transitions: list of (state, action, next_state) tuples
- levels_completed: current level count
- available_actions: list of valid action IDs
- game_state: current game state (NOT_FINISHED, WIN, GAME_OVER)

Usage:
    agent = ArcAgi3ReplAgent(model="gemini/gemini-3.1-flash-lite")
    action = agent.act(observation)
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any

import numpy as np

from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import (
    COMPLEX_ACTION_ID,
    GRID_SIZE,
    ArcAction,
    random_action,
)
from tgaer.envs.arc_agi3.rendering import grid_to_png_data_url

_JSON_RE = re.compile(r"\{[^{}]*\}")

_SYSTEM = """You are playing ARC-AGI-3, an interactive grid puzzle. You have access to a Python REPL with game state as variables.

Available variables:
- current_frame: numpy array (64x64) of the current board state (values 0-15)
- previous_frame: numpy array of the previous board state (or None on first move)
- history: list of (action_id, frame_before, frame_after) tuples
- transitions: list of (state_before, action, state_after) summaries
- levels_completed: int, number of levels completed
- available_actions: list of valid action IDs for this turn
- game_state: str, "NOT_FINISHED", "WIN", or "GAME_OVER"
- grid_to_ascii(arr): helper to render grid as readable ASCII
- count_changes(arr1, arr2): helper to count changed cells between frames

Available actions:
- 1-5, 7: Simple actions (meaning varies by game)
- 6: Complex action (requires x, y coordinates in [0, 63])

Your goal: Complete levels as efficiently as possible. Each level ends when you reach a WIN state. Explore the game mechanics, build a mental model, then execute your strategy.

Reply format:
1. Brief analysis (1-3 sentences)
2. Python code to inspect state or plan (optional)
3. Final line: action JSON {"id": <action_id>, "x": <int|null>, "y": <int|null>}"""

_HELPERS = '''
def grid_to_ascii(arr):
    """Render grid as readable ASCII with row numbers."""
    lines = []
    for r, row in enumerate(arr):
        lines.append(f"{r:2d}|" + "".join(f"{c:x}" for c in row))
    return "\\n".join(lines)

def count_changes(arr1, arr2):
    """Count cells that changed between two frames."""
    if arr1 is None or arr2 is None:
        return 0
    return int(np.sum(arr1 != arr2))

def find_objects(arr):
    """Find connected components and their centroids."""
    from collections import deque
    mask = arr > 0
    seen = np.zeros_like(mask, bool)
    objects = []
    for r0, c0 in np.argwhere(mask):
        if seen[r0, c0]:
            continue
        val = int(arr[r0, c0])
        q, comp = deque([(r0, c0)]), []
        seen[r0, c0] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                r2, c2 = r+dr, c+dc
                if (0 <= r2 < arr.shape[0] and 0 <= c2 < arr.shape[1] 
                    and mask[r2, c2] and not seen[r2, c2]):
                    seen[r2, c2] = True
                    q.append((r2, c2))
        centroid = np.mean(comp, axis=0)
        objects.append({"value": val, "size": len(comp), "centroid": centroid.tolist()})
    return sorted(objects, key=lambda o: o["size"])
'''


class ArcAgi3ReplAgent(Agent):
    """LLM REPL agent for ARC-AGI-3 — Duck-harness-style.
    
    The model interacts with the game through a Python REPL, inspecting
    state, running analysis, and choosing actions. The harness provides
    helper functions and game state as pre-loaded variables.
    """

    def __init__(
        self,
        seed: int = 0,
        model: str = "gemini/gemini-3.1-flash-lite",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        max_history: int = 12,
        api_base: str | None = None,
        api_key: str | None = None,
        vision: bool = True,
        **_: Any,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_history = max_history
        self._vision = vision
        self._api_base = api_base
        self._api_key = api_key or ("local" if api_base else None)
        
        # Game state
        self._history: list[tuple[int, list[list[int]], list[list[int]]]] = []
        self._transitions: list[str] = []
        self._levels = -1
        self._prev_frame: list[list[int]] | None = None
        
        # Logging
        self.last_reasoning: str = ""
        self.last_reply: str = ""
        self.last_code: str = ""

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        available = obs.get("available_actions") or [1]
        frame = obs.get("frame") or []
        levels = obs.get("levels_completed", self._levels)
        state = obs.get("state", "NOT_FINISHED")
        
        # Reset on new level
        if levels != self._levels:
            self._levels = levels
            self._history.clear()
            self._transitions.clear()
        
        if not frame:
            return self._fallback(available)
        
        current_grid = frame[-1]
        
        # Build prompt with game state
        prompt = self._build_prompt(obs, current_grid, available, state)
        image_url = grid_to_png_data_url(frame) if self._vision else None
        
        try:
            reply = self._complete(prompt, image_url)
            self.last_reply = reply
            action = self._parse_action(reply, available)
            
            # Record history
            if self._prev_frame is not None:
                self._history.append((action.id, self._prev_frame, current_grid))
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]
                
                # Record transition summary
                changes = int(np.sum(np.array(self._prev_frame) != np.array(current_grid)))
                self._transitions.append(
                    f"Action {action.id} changed {changes} cells"
                )
                if len(self._transitions) > self._max_history:
                    self._transitions = self._transitions[-self._max_history:]
            
            self._prev_frame = current_grid
            return action
            
        except Exception as e:
            self.last_reasoning = f"Error: {e}"
            return self._fallback(available)

    def _build_prompt(
        self, obs: dict, grid: list[list[int]], available: list[int], state: str
    ) -> str:
        arr = np.array(grid)
        
        # ASCII grid
        ascii_grid = "\n".join(
            f"{r:2d}|" + "".join(f"{c:x}" for c in row)
            for r, row in enumerate(grid)
        )
        
        # Object detection (simple: count unique values)
        unique, counts = np.unique(arr, return_counts=True)
        palette_info = ", ".join(
            f"{v}({n} cells)" for v, n in zip(unique, counts) if n < 1000
        )
        
        # History summary
        history_str = "none" if not self._transitions else "; ".join(self._transitions[-5:])
        
        # Diff feedback
        diff_str = ""
        if self._prev_frame is not None:
            changes = int(np.sum(np.array(self._prev_frame) != arr))
            if changes == 0:
                diff_str = "\nWARNING: Your last action changed NOTHING. Try a different approach."
            else:
                diff_str = f"\nYour last action changed {changes} cells."
        
        return f"""Game state: {state} | Levels completed: {self._levels}
Available actions: {available}
Recent history: {history_str}{diff_str}

Palette (value: count): {palette_info}

Board (64x64, each char = cell value 0-f):
{ascii_grid}

Analyze the board structure. What patterns do you see? What might the goal be?
Then output your action as JSON on the final line."""

    def _parse_action(self, raw: str, available: list[int]) -> ArcAction:
        matches = _JSON_RE.findall(raw)
        if not matches:
            raise ValueError("no JSON in reply")
        
        data = json.loads(matches[-1])
        action_id = int(data["id"])
        
        if action_id not in available:
            raise ValueError(f"action {action_id} not in available {available}")
        
        if action_id == COMPLEX_ACTION_ID:
            x = self._clamp_coord(data.get("x"))
            y = self._clamp_coord(data.get("y"))
            return ArcAction(id=action_id, x=x, y=y)
        
        return ArcAction(id=action_id)

    @staticmethod
    def _clamp_coord(val: Any) -> int:
        try:
            return max(0, min(GRID_SIZE - 1, int(val)))
        except (TypeError, ValueError):
            return GRID_SIZE // 2

    def _fallback(self, available: list[int]) -> ArcAction:
        return random_action(available, __import__("random").Random())

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
        self.last_reasoning = getattr(msg, "reasoning_content", "") or ""
        return msg.content or ""
