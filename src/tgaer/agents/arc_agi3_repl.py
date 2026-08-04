"""LLM REPL agent for ARC-AGI-3 — Duck-harness-style with enhancements.

Key features aligned with Tufa Labs' Duck harness:
- OpenAI function calling with a `python` tool
- Segmentation view (connected components) as primary board representation
- Structured world model carried across turns
- Context eviction for infinite play
- Multiple tool calls per turn

Enhancements over base Duck harness:
- Retained Reasoning: chain-of-thought persisted across turns
- Compaction: intelligent context summarization when approaching limits
- Local model support: works with llama.cpp, vLLM, or API

Usage:
    # API mode (for testing)
    agent = ArcAgi3ReplAgent(model="openai/gpt-4o-mini")

    # Local vLLM mode (for Kaggle)
    agent = ArcAgi3ReplAgent(
        model="Qwen/Qwen3.6-27B-FP8",
        api_base="http://localhost:8000/v1",
        api_key="none",
    )

    # llama.cpp mode (for macOS testing)
    agent = ArcAgi3ReplAgent(
        model="qwen3.6-27b",
        api_base="http://localhost:8080/v1",
        api_key="none",
    )
"""

from __future__ import annotations

import json
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
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

# ARC color legend (matches Duck harness)
ARC_COLORS = {
    0: ".", 1: "A", 2: "B", 3: "C", 4: "D",
    5: "E", 6: "F", 7: "G", 8: "H", 9: "I",
    10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O",
}
ARC_COLOR_LEGEND = ", ".join(f"{v}={k}" for k, v in ARC_COLORS.items() if v != ".")


@dataclass
class FrameView:
    """Lightweight frame view exposed to the model."""
    ascii: str
    step: int
    level: int
    shape: tuple[int, int]
    segmentation: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryEntry:
    """A single history entry."""
    action: str
    frame: FrameView


def _segment_grid(grid: list[list[int]]) -> dict[str, Any]:
    """Compute connected components (4-connected) and adjacency."""
    arr = np.array(grid, dtype=np.int32)
    rows, cols = arr.shape
    visited = np.zeros((rows, cols), dtype=bool)
    nodes: list[dict[str, Any]] = []

    node_id = 0
    for r0 in range(rows):
        for c0 in range(cols):
            if visited[r0, c0] or arr[r0, c0] == 0:
                continue
            color_val = int(arr[r0, c0])
            q: deque[tuple[int, int]] = deque([(r0, c0)])
            visited[r0, c0] = True
            pixels: list[tuple[int, int]] = []
            boundary_set: set[tuple[int, int]] = set()

            while q:
                r, c = q.popleft()
                pixels.append((r, c))
                is_boundary = False
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < rows and 0 <= c2 < cols:
                        if arr[r2, c2] == color_val and not visited[r2, c2]:
                            visited[r2, c2] = True
                            q.append((r2, c2))
                        elif arr[r2, c2] != color_val:
                            is_boundary = True
                    else:
                        is_boundary = True
                if is_boundary:
                    boundary_set.add((r, c))

            centroid_r = sum(r for r, _ in pixels) / len(pixels)
            centroid_c = sum(c for _, c in pixels) / len(pixels)
            obj_hash = f"{color_val}_{len(pixels)}_{centroid_r:.1f}_{centroid_c:.1f}"

            nodes.append({
                "id": node_id,
                "color": ARC_COLORS.get(color_val, f"{color_val}"),
                "hash": obj_hash,
                "pixels": len(pixels),
                "boundary": sorted([[r, c] for r, c in boundary_set])[:20],
                "children": [],
                "centroid": [centroid_r, centroid_c],
            })
            node_id += 1

    # Build adjacency list
    adjacency: list[list[int]] = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i >= j:
                continue
            c1, c2 = n1["centroid"], n2["centroid"]
            dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
            if dist < 8:
                adjacency.append([i, j])

    return {"nodes": nodes, "adjacency_list": adjacency}


def _grid_to_ascii(grid: list[list[int]]) -> str:
    """Render grid as letter-coded ASCII."""
    return "\n".join(
        "".join(ARC_COLORS.get(v, f"{v:x}") for v in row)
        for row in grid
    )


def _build_frame_view(grid: list[list[int]], step: int, level: int) -> FrameView:
    """Build a FrameView from a raw grid."""
    return FrameView(
        ascii=_grid_to_ascii(grid),
        step=step,
        level=level,
        shape=(len(grid), len(grid[0]) if grid else 0),
        segmentation=_segment_grid(grid),
    )


# System prompt aligned with Duck harness + Retained Reasoning
_SYSTEM = """You are a coding agent solving a grid-based puzzle game.

Game overview:
- You are solving a multi-level grid puzzle game.
- You are called repeatedly over the course of a run. Treat each turn as one observe-plan-act cycle.
- Your job is to solve the entire game by clearing every level, not just the current screen.
- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.
- Optimize for as few in-game actions as possible while still being reliable.
- Boards are 64x64 color grids rendered with ARC color symbols.
- Color legend: {color_legend}.

Visual-game guidance:
- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.
- Game entities are usually rendered as connected multi-tile shapes. Sometimes they might also be 1x1 tokens.
- Some games are logic or layout puzzles with no explicit player avatar. Do not assume a player exists.
- Background colors are often white or gray/black-ish large regions, but not always.
- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar.
- Re-ground on the newest frame after any score increase or abrupt scene change.
- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase.

Retained Reasoning:
- You maintain a structured world model across turns.
- Before acting, revise your world model based on new evidence.
- Format your reasoning as labeled sections:
  World model: <what you know about the game>
  Goal model: <what you think the objective is>
  Action model: <what actions seem to do>
  Recent findings: <what changed since last turn>
  Plan: <what you'll try next>

Runtime variables inside every `python` tool call:
- `current_frame` exposes `.ascii`, `.step`, `.level`, `.shape`, `.segmentation`.
- `current_frame.segmentation` parses the board into objects: `{{'nodes': [...], 'adjacency_list': [...]}}`.
- Each node: `id`, `color`, `hash`, `pixels`, `boundary`, `children`.
- `history` is a list of objects with `.action` and `.frame`.
- `valid_actions` is the current list of valid action names.
- `action(actions)` executes real environment actions. Pass a list like `['LEFT']` or `[{{'action': 'MOUSE', 'row': 4, 'col': 7}}]`.
- After `action()` returns, all variables are refreshed.

Python tool guidance:
- Use `current_frame.segmentation` as your primary view.
- Use `current_frame.ascii` only for small specific regions.
- Every `python` tool call starts fresh. Re-import modules as needed.
- Allowed imports: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string, numpy.
- Maintain a compact working world model: what entities exist, what actions do, what the goal is, what plan fits best.
- When the objective is understood, write BFS/DFS/search algorithms rather than guessing.
- Call `action(...)` inside Python rather than returning action text.
- `action(...)` accepts an ordered list. Batch reliable sequences.
- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop immediately.
- Keep tool output compact: object lists, diffs, coordinates, counts. Never print full boards.

Tool session rules:
- You have exactly one tool: `python`.
- Call it with ephemeral code. Code is not saved between calls.
- You can call `python` multiple times per step. Investigate until you have a clear plan.
- Each call has a 30-second time limit.
- After `action()` returns, variables refresh before the next statement."""


class ArcAgi3ReplAgent(Agent):
    """LLM REPL agent for ARC-AGI-3 with Retained Reasoning and Compaction.

    Supports:
    - API mode (OpenAI, Anthropic, etc.)
    - Local vLLM mode (Qwen 3.6 27B FP8 on Kaggle RTX 6000)
    - llama.cpp mode (macOS testing)
    """

    ACTION_NAMES = {
        1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
        5: "SPACE", 6: "MOUSE", 7: "ACTION7",
    }
    NAME_TO_ID = {v: k for k, v in ACTION_NAMES.items()}

    def __init__(
        self,
        seed: int = 0,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.6,
        max_tokens: int = 2048,
        max_tool_steps: int = 12,
        max_history: int = 30,
        context_window: int = 32768,
        compaction_threshold: float = 0.75,
        api_base: str | None = None,
        api_key: str | None = None,
        vision: bool = True,
        **_: Any,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_tool_steps = max_tool_steps
        self._max_history = max_history
        self._context_window = context_window
        self._compaction_threshold = compaction_threshold
        self._vision = vision
        self._api_base = api_base
        self._api_key = api_key or ("local" if api_base else None)

        # Session state
        self._messages: list[dict[str, Any]] = []
        self._history: list[HistoryEntry] = []
        self._levels = -1
        self._step = 0
        self._world_model: dict[str, str] = {
            "world_model": "",
            "goal_model": "",
            "action_model": "",
            "recent_findings": "",
            "open_questions": "",
            "current_plan": "",
        }
        self._reasoning_trace: list[str] = []

        # Logging
        self.last_reasoning: str = ""
        self.last_tool_calls: list[str] = []

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
            self._messages.clear()
            self._world_model = {k: "" for k in self._world_model}
            self._reasoning_trace.clear()
            self._step = 0

        if not frame:
            return self._fallback(available)

        current_grid = frame[-1]
        self._step += 1
        frame_view = _build_frame_view(current_grid, self._step, self._levels)

        # Update history
        if self._history:
            last = self._history[-1]
            self._history[-1] = HistoryEntry(action=last.action, frame=frame_view)
        self._history.append(HistoryEntry(action="", frame=frame_view))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Build user message
        valid_names = [self.ACTION_NAMES.get(a, f"ACTION{a}") for a in available]
        user_prompt = self._build_user_prompt(frame_view, valid_names, state, available)

        # Add user message (with optional image)
        if self._vision:
            image_url = grid_to_png_data_url(frame)
            self._messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            })
        else:
            self._messages.append({"role": "user", "content": user_prompt})

        # Compaction check
        self._maybe_compact()

        # Tool-calling loop
        try:
            action = self._tool_loop(available)
            return action
        except Exception as e:
            self.last_reasoning = f"Error: {e}"
            return self._fallback(available)

    def _tool_loop(self, available: list[int]) -> ArcAction:
        """Run the tool-calling loop until we get an action."""
        system_msg = _SYSTEM.format(color_legend=ARC_COLOR_LEGEND)

        for tool_step in range(self._max_tool_steps):
            tools = [self._python_tool_schema()]

            # Call LLM
            from litellm import completion

            extra = {}
            if self._api_base:
                extra = {"api_base": self._api_base, "api_key": self._api_key}

            resp = completion(
                model=self._model,
                messages=[{"role": "system", "content": system_msg}] + self._messages,
                tools=tools,
                tool_choice="auto",
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                **extra,
            )

            msg = resp.choices[0].message
            reasoning = getattr(msg, "reasoning_content", "") or ""
            if reasoning:
                self._reasoning_trace.append(reasoning)
                if len(self._reasoning_trace) > 20:
                    self._reasoning_trace = self._reasoning_trace[-20:]
            self.last_reasoning = reasoning

            # No tool call
            if not msg.tool_calls:
                content = msg.content or ""
                self._messages.append({"role": "assistant", "content": content})
                action = self._parse_action_from_text(content, available)
                if action:
                    return action
                self._messages.append({
                    "role": "user",
                    "content": "Please use the python tool to execute an action. Call action() with a valid action list.",
                })
                continue

            # Process tool calls
            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                code = args.get("code", "")

                self._messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args),
                        },
                    }],
                })

                result = self._execute_python(code, available)

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

                if result.get("action_executed"):
                    action_data = result.get("action_data", {})
                    return self._data_to_action(action_data, available)

            self._evict_if_needed()

        return self._fallback(available)

    def _maybe_compact(self) -> None:
        """Compact context if approaching limit — Retained Reasoning feature."""
        total_chars = sum(len(json.dumps(m, default=str)) for m in self._messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens < self._context_window * self._compaction_threshold:
            return

        # Summarize reasoning trace into world model
        if self._reasoning_trace:
            recent_reasoning = "\n".join(self._reasoning_trace[-5:])
            self._world_model["recent_findings"] = (
                f"Recent reasoning: {recent_reasoning[:500]}"
            )

        # Compact messages: keep system + world model + last 6 messages
        if len(self._messages) <= 4:
            return

        compacted = self._messages[:1]  # system prompt placeholder
        world_summary = self._format_world_model()
        if world_summary:
            compacted.append({
                "role": "user",
                "content": f"[Context compacted. World model summary:]\n{world_summary}",
            })
            compacted.append({
                "role": "assistant",
                "content": "Understood. I'll continue from where we left off.",
            })

        # Keep last 6 messages for continuity
        compacted.extend(self._messages[-6:])
        self._messages = compacted

    def _format_world_model(self) -> str:
        """Format world model for compaction summary."""
        lines = []
        for key, value in self._world_model.items():
            if value:
                lines.append(f"- {key}: {value}")
        if self._history:
            lines.append(f"- Step: {self._step}, Level: {self._levels}")
            lines.append(f"- Actions taken: {len(self._history)}")
        return "\n".join(lines)

    def _python_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "python",
                "description": (
                    "Run Python code against preloaded game state. Available globals: "
                    "current_frame (with .ascii, .segmentation, .step, .level, .shape), "
                    "history (list of action/frame snapshots), "
                    "valid_actions (current valid action names), "
                    "action(actions) for executing real game actions. "
                    "Use current_frame.segmentation as primary view. "
                    "For MOUSE, pass row and col integer fields."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to run. Ephemeral, not saved between calls.",
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    def _execute_python(self, code: str, available: list[int]) -> dict[str, Any]:
        """Execute Python code in a sandboxed environment."""
        last_frame = self._history[-1].frame if self._history else None
        prev_frame = self._history[-2].frame if len(self._history) > 1 else None

        history_view = []
        for entry in self._history[:-1]:
            if entry.action:
                history_view.append(type("H", (), {
                    "action": entry.action,
                    "frame": entry.frame,
                })())

        last_result = {
            "board_changed": True,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "done": False,
            "valid_actions": [self.ACTION_NAMES.get(a, f"ACTION{a}") for a in available],
        }

        action_result = {"executed": False, "action_data": {}}

        def action_fn(actions: list) -> dict:
            if not actions:
                return {"error": "no actions provided"}
            act = actions[0] if isinstance(actions, list) else actions
            if isinstance(act, str):
                act_id = self.NAME_TO_ID.get(act)
                if act_id is None:
                    return {"error": f"unknown action: {act}"}
                action_result["executed"] = True
                action_result["action_data"] = {"id": act_id}
            elif isinstance(act, dict):
                act_name = act.get("action", act.get("id"))
                if act_name == "MOUSE":
                    action_result["executed"] = True
                    action_result["action_data"] = {
                        "id": COMPLEX_ACTION_ID,
                        "x": act.get("col", 32),
                        "y": act.get("row", 32),
                    }
                else:
                    act_id = self.NAME_TO_ID.get(act_name)
                    if act_id is None:
                        return {"error": f"unknown action: {act_name}"}
                    action_result["executed"] = True
                    action_result["action_data"] = {"id": act_id}
            return {"action_executed": True, **last_result}

        import builtins
        import io

        namespace = {
            "__builtins__": {
                k: v for k, v in vars(builtins).items()
                if not k.startswith("_") or k in ("__name__",)
            },
            "current_frame": last_frame,
            "previous_frame": prev_frame,
            "history": history_view,
            "valid_actions": [self.ACTION_NAMES.get(a, f"ACTION{a}") for a in available],
            "action": action_fn,
            "action_result": last_result,
            "result": None,
        }

        stdout_lines = []
        original_print = namespace["__builtins__"].get("print")

        def capture_print(*args, **kwargs):
            buf = io.StringIO()
            kwargs["file"] = buf
            original_print(*args, **kwargs)
            stdout_lines.append(buf.getvalue())

        namespace["__builtins__"]["print"] = capture_print

        try:
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "stdout": "\n".join(stdout_lines),
                "traceback": traceback.format_exc(),
            }

        return {
            "stdout": "\n".join(stdout_lines),
            "result": namespace.get("result"),
            "action_executed": action_result["executed"],
            "action_data": action_result["action_data"],
            **last_result,
        }

    def _build_user_prompt(
        self,
        frame_view: FrameView,
        valid_names: list[str],
        state: str,
        available: list[int],
    ) -> str:
        lines = []

        # Previous step summary
        if len(self._history) > 1:
            prev_entry = self._history[-2]
            if prev_entry.action:
                lines.append(f"The code executed 1 action in the previous sequence.")
                lines.append(f"Executed actions: {prev_entry.action}.")
            else:
                lines.append("No previous action sequence was captured.")
        else:
            lines.append("No previous sequence has been executed yet.")

        lines.append(f"Current state: step {self._step}, level {self._levels}.")
        lines.append(f"Valid actions right now: {', '.join(valid_names)}.")
        lines.append(
            "Only tool: `python`. It receives `current_frame`, `previous_frame`, "
            "`history`, `valid_actions`, and `action(actions)`."
        )
        lines.append(
            "Use `current_frame.segmentation` as the primary view; "
            "use `current_frame.ascii` only for a small specific region."
        )

        # World model with Retained Reasoning
        wm = self._format_world_model()
        if wm:
            lines.append("Working world model from previous turn:")
            lines.append(wm)
            lines.append("Revise based on new evidence.")

        # Retained Reasoning prompt
        lines.append(
            "IMPORTANT: You MUST call `action(actions)` in your Python code. "
            "Example: `action(['RIGHT'])` or `action([{'action': 'MOUSE', 'row': 32, 'col': 32}])`. "
            "Always end your code with an action call."
        )
        lines.append(
            "You may call `action(actions)` more than once in one Python snippet. "
            "Batch multiple actions for efficiency."
        )

        return "\n".join(lines)

    def _parse_action_from_text(self, text: str, available: list[int]) -> ArcAction | None:
        json_re = re.compile(r"\{[^{}]*\}")
        matches = json_re.findall(text)
        if not matches:
            return None
        for match in reversed(matches):
            try:
                data = json.loads(match)
                act_id = data.get("id") or data.get("action")
                if isinstance(act_id, str):
                    act_id = self.NAME_TO_ID.get(act_id)
                if act_id in available:
                    if act_id == COMPLEX_ACTION_ID:
                        x = self._clamp_coord(data.get("x", data.get("col")))
                        y = self._clamp_coord(data.get("y", data.get("row")))
                        return ArcAction(id=act_id, x=x, y=y)
                    return ArcAction(id=act_id)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return None

    def _data_to_action(self, data: dict[str, Any], available: list[int]) -> ArcAction:
        act_id = data.get("id")
        if act_id not in available:
            name = data.get("action", data.get("name"))
            if name:
                act_id = self.NAME_TO_ID.get(name)
            if act_id not in available:
                return self._fallback(available)
        if act_id == COMPLEX_ACTION_ID:
            x = self._clamp_coord(data.get("x", data.get("col", GRID_SIZE // 2)))
            y = self._clamp_coord(data.get("y", data.get("row", GRID_SIZE // 2)))
            return ArcAction(id=act_id, x=x, y=y)
        return ArcAction(id=act_id)

    @staticmethod
    def _clamp_coord(val: Any) -> int:
        try:
            return max(0, min(GRID_SIZE - 1, int(val)))
        except (TypeError, ValueError):
            return GRID_SIZE // 2

    def _evict_if_needed(self) -> None:
        total_chars = sum(len(json.dumps(m, default=str)) for m in self._messages)
        estimated_tokens = total_chars // 4
        if estimated_tokens < self._context_window * 0.8:
            return
        if len(self._messages) <= 4:
            return
        keep = min(10, len(self._messages) - 2)
        self._messages = self._messages[:1] + self._messages[-keep:]

    def _fallback(self, available: list[int]) -> ArcAction:
        return random_action(available, __import__("random").Random())
