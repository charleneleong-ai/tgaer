"""When a level stalls, the harness itself tries what the model has not: each untried action, then one
click on each kind of object. The observed effects are shown to the model as facts on its next turn.

Runs once per level, at the start of a turn (before the harness loads the board for the opener), through
the same step_env path as the model's own action() calls, so the harness's death/no-op guards still apply.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

from inference.agent.action_names import to_model_action
from inference.agent.runtime_state import load_runtime_state

PROBE_AFTER_TOKENS = int(os.environ.get("ARC3_STALL_PROBE_AFTER_TOKENS", "80000"))
MAX_PROBE_ACTIONS = 20
MAX_CLICK_KINDS = 12
SKIP = {"RESET", "UNDO"}


class StallProbe:
    """Per-agent stall detection and the probe itself, installed on dfranzen's ToolAgent class."""

    @staticmethod
    def objects(grid: tuple[tuple[int, ...], ...]) -> list[tuple[int, int, int, int]]:
        """One representative cell per (colour, size) kind of 4-connected object, largest kind (background) excluded."""
        rows, cols = len(grid), len(grid[0]) if grid else 0
        seen = [[False] * cols for _ in range(rows)]
        kinds: dict[tuple[int, int], tuple[int, int]] = {}
        for r in range(rows):
            for c in range(cols):
                if seen[r][c]:
                    continue
                colour, cells, q = grid[r][c], [], deque([(r, c)])
                seen[r][c] = True
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if (
                            0 <= ny < rows
                            and 0 <= nx < cols
                            and not seen[ny][nx]
                            and grid[ny][nx] == colour
                        ):
                            seen[ny][nx] = True
                            q.append((ny, nx))
                cells.sort()
                kinds.setdefault((colour, len(cells)), cells[len(cells) // 2])
        ordered = sorted(kinds.items(), key=lambda kv: kv[0][1])
        return [(colour, size, rc[0], rc[1]) for (colour, size), rc in ordered[:-1]]

    @staticmethod
    def tried(entries: list[Any], level: int) -> set[str]:
        return {
            to_model_action(e.action) or str(e.action)
            for e in entries
            if e.frame is not None and e.frame.level == level
        }

    @staticmethod
    def changed_cells(a: Any, b: Any) -> int:
        return sum(x != y for ra, rb in zip(a.grid, b.grid) for x, y in zip(ra, rb))

    @staticmethod
    def plan(
        entries: list[Any], frame: Any, valid_actions: list[str]
    ) -> list[dict[str, Any]]:
        done = StallProbe.tried(entries, frame.level)
        keys = [
            {"action": a}
            for a in valid_actions
            if a not in SKIP and a != "MOUSE" and a not in done
        ]
        clicks = []
        if "MOUSE" in valid_actions:
            clicks = [
                {
                    "action": "MOUSE",
                    "row": r,
                    "col": c,
                    "kind": f"colour {colour}, {size} cells",
                }
                for colour, size, r, c in StallProbe.objects(frame.grid)[
                    :MAX_CLICK_KINDS
                ]
            ]
        return (keys + clicks)[:MAX_PROBE_ACTIONS]

    @staticmethod
    def run(state_path: Path, step_env: Any, valid_actions: list[str]) -> list[str]:
        frame, entries = load_runtime_state(state_path)
        lines = []
        for step in StallProbe.plan(entries, frame, valid_actions):
            label = (
                step.get("kind")
                and f"MOUSE at row {step['row']}, col {step['col']} ({step['kind']})"
                or step["action"]
            )
            payload = (
                step_env({"actions": [{k: v for k, v in step.items() if k != "kind"}]})
                or {}
            )
            after, _ = load_runtime_state(state_path)
            if not payload.get("executed", True) or after is None:
                lines.append(f"- {label}: refused by the harness guards")
                continue
            n = StallProbe.changed_cells(frame, after)
            if after.level != frame.level or payload.get("level_completed"):
                lines.append(f"- {label}: the level was completed")
                return lines
            if payload.get("game_over"):
                lines.append(f"- {label}: game over (the level was reset)")
                return lines
            lines.append(
                f"- {label}: {'changed ' + str(n) + ' cells' if n else 'no visible change'}"
            )
            frame = after
        return lines

    @staticmethod
    def install(agent_cls: type) -> None:
        analyze = agent_cls.analyze
        build_prompt = agent_cls._build_user_prompt

        def _analyze(
            agent: Any,
            state_path: Path,
            action_num: int,
            valid_actions: list[str] | None = None,
            step_env: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            frame, _ = (
                load_runtime_state(state_path)
                if Path(state_path).exists()
                else (None, [])
            )
            if frame is not None and agent.__dict__.get("_probe_level") != frame.level:
                if agent.__dict__.get("_probe_level_seen") != frame.level:
                    agent._probe_level_seen = frame.level
                    agent._probe_tokens_base = agent._session_generated_tokens
                stalled = (
                    agent._session_generated_tokens - agent._probe_tokens_base
                    >= PROBE_AFTER_TOKENS
                )
                if stalled and step_env is not None and valid_actions:
                    agent._probe_level = frame.level
                    try:
                        lines = StallProbe.run(
                            Path(state_path), step_env, list(valid_actions)
                        )
                    except (
                        Exception
                    ) as exc:  # a failed probe must never cost the model its turn
                        print(f"[stall-probe] skipped: {exc!r}", flush=True)
                        lines = []
                    if lines:
                        print(
                            f"[stall-probe] level {frame.level}: {len(lines)} probe actions",
                            flush=True,
                        )
                        agent._probe_report = lines
            return analyze(
                agent, state_path, action_num, valid_actions, step_env, *args, **kwargs
            )

        def _build_user_prompt(agent: Any, action_num: int, **kwargs: Any) -> str:
            prompt = build_prompt(agent, action_num, **kwargs)
            report = agent.__dict__.pop("_probe_report", None)
            if report:
                prompt += (
                    "\n[Harness probe] Progress on this level had stalled, so the harness tried each untried "
                    "action and one click on each kind of object. Observed effects, in order:\n"
                    + "\n".join(report)
                )
            return prompt

        agent_cls.analyze = _analyze
        agent_cls._build_user_prompt = _build_user_prompt
