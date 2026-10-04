"""Pinned level-win records for dfranzen's ToolAgent: what won each level, re-pinned whenever history is trimmed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness import EvalRun
from messages import Messages

try:
    from inference.agent.action_names import to_model_action
    from inference.agent.runtime_state import Frame, HistoryEntry
    from inference.utils.frame_diff import compute_frame_diff
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer
except ImportError as exc:
    raise ImportError(
        "win_records needs dfranzen's harness importable: call Harness.install(DFZ_SRC) first"
    ) from exc

DIFF_BUDGET_TOKENS = 600
LAST_ACTIONS = 8


class WinRecords:
    """Builds, stores and pins per-level win records on a ToolAgent instance."""

    def __init__(self, render_diff: Any, max_group: int) -> None:
        self.render_diff = render_diff
        self.max_group = max_group

    @staticmethod
    def history_from_events(path: Path) -> list[Any]:
        return [
            HistoryEntry(
                action=e.get("action_name") or "RESET",
                frame=Frame(
                    tuple(tuple(r) for r in e["board"]), e["action_num"], e["level"]
                ),
            )
            for e in EvalRun.load(path)
            if e["type"] in ("initial", "action")
        ]

    def record(self, entries: list[Any], completed: int) -> str | None:
        framed = [e for e in entries if e.frame is not None]
        on_level = [e for e in framed if e.frame.level == completed]
        after = [e for e in framed if e.frame.level > completed]
        if len(on_level) < 2 or not after:
            return None
        start, pre_win = on_level[0].frame, on_level[-1].frame
        # a RESET after a game over is a counted action, as in the game's own tally
        actions = [to_model_action(e.action) or str(e.action) for e in on_level[1:]]
        winning = to_model_action(after[0].action) or str(after[0].action)
        lines = [
            f"Level {completed}: cleared by {winning!r} after {len(actions) + 1} actions on the level; "
            f"its last actions were {', '.join([*actions[-LAST_ACTIONS:], winning])}.",
            "Board changes from that level's START to the moment before the winning action "
            "(the configuration that won):",
            *self.diff_lines(start.grid, pre_win.grid),
        ]
        return "\n".join(lines)

    def diff_lines(self, before: Any, after: Any) -> list[str]:
        if len(before) != len(after) or before == after:
            return ["(no board difference recorded)"]
        diff = compute_frame_diff(
            before,
            after,
            segment_layer(before, ARC_COLOR_CHARS).get("nodes", []),
            segment_layer(after, ARC_COLOR_CHARS).get("nodes", []),
            max_group_match=self.max_group,
        )
        cells = len(after) * max((len(r) for r in after), default=0)
        return self.render_diff(diff, DIFF_BUDGET_TOKENS, total_cells=cells)

    def opener_block(self, agent: Any, kwargs: dict[str, Any]) -> str | None:
        summary = kwargs.get("previous_step_summary") or {}
        frame = kwargs.get("current_frame")
        if (
            not summary.get("level_transition")
            or summary.get("run_complete")
            or frame is None
        ):
            return None
        records = agent.__dict__.setdefault("_win_records", {})
        completed = int(frame.level) - 1
        if completed in records:
            return None
        if (
            record := self.record(list(kwargs.get("history_entries") or []), completed)
        ) is None:
            return None
        records[completed] = record
        return Messages.block({completed: record})

    def install(self, agent_cls: type) -> None:
        build_prompt = agent_cls._build_user_prompt
        trim = agent_cls._trim_messages_for_context

        def _build_user_prompt(agent: Any, action_num: int, **kwargs: Any) -> str:
            text = build_prompt(agent, action_num, **kwargs)
            block = self.opener_block(agent, kwargs)
            return text if block is None else f"{text}\n{block}"

        def _trim_messages_for_context(
            agent: Any, messages: list[dict[str, Any]], **kwargs: Any
        ) -> list[dict[str, Any]]:
            records = agent.__dict__.get("_win_records") or {}
            if not records:
                return trim(agent, messages, **kwargs)
            reserve = agent._estimate_request_input_tokens(
                [{"role": "user", "content": Messages.block(records)}]
            )
            kwargs["extra_safety_tokens"] = (
                kwargs.get("extra_safety_tokens", 0) + reserve
            )
            out = trim(agent, messages, **kwargs)
            if (
                len(out) < len(messages)
                and len(out) > 1
                and out[1].get("role") == "user"
            ):
                out = [out[0], Messages.pin(out[1], records), *out[2:]]
            return out

        agent_cls._build_user_prompt = _build_user_prompt
        agent_cls._trim_messages_for_context = _trim_messages_for_context
