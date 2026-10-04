"""Per-level warm-start records (history, opener, retained functions) and the hook that seeds a ToolAgent with them."""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any

from harness import EvalRun, level_key, parse_episode_id
from messages import Messages

RETAINED = re.compile(r"Retained your function (\w+)\(")
DROPPED = re.compile(
    r"(?:Your previous function (\w+) is no longer retained|Your function (\w+) was not retained)"
)
CLEARED = "retained functions were cleared"


class WarmStart:
    """Builds per-level warm-start records and installs them on a ToolAgent class."""

    @staticmethod
    def first_request(
        requests_path: Path, events_path: Path, level: int
    ) -> dict[str, Any]:
        steps: dict[int, int] = {}
        for e in EvalRun.load(events_path):
            if e["type"] == "analysis":
                steps.setdefault(e["analysis_step"], e["level"])
        first = min(s for s, lv in steps.items() if lv == level)
        with requests_path.open() as f:
            for line in f:
                if (r := json.loads(line)).get("analysis_step") == first:
                    return r
        raise LookupError(f"no request logged for step {first}")

    @staticmethod
    def snippet_defs(code: str) -> dict[str, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}
        return {
            n.name: ast.get_source_segment(code, n) or ""
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
        }

    @staticmethod
    def call_code(call: dict[str, Any]) -> str | None:
        try:
            return json.loads(call["function"]["arguments"]).get("code", "")
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def kept_functions(history: list[dict[str, Any]]) -> dict[str, str]:
        defined: dict[str, str] = {}
        kept: dict[str, str] = {}
        for m in history:
            for call in m.get("tool_calls") or []:
                if (code := WarmStart.call_code(call)) is not None:
                    defined.update(WarmStart.snippet_defs(code))
            if m.get("role") != "tool":
                continue
            body = Messages.text(m)
            if CLEARED in body:
                kept.clear()
            for name in RETAINED.findall(body):
                if name in defined:
                    kept[name] = defined[name]
            for a, b in DROPPED.findall(body):
                kept.pop(a or b, None)
        return kept

    @staticmethod
    def record(request: dict[str, Any]) -> dict[str, Any]:
        messages = request["messages"]
        history = messages[1:-1]
        return {
            "history": history,
            "opener": messages[-1],
            "kept": WarmStart.kept_functions(history),
        }

    @staticmethod
    def record_for(
        state_path: Path, records: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        parsed = parse_episode_id(Path(state_path).name.split("_p0_")[0])
        return records.get(level_key(*parsed[:2])) if parsed else None

    @staticmethod
    def install(agent_cls: type, records: dict[str, dict[str, Any]]) -> None:
        ensure = agent_cls._ensure_session
        build_message = agent_cls._build_user_message

        def _ensure_session(agent: Any, state_path: Path) -> Any:
            out = ensure(agent, state_path)
            if not agent.__dict__.get("_warm_done"):
                agent._warm_done = True
                if (rec := WarmStart.record_for(state_path, records)) is not None:
                    agent._history_messages = copy.deepcopy(rec["history"])
                    agent._kept_functions = dict(rec["kept"])
                    if rec.get("win_records"):
                        agent._win_records = {
                            int(k): v for k, v in rec["win_records"].items()
                        }
                    agent._warm_opener = rec["opener"]
            return out

        def _build_user_message(
            agent: Any, user_prompt: str, current_frame: Any
        ) -> dict[str, Any]:
            opener = agent.__dict__.pop("_warm_opener", None)
            if opener is None:
                return build_message(agent, user_prompt, current_frame)
            return copy.deepcopy(opener)

        agent_cls._ensure_session = _ensure_session
        agent_cls._build_user_message = _build_user_message
