"""Keep the model's reasoning text only on its most recent turns; older turns keep their content and tool calls.

Stripping is batched: nothing changes until more than KEEP + DRAIN assistant turns carry reasoning, then the
oldest are stripped down to KEEP. The cached prompt prefix therefore changes once per DRAIN turns, not every turn.
"""

from __future__ import annotations

import os
from typing import Any

KEEP_TURNS = int(os.environ.get("ARC3_REASONING_KEEP_TURNS", "4"))
DRAIN_TURNS = int(os.environ.get("ARC3_REASONING_DRAIN_TURNS", "12"))
KEYS = ("reasoning", "reasoning_content")


class ReasoningWindow:
    """Installed on dfranzen's ToolAgent class, after its own history trimming."""

    @staticmethod
    def has_reasoning(message: dict[str, Any]) -> bool:
        return message.get("role") == "assistant" and any(message.get(k) for k in KEYS)

    @staticmethod
    def apply(history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        carrying = [
            i for i, m in enumerate(history) if ReasoningWindow.has_reasoning(m)
        ]
        if len(carrying) <= KEEP_TURNS + DRAIN_TURNS:
            return history, 0
        strip = set(carrying[: len(carrying) - KEEP_TURNS])
        trimmed = [
            {k: v for k, v in m.items() if k not in KEYS} if i in strip else m
            for i, m in enumerate(history)
        ]
        return trimmed, len(strip)

    @staticmethod
    def install(agent_cls: type) -> None:
        persistent = agent_cls._persistent_history_messages

        def _persistent_history_messages(
            agent: Any, messages: list[dict[str, Any]], **kwargs: Any
        ) -> list[dict[str, Any]]:
            history, stripped = ReasoningWindow.apply(
                persistent(agent, messages, **kwargs)
            )
            if stripped:
                agent._note_history_evicted()
                agent._reasoning_stripped = (
                    agent.__dict__.get("_reasoning_stripped", 0) + stripped
                )
                print(
                    f"[reasoning-window] stripped reasoning from {stripped} turns "
                    f"({agent._reasoning_stripped} so far)",
                    flush=True,
                )
            return history

        agent_cls._persistent_history_messages = _persistent_history_messages
