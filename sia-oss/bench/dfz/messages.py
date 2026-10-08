"""Chat-message helpers shared by the warm-start and win-record hooks; no harness imports."""

from __future__ import annotations

from typing import Any

MARK = "[VERIFIED LEVEL WINS - kept when older history is trimmed]"
END = "[END VERIFIED LEVEL WINS]"


class Messages:
    """Text access, appending, and the pinned win-record block on OpenAI-style messages."""

    @staticmethod
    def text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        return " ".join(str(p.get("text") or "") for p in content or [])

    @staticmethod
    def append_text(message: dict[str, Any], text: str) -> dict[str, Any]:
        m = dict(message)
        if not isinstance(m["content"], list):
            m["content"] = f"{m['content']}\n{text}"
            return m
        parts = [dict(p) for p in m["content"]]
        i = next(i for i, p in enumerate(parts) if p.get("type") == "text")
        parts[i]["text"] = f"{parts[i]['text']}\n{text}"
        m["content"] = parts
        return m

    @staticmethod
    def block(records: dict[int, str]) -> str:
        body = "\n\n".join(records[k] for k in sorted(records))
        return (
            f"{MARK}\nWhat cleared each completed level, recorded by the harness. Any goal "
            f"hypothesis for the current level should be consistent with this evidence.\n{body}\n{END}"
        )

    @staticmethod
    def strip(text: str) -> str:
        start = text.find(MARK)
        if start < 0:
            return text
        end = text.find(END, start)
        return (
            (text[:start] + text[end + len(END) :]).lstrip("\n") if end >= 0 else text
        )

    @staticmethod
    def pin(message: dict[str, Any], records: dict[int, str]) -> dict[str, Any]:
        block = Messages.block(records)
        pinned = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            parts = [
                p
                for p in content
                if not (
                    p.get("type") == "text" and str(p.get("text", "")).startswith(MARK)
                )
            ]
            pinned["content"] = [{"type": "text", "text": block}, *parts]
        else:
            pinned["content"] = f"{block}\n\n{Messages.strip(str(content or ''))}"
        return pinned
