"""One compact dossier per stalled game: every model request on its last, uncleared level."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import typer
from harness import EvalRun, Finished

SECTION = re.compile(r"^\[([A-Z][A-Z :_/a-z0-9-]+)\]", re.M)


class Dossier:
    """Renders the stuck level of one game's event log as markdown."""

    @staticmethod
    def sections(transcript: str) -> dict[str, str]:
        marks = list(SECTION.finditer(transcript))
        out: dict[str, str] = {}
        for m, nxt in zip(marks, [*marks[1:], None], strict=True):
            body = transcript[m.end() : nxt.start() if nxt else len(transcript)].strip()
            out[m.group(1)] = out.get(m.group(1), "") + body
        return out

    @staticmethod
    def clip(text: str, head: int, tail: int) -> str:
        if len(text) <= head + tail:
            return text
        return f"{text[:head]}\n[... {len(text) - head - tail} chars cut ...]\n{text[-tail:]}"

    @classmethod
    def request_lines(cls, i: int, total: int, e: dict[str, Any]) -> list[str]:
        s = cls.sections(e.get("transcript", ""))
        return [
            "",
            f"## Request {i}/{total} (turn {e['analysis_step']}, action #{e['action_num']})",
            "### Feedback the model received",
            cls.clip(s.get("USER PROMPT", ""), 500, 200),
            "### Model reasoning",
            cls.clip(s.get("THINKING", ""), 1200, 1800),
            "### Code it ran",
            cls.clip(s.get("TOOL CALL: python", ""), 500, 500),
            "### Tool output",
            cls.clip(s.get("TOOL RESULT: python", ""), 300, 400),
        ]

    @classmethod
    def render(cls, events: list[dict[str, Any]], fin: Finished) -> str:
        stuck = fin.level + 1
        on_level = [e for e in events if e.get("level") == stuck]
        requests = [e for e in on_level if e["type"] == "analysis"]
        actions = [e for e in on_level if e["type"] == "action"]
        deaths = sum(bool(e.get("game_over")) for e in actions)
        lines = [
            f"# {fin.code}: stalled on level {stuck} of {fin.n_levels} ({fin.state})",
            f"per-level actions/human-baseline: {fin.per_level}",
            f"on this level: {len(requests)} model requests, {len(actions)} actions, {deaths} deaths, "
            f"{len({e['analysis_step'] for e in requests})} turns",
            "",
            "## Board at the start of the level (letter-coded colours)",
            "```",
            next((e["board_ascii"] for e in on_level), ""),
            "```",
        ]
        for i, e in enumerate(requests, 1):
            lines += cls.request_lines(i, len(requests), e)
        return "\n".join(lines)


def main(run_dir: Path, out: Path) -> None:
    run = EvalRun(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    for gid, fin in sorted(run.finished().items()):
        if fin.won:
            continue
        text = Dossier.render(EvalRun.load(run.events(gid)), fin)
        (out / f"{gid}.md").write_text(text)
        print(
            f"{gid}: level {fin.level + 1}/{fin.n_levels}, {len(text) // 1000}K chars"
        )


if __name__ == "__main__":
    typer.run(main)
