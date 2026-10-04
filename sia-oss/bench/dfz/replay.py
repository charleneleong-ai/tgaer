"""Replay a recorded dfranzen game through the offline ARC-AGI-3 environment and check the boards match."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import arc_agi
import typer
from arc_agi import OperationMode
from arcengine import GameAction
from harness import EvalRun, Harness

CLICK = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")

Prefix = list[tuple[str, dict[str, int]]]


class RecordedGame:
    """A recorded run's action stream, replayable against the offline game."""

    def __init__(self, events_path: Path, env_files: Path | None = None) -> None:
        self.game_id = events_path.name.split("_p0_")[0]
        self.events = EvalRun.load(events_path)
        self.actions = [e for e in self.events if e["type"] == "action"]
        self.env_files = env_files or Harness.env_files()

    @staticmethod
    def click(event: dict[str, Any]) -> dict[str, int] | None:
        if m := CLICK.match(event.get("action_display") or ""):
            return {"x": int(m.group(2)), "y": int(m.group(1))}
        return None

    @staticmethod
    def board(frame: Any) -> list[list[int]]:
        layers = frame.frame
        return [list(map(int, row)) for row in (layers[-1] if layers else [])]

    def prefix(self, level: int) -> Prefix:
        if level <= 1:
            return []
        out: Prefix = []
        for e in self.actions:
            out.append((e["action_name"], self.click(e) or {}))
            if e["level"] >= level:
                break
        return out

    def start_board(self, level: int) -> list[list[int]]:
        if level <= 1:
            return self.events[0]["board"]
        return next(e["board"] for e in self.actions if e["level"] == level)

    def make_env(self) -> Any:
        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE, environments_dir=str(self.env_files)
        )
        return arc.make(self.game_id)

    def replay(self, stop_at_level: int | None = None) -> dict[str, Any]:
        env = self.make_env()
        frame = env.reset()
        mismatches: list[int] = []
        done = 0
        todo = self.actions if stop_at_level is None or stop_at_level > 1 else []
        for event in todo:
            data = self.click(event)
            action = GameAction[event["action_name"]]
            frame = env.step(action, data=data) if data else env.step(action)
            done += 1
            if self.board(frame) != event["board"]:
                mismatches.append(event["action_num"])
            if stop_at_level is not None and event["level"] >= stop_at_level:
                break
        return {
            "game": self.game_id,
            "replayed": done,
            "of": len(self.actions),
            "mismatches": mismatches,
            "levels_completed": getattr(frame, "levels_completed", None),
        }


def main(events: Path, stop_at_level: int | None = typer.Argument(None)) -> None:
    r = RecordedGame(events).replay(stop_at_level)
    print(
        f"{r['game']}: replayed {r['replayed']}/{r['of']} actions, {len(r['mismatches'])} board mismatches "
        f"{r['mismatches'][:10]}, levels_completed={r['levels_completed']}"
    )


if __name__ == "__main__":
    typer.run(main)
