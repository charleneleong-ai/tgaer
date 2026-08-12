"""Play one game and print every turn, so a failure can be read rather than guessed.

Seven mock runs have reported "0 levels completed" without once showing what the
agent actually did with its 150 actions. This prints the action, how it was
chosen, and what it changed — the minimum needed to tell a stuck agent from a
blind one from a mis-clicking one.

    .venv/bin/python scripts/trace_game.py --game sk48 --steps 60
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

from dotenv import load_dotenv  # noqa: E402

for env_file in (ROOT / ".env", ROOT.parents[1] / ".env"):
    if env_file.exists():
        load_dotenv(env_file)
        break

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

sys.path.insert(0, str(ROOT / "agent"))
import my_agent as ma  # noqa: E402

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


class TracingAgent(ma.MyAgent):
    """MyAgent that records what it chose and what changed, per turn."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace: list[dict[str, Any]] = []

    def choose_action(self, frames: Any, latest_frame: Any) -> Any:
        before = dict(self.stats)
        action = super().choose_action(frames, latest_frame)
        path = next(
            (k for k, v in self.stats.items()
             if k in {"tool_call", "tool_from_text", "raw_text_parsed",
                      "random_fallback", "no_model"} and v > before.get(k, 0)),
            "unknown",
        )
        self.trace.append({
            "step": len(self.trace) + 1,
            "action": self._last_action_name,
            "path": path,
            "effect": self.memory.describe_last(),
            "level": latest_frame.levels_completed,
            "state": str(latest_frame.state).replace("GameState.", ""),
            "available": list(latest_frame.available_actions or []),
        })
        return action


@app.command()
def main(
    game: str = typer.Option("sk48", help="Game id to trace."),
    steps: int = typer.Option(60, help="Actions to take."),
    host: str = typer.Option("http://127.0.0.1:8000/v1", help="vLLM base URL."),
    model: str = typer.Option("arc-agent", help="Served model name."),
    show: int = typer.Option(40, help="How many turns to print."),
) -> None:
    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game)
    if env is None:
        raise typer.BadParameter(f"could not create env for {game!r}")

    agent = TracingAgent(
        card_id="trace", game_id=game, agent_name=f"trace.{game}",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["trace"],
    )
    agent._llm = ma.HTTPChatBackend(host, model)
    agent.MAX_ACTIONS = steps
    agent.main()

    table = Table(title=f"{game}: first {show} turns", title_style="bold")
    for col in ("step", "action", "how", "level", "effect"):
        table.add_column(col, overflow="fold")
    for row in agent.trace[:show]:
        effect = row["effect"]
        style = "red" if "NOTHING" in effect else ""
        table.add_row(
            str(row["step"]), row["action"], row["path"], str(row["level"]),
            (effect[:90] + "…") if len(effect) > 90 else effect, style=style,
        )
    console.print(table)

    actions = Counter(r["action"].split("@")[0] for r in agent.trace)
    dead = sum(1 for r in agent.trace if "NOTHING" in r["effect"])
    console.print(f"\n[bold]actions[/]: {dict(actions)}")
    console.print(f"[bold]no-effect turns[/]: {dead}/{len(agent.trace)} "
                  f"({dead / max(1, len(agent.trace)):.0%})")
    console.print(f"[bold]decision paths[/]: {dict(agent.stats)}")
    console.print(f"[bold]levels completed[/]: {agent.frames[-1].levels_completed}")
    console.print(f"[bold]final state[/]: {agent.frames[-1].state}")


if __name__ == "__main__":
    app()
