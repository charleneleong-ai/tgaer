#!/usr/bin/env python3
"""Play the suite with explorer warmup + an LLM-written policy, then gate it.

The policy is asked for once per game, after `--warmup` actions of explorer
play, and only takes over if it survives falsification against the recorded
transitions (`arc_agi3_codegen.validate`). Every step it declines, throws on, or
answers with an unavailable action falls straight back to the explorer, so the
floor is explorer behaviour minus nothing.

Writes a scorecard in the same shape `measure.py` does, so the result is graded
by the same `evaluate.py` and judged by the same `gate.py` — a policy ships only
if RHAE improves *and* no game regresses.

    python sia-oss/bench/codegen_run.py --label codegen-v1 --gate-against final

Requires an OpenAI-compatible endpoint; point `--base-url` at a local vLLM for
development or at the in-kernel one in the competition notebook.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import typer

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "sia-oss/bench"
EVALUATE = REPO / "sia-oss/tasks/arc-agi3/data/public/evaluate.py"
SUITE = ["lp85", "ls20", "sp80", "sc25", "tu93", "bp35", "sk48", "cn04"]

sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

app = typer.Typer(add_completion=False)


class VLLMBackend:
    """OpenAI-compatible chat client with thinking disabled.

    Thinking off is not a preference: a reasoning model spends the whole token
    budget reasoning and returns empty content. See `arc_agi3_codegen`.
    """

    def __init__(self, base_url: str, model: str = "codegen") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 2048) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)["choices"][0]["message"]["content"]


def install_codegen(backend: Any, warmup: int, report: dict[str, str], rounds: int) -> Any:
    """Wrap the explorer's ``act`` with record -> ask -> policy-first-with-fallback.

    Patching `act` rather than subclassing keeps `load_agent_class(None,
    "explorer")` — and therefore the whole scoring path — exactly as the
    measured baseline uses it, so the only difference between this run and the
    baseline is the policy.
    """
    import tgaer.agents.arc_agi3_explorer as explorer_module
    from tgaer.agents import arc_agi3_codegen as cg

    original = explorer_module.ExplorerArcAgi3Agent.act
    to_arc = explorer_module.to_arc

    def act(self: Any, observation: Any) -> Any:
        state = self.__dict__.setdefault(
            "_codegen",
            {"evidence": None, "policy": None, "pending": {}, "steps": 0, "used": 0},
        )
        obs = observation if isinstance(observation, dict) else {}
        frame = obs.get("frame") or []
        grid = np.asarray(frame[-1]) if frame else None
        level = obs.get("levels_completed", 0)
        available = list(obs.get("available_actions") or [])

        if state["evidence"] is None:
            state["evidence"] = cg.GameEvidence(
                game=report.get("game", "?"), available_actions=available
            )
        evidence = state["evidence"]
        if not evidence.available_actions:
            evidence.available_actions = available

        pending = state["pending"]
        if pending and grid is not None and pending.get("grid") is not None:
            evidence.transitions.append(
                cg.Transition(
                    grid=pending["grid"],
                    action=pending["action"],
                    click=pending["click"],
                    next_grid=grid,
                    level=pending["level"],
                    level_after=level,
                )
            )

        state["steps"] += 1
        if state["steps"] == warmup and state["policy"] is None:
            started = time.monotonic()
            policy, reason = cg.refine_policy(backend, evidence, rounds=rounds)
            state["policy"] = policy
            report["reason"] = f"{reason} ({time.monotonic() - started:.1f}s)"

        chosen = original(self, observation)  # the explorer always has an answer
        if state["policy"] is not None and grid is not None:
            try:
                proposal = state["policy"](grid, list(available), state.setdefault("mem", {}))
            except Exception:  # noqa: BLE001 — a throwing policy just defers
                proposal = None
            primitive = to_primitive(proposal, available)
            if primitive is not None:
                chosen = to_arc(primitive)
                state["used"] += 1

        report["policy_actions"] = str(state["used"])
        pending.update(
            grid=grid,
            action=int(getattr(chosen, "id", 0)),
            click=(int(chosen.y), int(chosen.x)) if getattr(chosen, "x", None) is not None else None,
            level=level,
        )
        return chosen

    explorer_module.ExplorerArcAgi3Agent.act = act
    return original


def to_primitive(proposal: Any, available: list[int]) -> tuple[Any, ...] | None:
    """A policy's answer as an explorer primitive, or None to defer.

    Anything malformed, out of range, or not currently available defers rather
    than raising: the explorer's action is already in hand, so a bad proposal
    costs nothing.
    """
    from tgaer.agents import arc_agi3_codegen as cg
    from tgaer.envs.arc_agi3.arc_agi3_api import COMPLEX_ACTION_ID

    if (cell := cg.as_click(proposal)) is not None:
        if COMPLEX_ACTION_ID not in available:
            return None
        row, col = cell
        return ("click", row, col) if 0 <= row < 64 and 0 <= col < 64 else None
    if isinstance(proposal, int) and proposal in available:
        return ("act", proposal)
    return None


@app.command()
def main(
    label: str = typer.Option(..., "--label"),
    base_url: str = typer.Option("http://localhost:8001/v1", "--base-url"),
    warmup: int = typer.Option(80, "--warmup", help="Explorer actions before asking."),
    max_steps: int = typer.Option(600, "--max-steps"),
    rounds: int = typer.Option(
        4, "--rounds", help="Refinement attempts per game; each replays history, costing no game actions."
    ),
    gate_against: str | None = typer.Option(None, "--gate-against"),
) -> None:
    """Play every game with a generated policy and grade the result."""
    os.chdir(REPO)
    os.environ.setdefault("ARC_TGAER_REPO", str(REPO))

    import importlib.util

    spec = importlib.util.spec_from_file_location("arc_runner", BENCH / "arc_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner._load_local(
        "tgaer.agents.arc_agi3_explorer", REPO / "src/tgaer/agents/arc_agi3_explorer.py"
    )

    from tgaer.evaluation.arc_agi3_score_local import (
        OperationMode,
        arc_agi,
        load_agent_class,
        play,
        require_starter,
    )

    backend = VLLMBackend(base_url)
    run_dir = BENCH / "runs" / label
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    require_starter()
    scorecards: list[dict[str, Any]] = []
    for game in SUITE:
        report: dict[str, str] = {"game": game, "reason": "policy never requested"}
        original = install_codegen(backend, warmup, report, rounds)
        try:
            arc = arc_agi.Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(REPO / "environment_files"),
            )
            row = play(load_agent_class(None, "explorer"), game, arc, None, max_steps, seed=0)
            row.pop("_agent", None)
            if row.get("error"):
                scorecards.append({"game": game, "error": row["error"]})
                continue
            rows: list[dict[str, Any]] = []
            for env in getattr(arc.get_scorecard(), "environments", None) or []:
                if str(env.id).startswith(game + "-"):
                    for arcade_run in env.runs:
                        rows += runner._run_levels(arcade_run)
            scorecards.append({"game": game, "levels": runner._level_summary(rows), "row": row})
        finally:
            import tgaer.agents.arc_agi3_explorer as explorer_module

            explorer_module.ExplorerArcAgi3Agent.act = original
        print(
            f"[{game}] policy actions={report.get('policy_actions', '0')} :: {report['reason']}",
            flush=True,
        )

    (run_dir / "results" / "submission.json").write_text(
        json.dumps({"scorecards": scorecards}, indent=2)
    )
    subprocess.run(
        [sys.executable, str(EVALUATE), "--gen-dir", str(run_dir)], cwd=str(REPO), text=True
    )
    if gate_against:
        from gate import report as gate_report

        if not gate_report(run_dir, BENCH / "runs" / gate_against):
            raise SystemExit(1)


if __name__ == "__main__":
    app()
