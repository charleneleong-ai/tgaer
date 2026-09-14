#!/usr/bin/env python3
"""Measure one explorer variant against the 8-case suite and report RHAE.

Plays every game with the given ``arc_agi3_explorer.py``, writes a scorecard to
``sia-oss/bench/runs/<label>/results/submission.json`` (distinct from the
repo-root ``runs/`` the supervisor grades), then grades it with the same
``evaluate.py`` the SIA harness uses — so a number produced here is in the same
units as the promotion baseline and can be compared against it directly.

    python sia-oss/bench/measure.py --label blocked-reset
    python sia-oss/bench/measure.py --label baseline \\
        --explorer src/tgaer/agents/arc_agi3_explorer.py
    python sia-oss/bench/measure.py --label click-64 --seed 3

Defaults to the working-tree explorer, which is the variant under test.

Each run is logged to W&B as its own run (when ``WANDB_API_KEY`` is set),
grouped by ``--label`` so a seed sweep of one variant collapses into a single
group. This is the hand-driven counterpart to ``supervisor.py``'s logging: the
supervisor records what SIA generations do, this records what *we* try, and
without it a manual sweep survives only in terminal scrollback.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard import GAME_STATE_COLUMNS, game_state_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "sia-oss/bench"
EVALUATE = REPO / "sia-oss/tasks/arc-agi3/data/public/evaluate.py"
DATASET = REPO / "sia-oss/tasks/arc-agi3/data/public"
SUITE = ["lp85", "ls20", "sp80", "sc25", "tu93", "bp35", "sk48", "cn04"]
STATE = REPO / "sia-oss/supervisor_state.json"
# RHAE (%) of the unmodified explorer on this suite, used only as the fallback
# before anything has been promoted — see supervisor.py:INITIAL_BASELINE.
INITIAL_BASELINE = 0.4263


def current_baseline() -> float:
    """The score a variant has to beat, as `supervisor.py` currently holds it.

    Read live rather than copied: the supervisor promotes past the initial value
    into `supervisor_state.json`, so a frozen constant here would keep reporting
    deltas against a baseline that has already moved.
    """
    if STATE.is_file():
        return json.loads(STATE.read_text())["baseline_score"]
    return INITIAL_BASELINE

load_dotenv(REPO / ".env")

app = typer.Typer(add_completion=False)


def log_to_wandb(
    *,
    label: str,
    seed: int,
    max_steps: int,
    explorer: Path,
    results: dict[str, Any],
    submission: dict[str, Any],
) -> None:
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY unset; skipping W&B logging")
        return
    # Local: wandb is a hard dependency but a costly import, and every path
    # through this module that skips logging should not pay for it.
    import wandb

    rhae = results["rhae"]
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "arc-agi-3"),
        group=label,
        name=f"{label}-seed{seed}",
        job_type="bench",
        config={
            "label": label,
            "seed": seed,
            "max_steps": max_steps,
            "explorer": str(explorer.relative_to(REPO))
            if explorer.is_relative_to(REPO)
            else str(explorer),
            # The variant is defined by the file's content, so a sweep that edits
            # a constant in place is still distinguishable after the fact.
            "explorer_sha": hashlib.sha256(explorer.read_bytes()).hexdigest()[:12],
        },
    )
    run.log(
        {
            "rhae": rhae,
            "legacy_total_level_score": results["total_level_score"],
            "delta_vs_baseline": rhae - current_baseline(),
            # results["passed"] counts passing *games*; the table's `passed`
            # column is a per-game bool — distinct things, distinct names.
            "games_passed": results["passed"],
            "levels_cleared": sum(d["levels_completed"] for d in results["details"]),
            **{f"env_score/{d['game']}": d["env_score"] for d in results["details"]},
            "game_state": wandb.Table(
                columns=GAME_STATE_COLUMNS,
                data=game_state_rows(results, submission),
            ),
        }
    )
    run.finish()


@app.command()
def main(
    label: str = typer.Option(
        ..., "--label", help="Name for this variant's run directory."
    ),
    explorer: Path = typer.Option(
        REPO / "src/tgaer/agents/arc_agi3_explorer.py",
        "--explorer",
        help="Explorer file to measure (defaults to the working tree's).",
    ),
    max_steps: int = typer.Option(600, "--max-steps"),
    seed: int = typer.Option(0, "--seed"),
    wandb_log: bool = typer.Option(
        True, "--wandb/--no-wandb", help="Log this run to W&B."
    ),
) -> None:
    """Play the suite with one explorer variant and print its RHAE."""
    run_dir = BENCH / "runs" / label
    out = run_dir / "results" / "submission.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    runner = subprocess.run(
        [
            sys.executable,
            str(BENCH / "arc_runner.py"),
            "--working_dir",
            str(run_dir),
            "--explorer_path",
            str(explorer.resolve()),
            "--dataset_dir",
            str(DATASET),
            "--games",
            json.dumps(SUITE),
            "--max_steps",
            str(max_steps),
            "--seed",
            str(seed),
            "--out",
            str(out),
            "--repo_root",
            str(REPO),
        ],
        cwd=str(REPO),
        text=True,
    )
    if runner.returncode != 0 or not out.is_file():
        raise SystemExit(
            f"runner failed (rc={runner.returncode}); no scorecard at {out}"
        )

    graded = subprocess.run(
        [sys.executable, str(EVALUATE), "--gen-dir", str(run_dir)],
        cwd=str(REPO),
        text=True,
    )
    if graded.returncode != 0:
        raise SystemExit(graded.returncode)

    if wandb_log:
        log_to_wandb(
            label=label,
            seed=seed,
            max_steps=max_steps,
            explorer=explorer.resolve(),
            results=json.loads((run_dir / "results.json").read_text()),
            submission=json.loads(out.read_text()),
        )


if __name__ == "__main__":
    app()
