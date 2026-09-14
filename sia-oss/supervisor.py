#!/usr/bin/env python3
"""Cross-generation supervisor for the OSS-sia ARC-AGI-3 loop.

Runs after each `sia run` generation finishes. It is the missing piece
between our setup and NVIDIA AVO's: AVO pairs its propose/apply/evaluate
loop with persistent memory that "carries forward prior implementations,
evaluation results... and accumulated reasoning" plus a supervisor that
"monitors the broader trajectory for stagnation... and can redirect the
main agent toward alternative strategies." This script is that memory +
redirection mechanism for our loop:

1. Reads the generation's results.json and diffs its arc_agi3_explorer.py
   against the current reference copy.
2. If the diff is empty (no behavioral change attempted) or the score
   regressed/held flat, appends a one-line closed entry to AGENTS.md so
   the next generation doesn't retread the same ground.
3. If RHAE (the official ARC-AGI-3 metric, in percent — see evaluate.py)
   genuinely improved over BASELINE, promotes the generation's explorer.py
   into the reference copy — future generations build on the improvement
   instead of re-discovering it — and prints a loud success banner. The
   legacy unweighted ``total_level_score`` is still logged for continuity
   but never decides promotion.
4. Logs the generation's score trajectory and per-game/per-level scorecard
   to a single continuous W&B run (when WANDB_API_KEY is set), so the
   self-improvement loop's progress and each generation's game outcomes
   are inspectable outside the AGENTS.md ledger.

Usage: python sia-oss/supervisor.py --run-id N --gen M
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))
from scorecard import GAME_STATE_COLUMNS, game_state_rows  # noqa: E402

app = typer.Typer(add_completion=False)

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "sia-oss/tasks/arc-agi3/reference/arc_agi3_explorer.py"
AGENTS_MD = REPO / "sia-oss/tasks/arc-agi3/reference/AGENTS.md"
STATE = REPO / "sia-oss/supervisor_state.json"
# Official RHAE (%) of the unmodified explorer on the 8-case suite. The old
# unweighted total_level_score baseline was 131.08; see evaluate.py for why
# the two differ so much (level-index weighting + completion cap).
INITIAL_BASELINE = 0.4263

# The supervisor is often invoked from a shell that never sourced .env, which
# silently disabled W&B logging (no WANDB_API_KEY) — load it here instead.
load_dotenv(REPO / ".env")


def load_state() -> dict[str, Any]:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {"baseline_score": INITIAL_BASELINE}


def save_state(state: dict[str, Any]) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def current_baseline() -> float:
    return load_state()["baseline_score"]


def set_baseline(score: float) -> None:
    state = load_state()
    state["baseline_score"] = score
    save_state(state)


def wandb_run_id() -> str:
    """One continuous W&B run across every generation, resumed per-invocation
    since each `supervisor.py` call is a fresh process — this gives a single
    score-over-generations trajectory rather than a scattered run per call."""
    state = load_state()
    if "wandb_run_id" not in state:
        state["wandb_run_id"] = f"sia-supervisor-{uuid.uuid4().hex[:8]}"
        save_state(state)
    return state["wandb_run_id"]


def gen_dir(run_id: int, gen: int) -> Path:
    return REPO / f"runs/run_{run_id}/gen_{gen}"


def load_results(path: Path) -> dict[str, Any]:
    results_path = path / "results.json"
    if not results_path.is_file():
        raise SystemExit(f"no results.json at {results_path} — did the generation finish?")
    return json.loads(results_path.read_text())


def load_submission(path: Path) -> dict[str, Any] | None:
    """The raw per-level scorecards `target_agent.py` writes, if present —
    richer than results.json's per-game summary (per-level action counts and
    ratios), used for the W&B game-state table."""
    submission_path = path / "results" / "submission.json"
    if not submission_path.is_file():
        return None
    return json.loads(submission_path.read_text())


def diff_summary(candidate: Path) -> tuple[bool, str]:
    """(changed, unified-diff-stat) between candidate and the current reference."""
    ref_lines = REFERENCE.read_text().splitlines(keepends=True)
    cand_lines = candidate.read_text().splitlines(keepends=True)
    if ref_lines == cand_lines:
        return False, "byte-identical to reference — no behavioral change attempted"
    diff = list(difflib.unified_diff(ref_lines, cand_lines, lineterm=""))
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    return True, f"+{added}/-{removed} lines changed vs reference"


def parse_diff_counts(diff_note: str) -> tuple[int, int]:
    added_str, removed_str = diff_note.split(" lines")[0].lstrip("+").split("/-")
    return int(added_str), int(removed_str)


DRY_RUN = False


def append_ledger_row(run_id: int, gen: int, hypothesis: str, result: str, verdict: str) -> None:
    if DRY_RUN:
        return
    text = AGENTS_MD.read_text()
    marker = "## Do this"
    idx = text.index(marker)
    new_row = f"| {hypothesis} (OSS sia run_{run_id}/gen_{gen}) | {result} — {verdict} |\n"
    table_end = text.rindex("|\n", 0, idx) + 2  # just past the last table row
    updated = text[:table_end] + new_row + text[table_end:]
    AGENTS_MD.write_text(updated)


def log_to_wandb(
    *,
    run_id: int,
    gen: int,
    score: float,
    legacy_score: float,
    baseline_before: float,
    passed: int,
    changed: bool,
    diff_note: str,
    verdict: str,
    promoted: bool,
    results: dict[str, Any],
    submission: dict[str, Any] | None,
) -> None:
    if not os.getenv("WANDB_API_KEY"):
        return
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping W&B logging")
        return

    added, removed = parse_diff_counts(diff_note) if changed else (0, 0)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "arc-agi-3"),
        id=wandb_run_id(),
        resume="allow",
        name="sia-supervisor",
    )
    run.log(
        {
            "run_id": run_id,
            "gen": gen,
            "rhae": score,
            "legacy_total_level_score": legacy_score,
            "baseline_before": baseline_before,
            "baseline_after": max(score, baseline_before),
            "delta": score - baseline_before,
            "passed": passed,
            "changed": int(changed),
            "lines_added": added,
            "lines_removed": removed,
            "promoted": int(promoted),
            "verdict": verdict,
            "game_state": wandb.Table(
                columns=GAME_STATE_COLUMNS,
                data=game_state_rows(results, submission),
            ),
        }
    )
    run.finish()


@app.command()
def main(
    run_id: int = typer.Option(..., "--run-id", help="SIA run_id whose generation to grade."),
    gen: int = typer.Option(..., "--gen", help="Generation number within run_id to grade."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Grade and log to W&B, but never touch AGENTS.md or the reference."
    ),
) -> None:
    global DRY_RUN
    DRY_RUN = dry_run
    """Grade one SIA generation against the current baseline, update the
    AGENTS.md ledger, promote a genuine improvement into the reference, and
    log the outcome (score trajectory + per-game/per-level scorecard) to W&B."""
    gdir = gen_dir(run_id, gen)
    results = load_results(gdir)
    submission = load_submission(gdir)
    if "rhae" not in results:
        raise SystemExit(
            f"{gdir}/results.json has no 'rhae' field — it was graded by the old "
            "unweighted evaluate.py; re-run evaluate.py --gen-dir on it first"
        )
    score = results["rhae"]
    legacy_score = results["total_level_score"]
    passed = results["passed"]

    candidate_explorer = gdir / "arc_agi3_explorer.py"
    changed, diff_note = diff_summary(candidate_explorer)
    improvement_md = gdir / "improvement.md"
    hypothesis = (
        improvement_md.read_text().splitlines()[0].lstrip("# ").strip()
        if improvement_md.is_file()
        else "(no improvement.md found)"
    )

    baseline = current_baseline()
    print(f"=== supervisor: run_{run_id}/gen_{gen} ===")
    print(
        f"RHAE = {score:.4f}% (current baseline {baseline:.4f}%), "
        f"legacy total_level_score = {legacy_score}, {passed}/8 passed"
    )
    print(f"explorer.py: {diff_note}")

    promoted = False
    if score > baseline:
        print(f"\n*** GENUINE IMPROVEMENT: RHAE {score:.4f}% > {baseline:.4f}% ***")
        print(f"Promoting run_{run_id}/gen_{gen}'s explorer.py into the reference.")
        if not DRY_RUN:
            shutil.copy2(candidate_explorer, REFERENCE)
            set_baseline(score)
        append_ledger_row(
            run_id,
            gen,
            hypothesis,
            f"RHAE {baseline:.4f}% -> {score:.4f}%",
            "**Promoted to reference — build on this next**",
        )
        verdict = "promoted"
        promoted = True
    elif not changed:
        append_ledger_row(
            run_id,
            gen,
            hypothesis,
            "no explorer.py change (scaffold-only or reverted)",
            "No-op — fine to bundle with a real fix, wastes a generation alone",
        )
        print("No behavioral change attempted. Ledger updated; no promotion.")
        verdict = "no-op"
    else:
        verdict = "Regressed — reverted" if score < baseline else "Neutral — no score change despite a diff"
        append_ledger_row(run_id, gen, hypothesis, f"RHAE={score:.4f}%, {diff_note}", verdict)
        print(f"{verdict}. Ledger updated; reference NOT changed.")
        print(f"\nAGENTS.md timestamp: {datetime.now(timezone.utc).isoformat()}")

    log_to_wandb(
        run_id=run_id,
        gen=gen,
        score=score,
        legacy_score=legacy_score,
        baseline_before=baseline,
        passed=passed,
        changed=changed,
        diff_note=diff_note,
        verdict=verdict,
        promoted=promoted,
        results=results,
        submission=submission,
    )


if __name__ == "__main__":
    app()
