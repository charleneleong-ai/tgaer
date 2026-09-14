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

import numpy as np
import typer
from dotenv import load_dotenv
from PIL import Image

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


# ARC grids are small integer palettes, so a fixed colour table keeps a cell's
# colour stable across games and runs — a per-image autoscale would recolour the
# same board between frames and make a diff impossible to read by eye.
ARC_PALETTE = np.array(
    [
        (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
        (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
        (127, 219, 255), (135, 12, 37), (255, 255, 255), (96, 96, 96),
        (0, 255, 200), (140, 90, 200), (60, 60, 60), (200, 200, 120),
    ],
    dtype=np.uint8,
)


def render(grid: np.ndarray, scale: int = 6) -> np.ndarray:
    """One ARC grid as an upscaled RGB image.

    Nearest-neighbour upscaling, not interpolation: these are discrete cell
    values, and a smoothed edge invents colours that no cell holds.
    """
    rgb = ARC_PALETTE[np.clip(grid, 0, len(ARC_PALETTE) - 1)]
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def load_frames(npz_path: Path) -> list[dict[str, Any]]:
    """The runner's saved grids re-joined to their JSON metadata index."""
    index_path = npz_path.with_suffix(".json")
    if not (npz_path.is_file() and index_path.is_file()):
        return []
    with np.load(npz_path) as bundle:
        meta = json.loads(index_path.read_text())
        return [{**row, "grid": bundle[str(i)]} for i, row in enumerate(meta)]


def frame_payload(
    frames: list[dict[str, Any]], focus: str | None, gif_dir: Path
) -> dict[str, Any]:
    """Key frames as per-game `wandb.Image` lists, plus a video for `focus`.

    Keyed per game rather than as one flat list so a game's start/end sit beside
    each other in the UI instead of being interleaved with seven other boards.
    """
    import wandb

    payload: dict[str, Any] = {}
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in frames:
        by_game.setdefault(row["game"], []).append(row)
    for game, rows in by_game.items():
        keyed = [r for r in rows if r["tag"] != "step"]
        payload[f"frames/{game}"] = [
            wandb.Image(
                render(r["grid"]), caption=f"{r['tag']} @ step {r['idx']} (L{r['level']})"
            )
            for r in keyed
        ]
        if game == focus and (steps := [r for r in rows if r["tag"] == "step"]):
            # Written with PIL and handed to wandb as a path: passing raw arrays
            # to wandb.Video needs moviepy, a dependency this repo does not have
            # and does not need for an 8fps GIF of a 64x64 board.
            gif = gif_dir / f"{game}.gif"
            images = [Image.fromarray(render(r["grid"], scale=3)) for r in steps]
            images[0].save(
                gif,
                save_all=True,
                append_images=images[1:],
                duration=125,
                loop=0,
            )
            payload[f"video/{game}"] = wandb.Video(str(gif), format="gif")
    return payload


def log_to_wandb(
    *,
    label: str,
    seed: int,
    max_steps: int,
    explorer: Path,
    results: dict[str, Any],
    submission: dict[str, Any],
    frames: list[dict[str, Any]],
    focus: str | None,
    run_dir: Path,
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
            **frame_payload(frames, focus, run_dir),
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
    frames: bool = typer.Option(
        False,
        "--frames/--no-frames",
        help="Capture and log board images (start, end, level transitions).",
    ),
    focus: str | None = typer.Option(
        None, "--focus", help="Also log a video of this one game's run."
    ),
) -> None:
    """Play the suite with one explorer variant and print its RHAE."""
    run_dir = BENCH / "runs" / label
    out = run_dir / "results" / "submission.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_npz = run_dir / "frames.npz"

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
            *(["--frames_out", str(frames_npz)] if frames or focus else []),
            *(["--focus", focus] if focus else []),
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
            frames=load_frames(frames_npz) if frames or focus else [],
            focus=focus,
            run_dir=run_dir,
        )


if __name__ == "__main__":
    app()
