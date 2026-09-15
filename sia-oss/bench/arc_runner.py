#!/usr/bin/env python3
"""In-process deterministic ARC-AGI-3 explorer runner.

Imported by ``target_agent.py`` inside a clean subprocess so that the local
working-directory copy of ``arc_agi3_explorer.py`` is what actually runs, not
the installed ``tgaer`` package's original. The trick mirrors
``reference_target_agent.py``: register the local file under the exact dotted
name the kaggle wrapper imports (``tgaer.agents.arc_agi3_explorer``) *before*
anything imports it by that name, so ``load_agent_class(None, "explorer")``
yields the ``ExplorerAgent`` that lazy-imports our copy.

Reproduces the known-good deterministic baseline (total_level_score 131.08):
``OperationMode.OFFLINE`` (no network) + absolute ``environments_dir`` +
``max_steps=600`` + seed 0, and reduces the arcade scorecard to per-level rows
with the score formula ``min((baseline/actions)**2 * 100, 115)``.

Usage:
    python arc_runner.py --working_dir DIR --explorer_path FILE \
        --dataset_dir DIR --games '["a","b"]' --max_steps 600 --seed 0 \
        --out results.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import typer
from loguru import logger

app = typer.Typer(add_completion=False)


def repo_root_for(working_dir: Path) -> Path:
    """The tgaer checkout: env override, else walk up from working_dir looking
    for the ``src/tgaer`` + ``environment_files`` layout, else env default."""
    env = os.environ.get("ARC_TGAER_REPO")
    if env and (Path(env) / "src" / "tgaer").is_dir():
        return Path(env).resolve()
    for p in [working_dir, *working_dir.parents]:
        if (p / "src" / "tgaer").is_dir() and (p / "environment_files").is_dir():
            return p
    return Path(__file__).resolve().parents[4]  # .../tgaer (repo)


def _load_local(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, filename)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_levels(run: Any) -> list[dict[str, Any]]:
    """Per-level (level, actions, baseline) rows for one arcade run; only levels
    the agent actually completed (a positive arcade score)."""
    la = run.level_actions or []
    lb = run.level_baseline_actions or []
    ls = run.level_scores or []
    out: list[dict[str, Any]] = []
    for i, act in enumerate(la):
        base = lb[i] if i < len(lb) else None
        sc = ls[i] if i < len(ls) else 0.0
        if sc <= 0 or not base or base <= 0:
            continue
        out.append({"level": i + 1, "actions": act, "baseline": int(base)})
    return out


def _level_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[row["level"]].append(row)
    out = []
    for level in sorted(by_level):
        group = by_level[level]
        baseline = group[0]["baseline"]
        actions = median(row["actions"] for row in group)
        out.append(
            {
                "level": level,
                "cleared_in_repeats": len(group),
                "repeats": 1,
                "human_baseline_actions": baseline,
                "our_actions_median": round(actions, 1),
                "ratio_vs_baseline": round(actions / baseline, 1),
                "level_score": round(min((baseline / actions) ** 2 * 100, 115.0), 2),
            }
        )
    return out


def key_frames(agent: Any, game: str, every: int | None) -> list[dict[str, Any]]:
    """The frames worth looking at, as ``{game, idx, tag, level, grid}`` rows.

    Start, end, and both sides of every level transition — the moments that
    explain a run. Whole trajectories are not kept: 8 games at 600 steps is 4800
    grids, which is noise to scroll through rather than evidence. ``every``
    additionally samples one frame in N for a single focus game, which is what
    makes a video of that game possible.
    """
    frames = list(getattr(agent, "frames", []) or [])
    if not frames:
        return []
    levels = [getattr(f, "levels_completed", 0) for f in frames]
    wanted: dict[int, str] = {0: "start", len(frames) - 1: "end"}
    for i in range(1, len(frames)):
        if levels[i] > levels[i - 1]:
            wanted.setdefault(i - 1, f"before-L{levels[i]}")
            wanted[i] = f"after-L{levels[i]}"
    if every:
        for i in range(0, len(frames), every):
            wanted.setdefault(i, "step")
    rows: list[dict[str, Any]] = []
    for i in sorted(wanted):
        grid = getattr(frames[i], "frame", None)
        if not grid:
            continue
        rows.append(
            {
                "game": game,
                "idx": i,
                "tag": wanted[i],
                "level": levels[i],
                "grid": np.asarray(grid[-1], dtype=np.int16),
            }
        )
    return rows


def save_frames(rows: list[dict[str, Any]], out: Path) -> None:
    """Grids to a compressed ``.npz`` plus a JSON index of their metadata.

    Raw grids, not images: rendering and W&B are `measure.py`'s business, and
    keeping them out of here leaves this module a pure play harness.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{str(i): r["grid"] for i, r in enumerate(rows)})
    out.with_suffix(".json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "grid"} for r in rows], indent=2
        )
    )


def run_suite(
    explorer_path: Path,
    games: list[str],
    repo_root: Path,
    max_steps: int,
    seed: int,
    frames_out: Path | None = None,
    focus: str | None = None,
    focus_every: int = 4,
) -> list[dict[str, Any]]:
    """Play every game and return per-game scorecard dicts. Isolated here so the
    only things that can import the local explorer are the harness internals."""
    os.chdir(repo_root)
    os.environ.setdefault("ARC_TGAER_REPO", str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    # The local working-dir explorer is the thing under test. The grid/semantics
    # helpers it imports are byte-identical to the installed copies, so leave
    # those to the package (avoiding any path/identity drift).
    _load_local("tgaer.agents.arc_agi3_explorer", explorer_path)

    from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
        OperationMode,
        arc_agi,
        load_agent_class,
        play,
        require_starter,
    )

    require_starter()
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(repo_root / "environment_files"),
    )
    agent_cls = load_agent_class(None, "explorer")

    scorecards: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for game in games:
        t0 = time.monotonic()
        try:
            row = play(agent_cls, game, arc, None, max_steps, seed=seed)
            agent = row.pop("_agent", None)
            if frames_out is not None and agent is not None:
                frame_rows += key_frames(
                    agent, game, focus_every if game == focus else None
                )
            if row.get("error"):
                scorecards.append({"game": game, "error": row["error"]})
                logger.error("[{}] {} in {:.1f}s", game, row["error"], time.monotonic() - t0)
                continue
            rows: list[dict[str, Any]] = []
            card = arc.get_scorecard()
            for env in getattr(card, "environments", None) or []:
                if not str(env.id).startswith(game + "-"):
                    continue
                for run in env.runs:
                    rows += _run_levels(run)
            scorecards.append(
                {"game": game, "levels": _level_summary(rows), "row": row}
            )
            tot = sum(lvl["level_score"] for lvl in scorecards[-1]["levels"])
            logger.info(
                "[{}] total={:.2f} state={} actions={} in {:.1f}s",
                game, tot, row["state"], row["actions"], time.monotonic() - t0,
            )
        except Exception as exc:  # noqa: BLE001 — one bad game must not sink the rest
            scorecards.append({"game": game, "error": f"{type(exc).__name__}: {exc}"})
            logger.exception("[{}] {}: {}", game, type(exc).__name__, exc)
    if frames_out is not None and frame_rows:
        save_frames(frame_rows, frames_out)
        logger.info("[frames] wrote {} grids to {}", len(frame_rows), frames_out)
    return scorecards


@app.command()
def main(
    working_dir: Path = typer.Option(..., "--working_dir"),
    explorer_path: Path = typer.Option(..., "--explorer_path"),
    dataset_dir: Path = typer.Option(..., "--dataset_dir"),  # noqa: ARG001 — kept for caller compatibility
    games: str = typer.Option(..., "--games", help="JSON list of game ids."),
    out: Path = typer.Option(..., "--out"),
    max_steps: int = typer.Option(600, "--max_steps"),
    seed: int = typer.Option(0, "--seed"),
    repo_root: Path | None = typer.Option(None, "--repo_root"),
    frames_out: Path | None = typer.Option(None, "--frames_out"),
    focus: str | None = typer.Option(None, "--focus"),
    focus_every: int = typer.Option(4, "--focus_every"),
) -> None:
    """Play a suite of games with one explorer and write their scorecards."""
    repo = repo_root if repo_root else repo_root_for(working_dir)
    scorecards = run_suite(
        explorer_path,
        json.loads(games),
        repo,
        max_steps,
        seed,
        frames_out=frames_out,
        focus=focus,
        focus_every=focus_every,
    )
    total = sum(lv["level_score"] for c in scorecards for lv in c.get("levels", []))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"total_level_score": round(total, 2), "scorecards": scorecards}, indent=2
        )
    )
    # stdout, not the logger: `measure.py` scrapes this line.
    print(f"RUNNER_TOTAL={total:.2f}", flush=True)


if __name__ == "__main__":
    app()
