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

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _repo_root(working_dir: Path) -> Path:
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


def run_suite(
    explorer_path: Path,
    games: list[str],
    repo_root: Path,
    max_steps: int,
    seed: int,
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
    for game in games:
        t0 = time.monotonic()
        try:
            row = play(agent_cls, game, arc, None, max_steps, seed=seed)
            row.pop("_agent", None)
            if row.get("error"):
                scorecards.append({"game": game, "error": row["error"]})
                print(f"[{game}] ERROR {row['error']} in {time.monotonic() - t0:.1f}s", flush=True)
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
            print(
                f"[{game}] total={tot:.2f} state={row['state']} "
                f"actions={row['actions']} in {time.monotonic() - t0:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — one bad game must not sink the rest
            scorecards.append({"game": game, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{game}] EXC {type(exc).__name__}: {exc}", flush=True)
    return scorecards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--working_dir", required=True, type=Path)
    ap.add_argument("--explorer_path", required=True, type=Path)
    ap.add_argument("--dataset_dir", required=True, type=Path)
    ap.add_argument("--games", required=True)  # JSON list
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo_root", type=Path, default=None)
    args = ap.parse_args()

    repo = args.repo_root if args.repo_root else _repo_root(args.working_dir)
    scorecards = run_suite(
        args.explorer_path, json.loads(args.games), repo, args.max_steps, args.seed
    )
    total = sum(lv["level_score"] for c in scorecards for lv in c.get("levels", []))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"total_level_score": round(total, 2), "scorecards": scorecards}, indent=2
        )
    )
    print(f"RUNNER_TOTAL={total:.2f}", flush=True)


if __name__ == "__main__":
    main()
