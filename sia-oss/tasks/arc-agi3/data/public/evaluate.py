#!/usr/bin/env python3
"""Grade a target-agent submission against the ARC-AGI-3 eval suite's baselines.

The headline metric is **RHAE** (Relative Human Action Efficiency), computed
exactly as the ARC-AGI-3 Technical Report §4.1 defines it, so local deltas are
in the same units as the Kaggle leaderboard:

    S_l = min(1.15, (h_l / a_l)^2)            per level, 0 if not completed
    E   = min( sum_{l<=k} l / W,  sum_l l*S_l / W )   W = sum_{l<=n} l, w_l = l
    T   = mean over environments of E          reported as a percent

where ``n`` is the environment's total level count, ``k`` the number of
sequential levels completed, and ``h_l`` the upper-median human baseline. Both
``n`` and every ``h_l`` come from the game's ``metadata.json``
(``baseline_actions`` has one entry per level).

The first term of ``E`` is a hard cap: an environment can never score more
than the weighted fraction of its levels completed, so clearing a later level
is worth more than perfecting an early one. ``total_level_score`` (the old
unweighted sum of per-level scores) is still emitted for ledger continuity but
is not the promotion metric.

Pass/fail gates (``sia_questions.jsonl``) are unchanged and remain a coarse
regression check independent of the score.

Usage:
    python evaluate.py --gen-dir path/to/generation/directory
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

GAME_RE = re.compile(r"\b(?=[a-z0-9]{4}\b)(?=[a-z0-9]*\d)[a-z0-9]{4}\b")
LEVEL_CAP = 1.15


def game_of(text: str) -> str:
    found = GAME_RE.findall(text.lower())
    return found[0]


def level_score(baseline: float, actions: float) -> float:
    return min(LEVEL_CAP, (baseline / actions) ** 2)


def levels_completed(levels: dict[int, dict[str, Any]]) -> int:
    """Levels are sequential: k is the longest run 1..k present in the scorecard."""
    k = 0
    while (k + 1) in levels:
        k += 1
    return k


def environment_score(
    levels: dict[int, dict[str, Any]], baselines: list[float]
) -> dict[str, float]:
    """Tech report eq. (2): linearly weighted level mean, capped by the weighted
    fraction of levels completed. Returns the components so the cap-vs-efficiency
    limit is visible per game."""
    n = len(baselines)
    total_weight = n * (n + 1) / 2
    k = levels_completed(levels)
    cap = sum(range(1, k + 1)) / total_weight
    weighted = sum(
        level * level_score(baselines[level - 1], levels[level]["our_actions_median"])
        for level in range(1, k + 1)
    ) / total_weight
    return {
        "env_score": min(cap, weighted),
        "cap": cap,
        "weighted_efficiency": weighted,
        "levels_completed": k,
        "total_levels": n,
    }


def rhae_percent(env_scores: list[float]) -> float:
    return 100.0 * sum(env_scores) / len(env_scores)


def check_levels(levels: dict[int, dict[str, Any]], spec: dict[str, Any]) -> str:
    """The reason this game's run failed its check, or "" if it passed."""
    repeats = max((row.get("repeats", 1) for row in levels.values()), default=1)
    for want in spec.get("require_levels", []):
        if (level := levels.get(want)) is None:
            return f"level {want} was never cleared"
        if level["cleared_in_repeats"] < repeats:
            return (
                f"level {want} cleared in only {level['cleared_in_repeats']} of "
                f"{repeats} repeats, so it is not reliably solved"
            )
    if (need := spec.get("min_levels_any")) and len(levels) < need:
        return f"cleared {len(levels)} level(s), needed at least {need}"
    if (cap := spec.get("max_ratio")) is not None:
        slow = [row for row in levels.values() if row["ratio_vs_baseline"] > cap]
        if slow:
            return "; ".join(
                f"level {row['level']} took {row['our_actions_median']} actions "
                f"against a {row['human_baseline_actions']}-action baseline "
                f"({row['ratio_vs_baseline']}x, cap {cap}x)"
                for row in sorted(slow, key=lambda r: -r["ratio_vs_baseline"])
            )
    return ""


def grade_game(
    scorecard: dict[str, Any], spec: dict[str, Any], baselines: list[float]
) -> dict[str, Any]:
    if error := scorecard.get("error"):
        return {
            "game": spec["game"],
            "passed": False,
            "reason": error,
            "level_score": 0.0,
            "env_score": 0.0,
            "cap": 0.0,
            "weighted_efficiency": 0.0,
            "levels_completed": 0,
            "total_levels": len(baselines),
        }
    levels = {row["level"]: row for row in scorecard.get("levels") or []}
    reason = check_levels(levels, spec["check"])
    return {
        "game": spec["game"],
        "passed": not reason,
        "reason": reason or "ok",
        "level_score": sum(row["level_score"] for row in levels.values()),
        **environment_score(levels, baselines),
    }


def load_specs(private_dir: Path) -> list[dict[str, Any]]:
    specs = []
    for line in (private_dir / "sia_questions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        specs.append({"game": game_of(row["input"]), "check": row["check"], "id": row["id"]})
    return specs


def repo_root(script_dir: Path) -> Path:
    if env := os.environ.get("ARC_TGAER_REPO"):
        return Path(env)
    for parent in script_dir.parents:
        if (parent / "environment_files").is_dir():
            return parent
    raise FileNotFoundError(
        "cannot locate environment_files/ — set ARC_TGAER_REPO to the repo root"
    )


def load_baselines(root: Path, game: str) -> list[float]:
    matches = sorted((root / "environment_files" / game).glob("*/metadata.json"))
    if not matches:
        raise FileNotFoundError(f"no metadata.json for {game} under {root}/environment_files")
    return json.loads(matches[-1].read_text())["baseline_actions"]


def find_submission_file(gen_dir: Path) -> Path | None:
    """``submission.json`` by name, else the newest JSON in ``results/``.

    Name-first matters: a target agent that writes several scorecards (run_9
    emitted ``baseline.json`` *and* ``candidate.json``) would otherwise be
    graded on whichever file happened to be written last, which silently
    graded the baseline instead of the candidate.
    """
    results_dir = gen_dir / "results"
    if not results_dir.is_dir():
        return None
    if (named := results_dir / "submission.json").is_file():
        return named
    matches = list(results_dir.glob("*.json"))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    private_dir = script_dir.parent / "private"
    root = repo_root(script_dir)
    specs = {spec["game"]: spec for spec in load_specs(private_dir)}

    submission_path = find_submission_file(args.gen_dir)
    if submission_path is None:
        raise FileNotFoundError(f"No submission file found in {args.gen_dir}/results/")
    submission = json.loads(submission_path.read_text())

    by_game = {c["game"]: c for c in submission.get("scorecards", [])}
    # Every game played is scored, not only the eight carrying a pass/fail spec.
    # RHAE is a mean over environments, so omitting played games silently
    # inflates it; and the local set is the full 25-game / 183-level public set,
    # which is what published RHAE figures are quoted against.
    played = sorted(set(specs) | set(by_game))
    graded = [
        grade_game(
            by_game.get(game, {"game": game, "error": "no scorecard submitted"}),
            specs.get(game, {"game": game, "check": {}}),
            load_baselines(root, game),
        )
        for game in played
    ]

    passed = sum(1 for g in graded if g["passed"])
    rhae = rhae_percent([g["env_score"] for g in graded])
    total_level_score = sum(g["level_score"] for g in graded)
    results = {
        "total_games": len(graded),
        "passed": passed,
        "rhae": round(rhae, 4),
        "total_level_score": round(total_level_score, 2),
        "details": graded,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = args.output or (args.gen_dir / "results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print(f"{passed}/{len(graded)} games pass | RHAE={rhae:.4f}% | legacy total_level_score={total_level_score:.2f}")
    for g in sorted(graded, key=lambda r: r["game"]):
        status = "PASS" if g["passed"] else "FAIL"
        limit = "cap" if g["cap"] <= g["weighted_efficiency"] else "eff"
        print(
            f"  {status}  {g['game']:6s} E={100 * g['env_score']:6.3f}%  "
            f"{g['levels_completed']}/{g['total_levels']} lvls  "
            f"limited-by={limit}  {g['reason']}"
        )


if __name__ == "__main__":
    main()
