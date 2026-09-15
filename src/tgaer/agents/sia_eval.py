"""Score the ARC-AGI-3 explorer over `sia_questions.jsonl`. Run by `sia evals run`.

Invokes `sia_adapter.py` exactly as a human would — a JSON object piped to its
stdin, one JSON object read back from its stdout — and grades the
`--- ARC-AGI-3 SCORECARD ---` block it appends to its answer.

The checkers here are arithmetic, not a judge. ARC-AGI-3 pays
`min((baseline / actions)^2 * 100, 115)` per level, so "did this run get
better" has an exact answer, and handing that question to an LLM would add
variance to the one number in this project that does not need any more: the
public leaderboard score sat flat at 0.13 whether the agent cleared 2 levels
or 5, which is why scoring moved local in the first place.

Each question's `check` object says what passing means:

    require_levels    these levels must clear in *every* repeat
    max_ratio         no cleared level may exceed this multiple of its baseline
    min_levels_any    at least this many levels cleared in at least one repeat

`faults` is checked for every case regardless. It counts paths the agent is
never supposed to reach, and a run that took them is a wiring bug reporting
itself as a score.

Usage:
    python3 sia_eval.py                                     # all questions
    python3 sia_eval.py --questions FILE --out FILE
    python3 sia_eval.py --only ratio-tu93,control-lp85      # one case at a time
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ADAPTER = HERE / "sia_adapter.py"
SCORECARD_MARKER = "--- ARC-AGI-3 SCORECARD ---"
# Generous: one case is `ARC_SIA_REPEATS` full games, and the slowest game on
# the dev set takes ~160s per repeat.
TIMEOUT_S = int(os.environ.get("ARC_SIA_TIMEOUT_S", "2400"))


def run_adapter(case_input: str) -> dict[str, Any]:
    """Invoke the adapter the normal way: JSON on stdin, JSON on stdout."""
    proc = subprocess.run(
        [sys.executable, str(ADAPTER)],
        input=json.dumps({"input": case_input}),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=os.environ.copy(),
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"adapter printed no JSON: stdout={proc.stdout[-500:]!r} "
            f"stderr={proc.stderr[-500:]!r}"
        ) from exc


def parse_scorecard(output: str) -> dict[str, Any]:
    """Pull the trailing scorecard JSON block out of the adapter's answer."""
    if SCORECARD_MARKER not in (output or ""):
        return {}
    _, _, blob = output.partition(SCORECARD_MARKER)
    return json.loads(blob.strip())


def check_levels(
    levels: dict[int, dict[str, Any]], spec: dict[str, Any], repeats: int
) -> str:
    """The reason this run failed its check, or "" if it passed."""
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
                f"({row['ratio_vs_baseline']}x, cap {cap}x), scoring "
                f"{row['level_score']} of 100"
                for row in sorted(slow, key=lambda r: -r["ratio_vs_baseline"])
            )
    return ""


def check(card: dict[str, Any], spec: dict[str, Any]) -> tuple[bool, str]:
    if error := card.get("error"):
        return False, f"the run failed and scored nothing: {error}"
    if not card:
        return False, "the adapter printed no scorecard block"
    if broken := card.get("faults"):
        return False, f"fault counters above zero: {', '.join(broken)}"
    levels = {row["level"]: row for row in card.get("levels") or []}
    if reason := check_levels(levels, spec, card.get("repeats", 1)):
        return False, reason
    scored = ", ".join(
        f"L{row['level']} {row['ratio_vs_baseline']}x -> {row['level_score']}"
        for row in sorted(levels.values(), key=lambda r: r["level"])
    )
    return True, f"cleared {len(levels)} level(s): {scored or 'none required'}"


def score_case(case: dict[str, Any]) -> dict[str, Any]:
    row = {
        "id": case["id"],
        "input": case["input"],
        "category": case.get("category", ""),
        "tags": case.get("tags", []),
    }
    try:
        answer = run_adapter(case["input"])
        card = parse_scorecard(answer.get("output", ""))
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        return {**row, "correct": False, "reason": f"{type(exc).__name__}: {exc}"}
    correct, reason = check(card, case.get("check") or {})
    # The scorecard rides along so a failing row carries its own evidence: SIA
    # diagnoses from these rows, and "level 1 at 23x baseline" is actionable in
    # a way that a bare False is not.
    return {**row, "correct": correct, "reason": reason, "scorecard": card}


def load_questions(path: Path, only: set[str]) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not only:
        return cases
    if unknown := only - {case["id"] for case in cases}:
        raise SystemExit(f"unknown case id(s): {sorted(unknown)}")
    return [case for case in cases if case["id"] in only]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=HERE / "sia_questions.jsonl")
    parser.add_argument("--out", type=Path, default=Path("eval_results.jsonl"))
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    only = {name.strip() for name in args.only.split(",") if name.strip()}
    cases = load_questions(args.questions, only)

    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(cases))) as pool:
        rows = list(pool.map(score_case, cases))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    passed = sum(1 for row in rows if row["correct"])
    for row in rows:
        mark = "PASS" if row["correct"] else "FAIL"
        print(f"{mark}  {row['id']:16s} {row['reason']}", file=sys.stderr)
    # The pass/fail count above is a coarse gate — two patches have passed it
    # unchanged while this, the actual scored objective, moved (one of them
    # -1.99, invisible without summing scorecard.levels[].level_score by hand;
    # see FIX_NOTES.md's p2). Printed unconditionally so a before/after
    # comparison never again depends on remembering to compute it.
    total_score = sum(
        level["level_score"]
        for row in rows
        for level in (row.get("scorecard") or {}).get("levels") or []
    )
    print(f"\n{passed}/{len(rows)} cases pass -> {args.out}", file=sys.stderr)
    print(f"total level_score across the suite: {total_score:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
