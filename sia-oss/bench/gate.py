"""The no-regression gate: decide whether a variant may be promoted.

A headline RHAE delta is not enough evidence on this benchmark. The rollout is
deterministic and chaotic, so any perturbation reshuffles every downstream
decision, and a single suite score is close to one draw: three separate changes
in one session read 0.5089, 0.7296 and 0.5860 against a 0.4263 baseline and all
three evaporated once their parameters were swept. Two independent competitors
report the same and landed on the same discipline — sweep every game, let no
game lose a level, change one thing at a time.

So a variant passes only when RHAE improves *and* no game goes backwards. A
change that trades `sp80`'s only level for a lucky unlock elsewhere nets
positive on the headline and is still a bad change, because the trade will not
reproduce on the hidden set.

`--seed` deliberately plays no part here: this agent is fully deterministic, so
seeds 0-3 give byte-identical scorecards and a seed sweep measures nothing.
Per-game non-regression is the strongest signal the local suite can give.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

# Below this, a per-game env-score difference is not worth calling a change.
EPSILON = 1e-9


def by_game(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {d["game"]: d for d in results["details"]}


def verdict(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[bool, list[str], list[str]]:
    """``(passed, regressions, improvements)`` comparing candidate to baseline."""
    cand, base = by_game(candidate), by_game(baseline)
    regressions: list[str] = []
    improvements: list[str] = []
    for game in sorted(base):
        c, b = cand.get(game), base[game]
        if c is None:
            regressions.append(f"{game}: missing from the candidate scorecard")
            continue
        if c["levels_completed"] < b["levels_completed"]:
            regressions.append(
                f"{game}: {b['levels_completed']} -> {c['levels_completed']} levels"
            )
        elif c["env_score"] < b["env_score"] - EPSILON:
            regressions.append(
                f"{game}: E {100 * b['env_score']:.3f}% -> {100 * c['env_score']:.3f}%"
            )
        elif c["env_score"] > b["env_score"] + EPSILON:
            improvements.append(
                f"{game}: E {100 * b['env_score']:.3f}% -> {100 * c['env_score']:.3f}%"
                f" ({b['levels_completed']} -> {c['levels_completed']} levels)"
            )
    improved = candidate["rhae"] > baseline["rhae"]
    return (improved and not regressions), regressions, improvements


def report(candidate_dir: Path, baseline_dir: Path) -> bool:
    """Print the gate's decision; True if the candidate may be promoted."""
    candidate = json.loads((candidate_dir / "results.json").read_text())
    baseline = json.loads((baseline_dir / "results.json").read_text())
    passed, regressions, improvements = verdict(candidate, baseline)

    logger.info(
        "gate: {} vs {} | RHAE {:.4f}% -> {:.4f}% ({:+.4f}pp)",
        candidate_dir.name, baseline_dir.name,
        baseline["rhae"], candidate["rhae"], candidate["rhae"] - baseline["rhae"],
    )
    for line in improvements:
        logger.success("better   {}", line)
    for line in regressions:
        logger.warning("WORSE    {}", line)
    if passed:
        logger.success("PASS — RHAE up and no game regressed.")
    elif regressions:
        logger.error("FAIL — {} game(s) regressed. Do not promote.", len(regressions))
    else:
        logger.error("FAIL — no game regressed, but RHAE did not improve.")
    return passed
