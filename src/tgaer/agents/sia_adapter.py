"""SIA command adapter — play one ARC-AGI-3 game and report what it scored.

A JSON object on stdin, a JSON object on stdout:

    echo '{"input": "Play lp85."}' | .venv/bin/python src/tgaer/agents/sia_adapter.py

`sia_eval.py` invokes this once per eval case, and both the programmatic
checker and SIA's own diagnosis read the `--- ARC-AGI-3 SCORECARD ---` block
below — so a run whose real result is a number has to say what that number was.

What it reports is deliberately *not* levels cleared. A level scores
`min((baseline / actions_on_that_level)^2 * 100, 115)`, so the 12 levels the
explorer cleared on the 25-game roster were worth 0.174 against the 2.349 the
same levels score at human speed — 7.4%. `ratio_vs_baseline` is the term that
loses, so it is the term the scorecard leads with.

Every case is played `ARC_SIA_REPEATS` times under different seeds and reported
as a median. A single cleared level has failed to reproduce across three
targeted repeats before, and a one-shot number would hand SIA that noise as a
gradient to chase.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

# The ARC SDK writes its logs to stdout rather than stderr, and it starts doing
# so while the imports below are still running. One INFO line in front of the
# JSON is enough for the harness to read every case as unparseable output
# instead of as a result, so the real stdout is claimed here — before anything
# that logs is imported — and handed back only to answer.
ANSWER = sys.stdout
sys.stdout = sys.stderr

# This file's own directory, which is the SIA project root: the checked-out
# `src/tgaer/agents/` during development, and a `.sia/versions/vN/` copy of it
# once SIA has applied a patch. `tgaer` itself is editable-installed in the
# shared venv (`uv pip install -e .`), so unversioned infra — score_local's
# helpers below — resolves from the real repo regardless of which directory
# this file was copied into; no REPO-relative sys.path math needed to reach it.
HERE = Path(__file__).resolve().parent

# Loaded in dependency order and registered under their package names, so the
# explorer's own `from tgaer.agents.arc_agi3_grid import ...` resolves to the
# copy beside it rather than the installed one. This tuple and `[source]
# include` in .sia/config.toml are the same set on purpose: a file SIA may
# patch but this does not load would be measured unpatched, which reads
# exactly like a fix that did not help.
VERSIONED = (
    "arc_agi3_grid",
    "arc_agi3_semantics",
    "arc_agi3_explorer",
    "arc_agi3_kaggle",
)


def load_versioned() -> None:
    for stem in VERSIONED:
        name = f"tgaer.agents.{stem}"
        spec = importlib.util.spec_from_file_location(name, HERE / f"{stem}.py")
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load {HERE / f'{stem}.py'}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


load_versioned()

from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    faults,
    game_key,
    inert_features,
    load_agent_class,
    play,
    require_starter,
    run_levels,
)

# The model-free explorer, not the 27B LLM agent: it is the one that clears
# levels (4 games and 5 levels on the roster against 0), so it is the one whose
# action count is worth improving.
AGENT = os.environ.get("ARC_SIA_AGENT", "explorer")
REPEATS = int(os.environ.get("ARC_SIA_REPEATS", "3"))
MAX_ACTIONS = int(os.environ.get("ARC_SIA_MAX_ACTIONS", "600"))
SEED = int(os.environ.get("ARC_SIA_SEED", "0"))

# Game ids are four lowercase alphanumerics with at least one digit, which is
# what keeps them out of the surrounding prose: "Play lp85 and report" has one
# candidate, not three.
GAME_RE = re.compile(r"\b(?=[a-z0-9]{4}\b)(?=[a-z0-9]*\d)[a-z0-9]{4}\b")


def parse_game(text: str) -> str:
    """The game id an eval case is asking for.

    Raises rather than defaulting: a case whose id we failed to read would
    otherwise be scored against some other game entirely, and neither the
    checker nor the judge has any way to tell that apart from a bad run.
    """
    found = GAME_RE.findall(text.lower())
    if not found:
        raise ValueError(f"no ARC-AGI-3 game id found in {text!r}")
    if len(set(found)) > 1:
        raise ValueError(f"ambiguous game ids {sorted(set(found))} in {text!r}")
    return found[0]


def level_summary(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    """Collapse every repeat's per-level rows into one median row per level.

    `cleared_in_repeats` is kept because it is the difference between a level
    the agent solves and one it stumbled into: 1 of 3 and 3 of 3 produce the
    same median and mean very different things.
    """
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
                "repeats": repeats,
                "human_baseline_actions": baseline,
                "our_actions_median": round(actions, 1),
                "ratio_vs_baseline": round(actions / baseline, 1),
                "level_score": round(min((baseline / actions) ** 2 * 100, 115.0), 2),
                "level_score_at_human_speed": 100.0,
            }
        )
    return out


def verdict(levels: list[dict[str, Any]], game: str) -> str:
    """One plain sentence naming the term that lost, for the judge to grade."""
    if not levels:
        return f"No level of {game} was cleared, so it scored nothing."
    worst = max(levels, key=lambda row: row["ratio_vs_baseline"])
    return (
        f"Cleared {len(levels)} level(s) of {game}. The slowest was level "
        f"{worst['level']} at {worst['ratio_vs_baseline']}x the human baseline "
        f"({worst['our_actions_median']} actions against "
        f"{worst['human_baseline_actions']}), scoring {worst['level_score']} of "
        f"the 100 it is worth at human speed."
    )


def run_game(game: str) -> dict[str, Any]:
    """Play one game `REPEATS` times and reduce it to the reported scorecard."""
    require_starter()  # the games and the SDK live in the starter checkout
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    agent_cls = load_agent_class(None, AGENT, path=HERE / "arc_agi3_kaggle.py")
    agent_module = sys.modules[agent_cls.__module__]

    repeats: list[dict[str, Any]] = []
    decisions: dict[str, int] = defaultdict(int)
    for offset in range(REPEATS):
        row = play(agent_cls, game, arc, None, MAX_ACTIONS, seed=SEED + offset)
        row.pop("_agent", None)
        for name, count in (row.get("decisions") or {}).items():
            decisions[name] += count
        repeats.append(row)

    if error := next((row["error"] for row in repeats if row.get("error")), None):
        return {"game": game, "error": error}

    card = arc.get_scorecard()
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for env in getattr(card, "environments", None) or []:
        if game_key(env.id) != game:
            continue
        for run in env.runs:
            rows += run_levels(game, run, run.score)
            scores.append(run.score)

    actions = sum(row.get("actions") or 0 for row in repeats)
    return {
        "game": game,
        "agent": AGENT,
        "repeats": REPEATS,
        "action_budget_per_repeat": MAX_ACTIONS,
        "levels": level_summary(rows, REPEATS),
        "game_score_median": round(median(scores), 4) if scores else 0.0,
        # Both diagnostics separate a run that played badly from one that never
        # ran the code it was meant to: five features have shipped inert, each
        # reporting the same tidy zero as a feature that ran and did not help.
        "faults": faults(decisions, actions),
        "inert_features": inert_features(agent_module, decisions, agent_cls),
        "decisions_total": dict(sorted(decisions.items())),
    }


SCORECARD_MARKER = "--- ARC-AGI-3 SCORECARD ---"


def render(card: dict[str, Any]) -> str:
    if error := card.get("error"):
        return f"The run failed and scored nothing: {error}"
    body = json.dumps(card, indent=2, sort_keys=True)
    return f"{verdict(card['levels'], card['game'])}\n\n{SCORECARD_MARKER}\n{body}"


def answer(payload: dict[str, Any], code: int) -> int:
    print(json.dumps(payload), file=ANSWER)
    return code


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return answer({"error": "stdin was not JSON"}, 1)
    try:
        game = parse_game(str(payload.get("input") or ""))
    except ValueError as exc:
        return answer({"error": str(exc)}, 1)
    try:
        card = run_game(game)
    except Exception as exc:  # noqa: BLE001 — the adapter must always answer
        card = {"game": game, "error": f"{type(exc).__name__}: {exc}"}
    # No `tokens`: the explorer calls no model, so the cost axis SIA would plot
    # is actions, and those are in the scorecard where the checker can see them.
    return answer({"output": render(card)}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
