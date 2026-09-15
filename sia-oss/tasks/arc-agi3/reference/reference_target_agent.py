#!/usr/bin/env python3
"""ARC-AGI-3 explorer target agent — plays the eval suite's games and writes a
per-level scorecard to ``results/submission.json``.

This is the seed sia-agent (OSS) refactors every generation. ``arc_agi3_grid.py``,
``arc_agi3_semantics.py``, and ``arc_agi3_explorer.py`` sit alongside this file and
are copied fresh into each generation's working directory — those three are the
improvable surface; this entrypoint and the harness plumbing it calls into are not.

Those three files still say ``from tgaer.agents.arc_agi3_grid import (...)``
internally (unchanged from the real repo, so a diff against production stays
readable). ``_load_local`` below registers the LOCAL, possibly-edited copies under
those exact dotted names in ``sys.modules`` before anything imports them that way —
including ``arc_agi3_kaggle.py``'s own ``from tgaer.agents.arc_agi3_explorer import
ExplorerArcAgi3Agent`` — so the edited copies are what actually run, not the
installed ``tgaer`` package's originals. ``arc_agi3_kaggle.py`` itself is loaded
from the real repo (not copied here): it is a thin per-game wrapper around the
explorer, not part of what this task is improving.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

HERE = Path(__file__).resolve().parent

# Set by the harness that invokes this script (see profiles/arc-agi3-target.json's
# sibling requirements.txt, which editable-installs tgaer from this absolute path).
REPO = Path(os.environ["ARC_TGAER_REPO"])
KAGGLE_PATH = REPO / "src/tgaer/agents/arc_agi3_kaggle.py"


def _load_local(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {HERE / filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_local("tgaer.agents.arc_agi3_grid", "arc_agi3_grid.py")
_load_local("tgaer.agents.arc_agi3_semantics", "arc_agi3_semantics.py")
_load_local("tgaer.agents.arc_agi3_explorer", "arc_agi3_explorer.py")

from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    game_key,
    load_agent_class,
    play,
    require_starter,
    run_levels,
)

REPEATS = int(os.environ.get("ARC_SIA_REPEATS", "1"))
MAX_ACTIONS = int(os.environ.get("ARC_SIA_MAX_ACTIONS", "600"))
SEED = int(os.environ.get("ARC_SIA_SEED", "0"))


def level_summary(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    """Collapse every repeat's per-level rows into one median row per level."""
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
            }
        )
    return out


def run_game(game: str) -> dict[str, Any]:
    """Play one game ``REPEATS`` times and reduce it to a scorecard."""
    require_starter()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    agent_cls = load_agent_class(None, "explorer", path=KAGGLE_PATH)

    repeats: list[dict[str, Any]] = []
    for offset in range(REPEATS):
        row = play(agent_cls, game, arc, None, MAX_ACTIONS, seed=SEED + offset)
        row.pop("_agent", None)
        repeats.append(row)

    if error := next((row["error"] for row in repeats if row.get("error")), None):
        return {"game": game, "error": error}

    card = arc.get_scorecard()
    rows: list[dict[str, Any]] = []
    for env in getattr(card, "environments", None) or []:
        if game_key(env.id) != game:
            continue
        for run in env.runs:
            rows += run_levels(game, run, run.score)

    return {"game": game, "levels": level_summary(rows, REPEATS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--working_dir", type=Path, required=True)
    args = parser.parse_args()

    games = json.loads((args.dataset_dir / "games.json").read_text())
    scorecards = []
    for game in games:
        try:
            scorecards.append(run_game(game))
        except Exception as exc:  # noqa: BLE001 — one bad game must not sink the rest
            scorecards.append({"game": game, "error": f"{type(exc).__name__}: {exc}"})

    output_dir = args.working_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "submission.json"
    output_file.write_text(json.dumps({"scorecards": scorecards}, indent=2))
    cleared = sum(1 for c in scorecards if c.get("levels"))
    print(
        f"{cleared}/{len(games)} games cleared at least one level | saved {output_file}"
    )


if __name__ == "__main__":
    main()
