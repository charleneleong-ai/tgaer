#!/usr/bin/env python3
"""How much does the pixel-keyed state signature fragment, and what merges it?

`frame_signature` keys on every in-field pixel. Its own TODO records the cost —
"live ls20: 741 signatures for 30 avatar cells, blinding the StateGraph frontier
to revisits" — and the value-model work hit the same wall from the other side:
lp85 produced 728 states in one level with only 28 ever offering a second action
to rank.

A coarser key is only useful if it stays *deterministic*: merging two boards that
respond differently to the same action breaks the assumption the whole state
graph rests on. So this reports both, for each candidate key:

  states     distinct keys seen
  rankable   keys where two or more distinct actions were tried — what a value
             model can learn from, and what the frontier needs to see a revisit
  ambiguous  (key, action) pairs that led to more than one successor key
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
)
import m1_codegen as M  # noqa: E402
import value_model as V  # noqa: E402

from tgaer.evaluation.arc_agi3_score_local import require_starter  # noqa: E402

os.chdir(REPO)

app = typer.Typer(add_completion=False)


def pixel_key(step: dict[str, Any]) -> Any:
    """Exactly what the agent uses: the settled board inside its field box."""
    return step["sig"]


def object_key(step: dict[str, Any]) -> Any:
    """The settled board as a set of non-background components."""
    arr = step["settled"]
    return tuple(sorted(M.objects(arr, M.background(arr))))


def shape_key(step: dict[str, Any]) -> Any:
    """Objects without exact extent: colour, size and centroid only.

    Closer to treating a contiguous same-colour region as one button, which is
    what the 6.71% Preview agent keys on.
    """
    arr = step["settled"]
    bg = M.background(arr)
    out = []
    for value in np.unique(arr).tolist():
        if value == bg:
            continue
        from tgaer.agents.arc_agi3_grid import components

        for comp in components(arr, (value,)):
            out.append(
                (value, len(comp), int(comp[:, 0].mean()), int(comp[:, 1].mean()))
            )
    return tuple(sorted(out))


KEYS = {"pixel (shipped)": pixel_key, "object": object_key, "shape+centroid": shape_key}


def score_key(steps: list[dict[str, Any]], fn: Any) -> tuple[int, int, int, int]:
    """(states, rankable states, ambiguous transitions, transitions)."""
    keys = [fn(s) for s in steps]
    tried: dict[Any, set[Any]] = {}
    succ: dict[tuple[Any, Any], set[Any]] = {}
    for i in range(len(steps) - 1):
        prim = steps[i]["prim"]
        if not prim:
            continue
        prim = tuple(prim)
        tried.setdefault(keys[i], set()).add(prim)
        succ.setdefault((keys[i], prim), set()).add(keys[i + 1])
    rankable = sum(1 for v in tried.values() if len(v) >= 2)
    ambiguous = sum(1 for v in succ.values() if len(v) > 1)
    return len(set(keys)), rankable, ambiguous, len(succ)


@app.command()
def main(
    games: str = typer.Option("lp85,tu93,ls20,sp80", help="Comma-separated ids."),
    max_steps: int = typer.Option(1200),
    seed: int = typer.Option(0),
) -> None:
    """Compare state keys on how much they merge and whether they stay Markov."""
    require_starter()
    for game_id in games.split(","):
        try:
            steps = V.rollout(game_id, max_steps, seed)
        except Exception as exc:
            logger.error("{}: {}: {}", game_id, type(exc).__name__, exc)
            continue
        logger.info("{} — {} steps", game_id, len(steps))
        for name, fn in KEYS.items():
            try:
                n, rank, amb, trans = score_key(steps, fn)
            except Exception as exc:
                logger.error("  {:16} raised {}", name, type(exc).__name__)
                continue
            logger.info(
                "  {:16} states {:5}  rankable {:4}  ambiguous {:4}/{:5} ({:.0%})",
                name,
                n,
                rank,
                amb,
                trans,
                amb / max(trans, 1),
            )


if __name__ == "__main__":
    app()
