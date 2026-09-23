#!/usr/bin/env python3
"""Does click effect stay predictable from frame features mid-episode?

At opening frames, 68/69 well-sampled `(colour, component-size)` buckets agree
on whether clicking does anything. That licenses a shippable prior only if it
survives into the states the agent actually reaches, so this probes on-policy
frames sampled along the explorer's own trajectory, across the full roster.

Background and singleton buckets are excluded: both are pure for free.
"""

from __future__ import annotations

import copy
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

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

from tgaer.agents.arc_agi3_grid import components  # noqa: E402
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    load_agent_class,
    play,
    require_starter,
)

os.chdir(REPO)
os.environ.setdefault("ARC_TGAER_REPO", str(REPO))

app = typer.Typer(add_completion=False)

SUITE = sorted(
    p.name
    for p in (REPO / "environment_files").iterdir()
    if p.is_dir() and len(p.name) == 4
)


class Sample(NamedTuple):
    step: int
    pure: int
    scored: int
    effective: int


def cell_owner(base: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    """Map each cell to its (colour, component-size) bucket."""
    owner: dict[tuple[int, int], tuple[int, int]] = {}
    for v in np.unique(base).tolist():
        for comp in components(base, (v,)):
            bucket = (v, len(comp))
            for r, c in comp.tolist():
                owner[(r, c)] = bucket
    return owner


def fork(game: Any) -> Any:
    """Deepcopy the game, sharing `_clean_levels` by reference.

    Probing never sends RESET, the only path that reads it, and it is cloned
    rather than mutated even there — so sharing it is equivalent and halves the
    copy on the games whose cost is level state.
    """
    clean = getattr(game, "_clean_levels", None)
    memo = {id(clean): clean} if clean is not None else {}
    return copy.deepcopy(game, memo)


def probe(game: Any, base: np.ndarray, stride: int) -> tuple[int, int, int]:
    """Fork-probe a lattice; return (pure buckets, scored buckets, effective cells)."""
    owner = cell_owner(base)
    bg = M.background(base)
    by_bucket: dict[tuple[int, int], list[int]] = defaultdict(list)
    effective = 0
    n = base.shape[0]
    for r in range(0, n, stride):
        for c in range(0, n, stride):
            after = M.grid(M.act(fork(game), (r, c)))
            works = int(not np.array_equal(after, base))
            effective += works
            bucket = owner.get((r, c))
            if bucket is not None:
                by_bucket[bucket].append(works)
    # Exclude the background by colour, not by "largest component": a fragmented
    # floor has no single biggest piece, and a foreground object can tie one.
    scored = [v for (colour, _), v in by_bucket.items() if len(v) >= 2 and colour != bg]
    pure = sum(1 for v in scored if len(set(v)) == 1)
    return pure, len(scored), effective


def run_game(
    game_id: str, at: set[int], max_steps: int, stride: int, seed: int
) -> tuple[list[Sample], int]:
    """Play the explorer, fork-probing at the sampled steps."""
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    samples: list[Sample] = []
    dropped = 0

    def hook(step: int, observation: Any, env: Any, actor: Any) -> None:
        nonlocal dropped
        if step not in at or env is None or not observation:
            return
        frame = observation.get("frame") or []
        if not frame:
            return
        base = np.asarray(frame[-1], dtype=np.int16)
        try:
            # Only the probe runs the game's own step code, which raises on its
            # own account under exhaustive clicking (lf52's `list.remove`).
            samples.append(Sample(step, *probe(env._game, base, stride)))
        except Exception as exc:
            dropped += 1
            logger.warning("{} step {}: probe raised {}", game_id, step, exc)

    row = play(
        load_agent_class(None, "explorer"),
        game_id,
        arc,
        None,
        max_steps,
        seed=seed,
        on_step=hook,
    )
    if row.get("error"):
        raise RuntimeError(row["error"])
    return samples, dropped


@app.command()
def main(
    games: str = typer.Option(",".join(SUITE), help="Comma-separated game ids."),
    at: str = typer.Option("0,25,50,100,200,300", help="Steps to probe at."),
    max_steps: int = typer.Option(320),
    stride: int = typer.Option(3),
    seed: int = typer.Option(0),
) -> None:
    """Report per-game bucket purity across on-policy frames."""
    require_starter()
    steps = {int(s) for s in at.split(",")}
    tot_pure = tot_buckets = 0
    inert: list[str] = []
    for game_id in games.split(","):
        try:
            samples, dropped = run_game(game_id, steps, max_steps, stride, seed)
        except Exception as exc:
            logger.error("{}: {}: {}", game_id, type(exc).__name__, exc)
            continue
        if not samples:
            logger.warning("{}: no frames probed", game_id)
            continue
        pure = sum(s.pure for s in samples)
        n = sum(s.scored for s in samples)
        eff = sum(s.effective for s in samples)
        tot_pure += pure
        tot_buckets += n
        if eff == 0:
            inert.append(game_id)
        detail = " ".join(f"{s.step}:{s.pure}/{s.scored}" for s in samples)
        logger.info(
            "{:6} {:3}/{:3} pure ({:3.0f}%) over {}/{} frames, {:4} effective | {}",
            game_id,
            pure,
            n,
            100 * pure / max(n, 1),
            len(samples),
            len(samples) + dropped,
            eff,
            detail,
        )
    logger.success(
        "{}/{} buckets pure ({:.0f}%) across on-policy frames",
        tot_pure,
        tot_buckets,
        100 * tot_pure / max(tot_buckets, 1),
    )
    if inert:
        logger.warning("click-inert at every probed frame: {}", ", ".join(inert))


if __name__ == "__main__":
    app()
