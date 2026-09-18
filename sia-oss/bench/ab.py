#!/usr/bin/env python3
"""A/B two explorer variants across seeds, and judge the delta against noise.

`gate.py` compares one rollout of each variant. That was sound while the agent
was deterministic and there was nothing else on offer, but it cannot tell a real
gain from a reshuffle: the measured seed-to-seed spread of the unchanged agent is
sd ~= 0.030pp, and four of six changes gated in one session landed inside it.

So: run both arms over N seeds, compare means against the pooled spread, and
report per-game *frequency* rather than a single game's level count. A game that
scores in 4/5 seeds for one arm and 1/5 for the other is a finding; a game that
flips once is not.
"""

from __future__ import annotations

import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "sia-oss/bench"
RHAE = re.compile(r"RHAE=([0-9.]+)%")

app = typer.Typer(add_completion=False)


def one_run(explorer: Path, label: str, seed: int, max_steps: int) -> dict:
    """One 25-game measurement; returns {'rhae': float, 'levels': {game: n}}."""
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH / "measure.py"),
            "--label",
            label,
            "--seed",
            str(seed),
            "--max-steps",
            str(max_steps),
            "--explorer",
            str(explorer.resolve()),
            "--no-wandb",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    found = RHAE.findall(proc.stdout + proc.stderr)
    results = BENCH / "runs" / label / "results.json"
    levels: dict[str, int] = {}
    if results.exists():
        data = json.loads(results.read_text())
        levels = {d["game"]: d["levels_completed"] for d in data["details"]}
        return {"rhae": float(data["rhae"]), "levels": levels}
    if not found:
        raise RuntimeError(f"no RHAE from {label}: {proc.stderr[-400:]}")
    return {"rhae": float(found[-1]), "levels": levels}


def arm(explorer: Path, name: str, seeds: list[int], max_steps: int) -> list[dict]:
    out = []
    for s in seeds:
        run = one_run(explorer, f"ab-{name}-s{s}", s, max_steps)
        logger.info("{} seed {}: RHAE={:.4f}%", name, s, run["rhae"])
        out.append(run)
    return out


def score_frequency(runs: list[dict]) -> dict[str, int]:
    """How many seeds each game scored at least one level in."""
    freq: dict[str, int] = {}
    for run in runs:
        for game, n in run["levels"].items():
            freq[game] = freq.get(game, 0) + (1 if n > 0 else 0)
    return freq


@app.command()
def main(
    baseline: Path = typer.Option(..., help="Baseline explorer file."),
    candidate: Path = typer.Option(..., help="Candidate explorer file."),
    seeds: int = typer.Option(5, help="Seeds per arm."),
    max_steps: int = typer.Option(600),
) -> None:
    """Report whether the candidate beats the baseline by more than noise."""
    ids = list(range(seeds))
    base = arm(baseline, "base", ids, max_steps)
    cand = arm(candidate, "cand", ids, max_steps)

    b = [r["rhae"] for r in base]
    c = [r["rhae"] for r in cand]
    bm, cm = st.mean(b), st.mean(c)
    sd = st.stdev(b + c) if len(b + c) > 1 else 0.0
    delta = cm - bm
    logger.info(
        "baseline  mean {:.4f}%  sd {:.4f}", bm, st.stdev(b) if len(b) > 1 else 0
    )
    logger.info(
        "candidate mean {:.4f}%  sd {:.4f}", cm, st.stdev(c) if len(c) > 1 else 0
    )
    logger.info("delta {:+.4f}pp against pooled sd {:.4f}pp", delta, sd)

    bf, cf = score_frequency(base), score_frequency(cand)
    moved = [
        (g, bf.get(g, 0), cf.get(g, 0))
        for g in sorted(set(bf) | set(cf))
        if bf.get(g, 0) != cf.get(g, 0)
    ]
    for g, x, y in moved:
        line = logger.success if y > x else logger.warning
        line("{}: scores in {}/{} seeds -> {}/{}", g, x, seeds, y, seeds)
    if not moved:
        logger.info("no game changed how often it scores")

    if sd and abs(delta) < 2 * sd:
        logger.error(
            "INSIDE NOISE — |{:+.4f}| < 2 sd ({:.4f}). Not evidence either way.",
            delta,
            2 * sd,
        )
        raise typer.Exit(1)
    verdict = logger.success if delta > 0 else logger.error
    verdict("{} — {:+.4f}pp is outside 2 sd", "BETTER" if delta > 0 else "WORSE", delta)


if __name__ == "__main__":
    app()
