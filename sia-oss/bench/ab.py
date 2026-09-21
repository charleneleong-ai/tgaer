#!/usr/bin/env python3
"""A/B two explorer variants across seeds, and judge the delta against noise.

This is the promotion gate. It replaces `gate.py`, which compared one rollout of
each variant — sound only while the agent was deterministic, which stopped being
true when the salted tie-break landed. A single rollout cannot tell a real gain
from a reshuffle: the seed-to-seed spread of the *unchanged* agent is sd ~=
0.030pp, and four of six changes gated in one session landed inside it.

Worse, single-run gating was biased. The 0.1879% baseline every change was
measured against sits 1.13 sd above the unchanged agent's 5-seed mean of
0.1561%, so candidates had to beat a lucky draw while being judged on one of
their own.

So: run both arms over N seeds, compare means against the pooled spread, and
carry forward the one rule `gate.py` got right — no game may go backwards —
applied to how *often* a game scores rather than to a single rollout of it. A
game scoring in 5/5 seeds for one arm and 0/5 for the other is a finding; a game
that flips once is not.
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

# A game must lose this many seeds' worth of scoring before it counts as a
# regression: one flip is the noise this tool exists to see through.
MIN_FREQ_DROP = 2

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


def pooled_sd(b: list[float], c: list[float]) -> float:
    """Within-arm spread, pooled across both arms.

    Not `stdev(b + c)`: pooling the arms together folds the effect being tested
    into the denominator, so a large true gain inflates its own bar and can
    never clear it. A +0.5pp shift on these five seeds reads sd 0.25, making
    2 sd exactly 0.5.
    """
    nb, nc = len(b), len(c)
    if nb < 2 or nc < 2:
        return 0.0
    vb, vc = st.variance(b), st.variance(c)
    return (((nb - 1) * vb + (nc - 1) * vc) / (nb + nc - 2)) ** 0.5


def score_frequency(runs: list[dict]) -> dict[str, int]:
    """How many seeds each game scored at least one level in."""
    freq: dict[str, int] = {}
    for run in runs:
        for game, n in run["levels"].items():
            freq[game] = freq.get(game, 0) + (1 if n > 0 else 0)
    return freq


def verdict(
    base: list[dict], cand: list[dict], min_freq_drop: int = MIN_FREQ_DROP
) -> tuple[bool, list[str], list[str]]:
    """``(passed, regressions, improvements)`` for a candidate against a baseline.

    Passes only when the mean RHAE gain clears two pooled standard deviations
    *and* no game scores in materially fewer seeds. Either alone has promoted a
    change that did not reproduce.
    """
    b = [r["rhae"] for r in base]
    c = [r["rhae"] for r in cand]
    delta = st.mean(c) - st.mean(b)
    sd = pooled_sd(b, c)

    bf, cf = score_frequency(base), score_frequency(cand)
    regressions: list[str] = []
    improvements: list[str] = []
    for game in sorted(set(bf) | set(cf)):
        was, now = bf.get(game, 0), cf.get(game, 0)
        if now - was <= -min_freq_drop:
            regressions.append(f"{game}: scores in {was} seeds -> {now}")
        elif now - was >= min_freq_drop:
            improvements.append(f"{game}: scores in {was} seeds -> {now}")
    for game in sorted(set(bf) - {g for r in cand for g in r["levels"]}):
        # A crashed game yields no scorecard row; that must not read as equal.
        regressions.append(f"{game}: missing from the candidate scorecards")

    beats_noise = delta > 2 * sd if sd else delta > 0
    return (beats_noise and not regressions), regressions, improvements


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
    sd = pooled_sd(b, c)
    delta = cm - bm
    logger.info(
        "baseline  mean {:.4f}%  sd {:.4f}", bm, st.stdev(b) if len(b) > 1 else 0
    )
    logger.info(
        "candidate mean {:.4f}%  sd {:.4f}", cm, st.stdev(c) if len(c) > 1 else 0
    )
    logger.info("delta {:+.4f}pp against pooled sd {:.4f}pp", delta, sd)

    passed, regressions, improvements = verdict(base, cand)
    for line in improvements:
        logger.success("better   {}", line)
    for line in regressions:
        logger.warning("WORSE    {}", line)
    if not (improvements or regressions):
        logger.info("no game changed how often it scores")

    if passed:
        logger.success("PASS — {:+.4f}pp clears 2 sd and no game regressed.", delta)
        return
    if regressions:
        logger.error("FAIL — {} game(s) regressed. Do not promote.", len(regressions))
    elif sd and abs(delta) < 2 * sd:
        logger.error("FAIL — INSIDE NOISE: |{:+.4f}| < 2 sd ({:.4f}).", delta, 2 * sd)
    else:
        logger.error("FAIL — {:+.4f}pp is not an improvement.", delta)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
