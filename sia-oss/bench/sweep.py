#!/usr/bin/env python3
"""Sweep one tuning constant and decide whether a gain is real or a lucky draw.

The no-regression gate in `ab.py` catches bad trades but not luck: it passes
a variant whose only effect is that one perturbation happened to land well. Three
changes in one session passed it and were still wrong — a click-repeat limit
reading 0.5089 at exactly 64 with plain baseline at 48 and 96, and a chrome mask
reading 0.5860 at (0.5, 20) while the same mask found the same cells and cleared
a third as many levels at four neighbouring settings.

What separated them from a real change was the shape of the response, so that is
what this measures. A real mechanism improves over a *range*; a lucky rollout
spikes at one value with baseline on both sides.

    python sia-oss/bench/sweep.py --constant CLICK_REPEAT_LIMIT \\
        --values 16,32,48,64,80,96

Restores the file afterwards, including on failure — a half-swept explorer left
on disk would silently poison every later measurement.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "sia-oss/bench"
EXPLORER = REPO / "src/tgaer/agents/arc_agi3_explorer.py"
# A gain appearing at fewer than this fraction of swept values is a spike, not a
# trend, no matter how large it is.
STABLE_FRACTION = 0.5

app = typer.Typer(add_completion=False)


def set_constant(path: Path, name: str, value: str) -> None:
    pattern = re.compile(rf"^{re.escape(name)} = .*$", re.MULTILINE)
    text = path.read_text()
    if not pattern.search(text):
        raise SystemExit(f"{name} is not a module-level constant in {path.name}")
    path.write_text(pattern.sub(f"{name} = {value}", text))


def measure(label: str) -> tuple[float, int]:
    """``(rhae, levels_cleared)`` for the explorer as it currently stands."""
    run = subprocess.run(
        [sys.executable, str(BENCH / "measure.py"), "--label", label, "--no-wandb"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    results = BENCH / "runs" / label / "results.json"
    if run.returncode != 0 or not results.is_file():
        raise SystemExit(f"measure failed for {label}:\n{run.stdout}\n{run.stderr}")
    data = json.loads(results.read_text())
    return data["rhae"], sum(d["levels_completed"] for d in data["details"])


def stability_verdict(
    rows: list[tuple[str, float, int]], baseline: float
) -> tuple[str, str]:
    """``(verdict, explanation)`` for a finished sweep.

    Split out so it can be tested against the real sweeps rather than restated
    in a test: the click-repeat limit gained at 1 of 6 values and the chrome
    mask at 2 of 11, and both were lucky rollouts rather than mechanisms.
    """
    better = [r for r in rows if r[1] > baseline + 1e-9]
    if not better:
        return "NONE", "no gain anywhere. Drop it."
    if len(better) / len(rows) < STABLE_FRACTION:
        best = max(better, key=lambda r: r[1])
        return "SPIKE", (
            f"best is {best[1]:.4f}% at {best[0]} but only {len(better)}/{len(rows)} "
            "values gain. This is what a lucky rollout looks like; do not ship a "
            "tuned constant."
        )
    return "STABLE", (
        f"gains at {len(better)}/{len(rows)} values, worst of them "
        f"{min(r[1] for r in better):.4f}%. Worth gating and promoting."
    )


@app.command()
def main(
    constant: str = typer.Option(
        ..., "--constant", help="Module-level constant to sweep."
    ),
    values: str = typer.Option(..., "--values", help="Comma-separated values to try."),
    baseline: float = typer.Option(
        0.4263, "--baseline", help="RHAE to call 'no change' against."
    ),
) -> None:
    """Try each value, then judge whether any gain is stable or a spike."""
    wanted = [v.strip() for v in values.split(",") if v.strip()]
    original = EXPLORER.read_text()
    rows: list[tuple[str, float, int]] = []
    try:
        for value in wanted:
            set_constant(EXPLORER, constant, value)
            rhae, levels = measure(f"sweep-{constant.lower()}-{value}")
            rows.append((value, rhae, levels))
            logger.info("{}={} RHAE={:.4f}%  levels={}", constant, value, rhae, levels)
    finally:
        EXPLORER.write_text(original)  # never leave a swept file behind

    unchanged = [r for r in rows if abs(r[1] - baseline) <= 1e-9]
    verdict, explanation = stability_verdict(rows, baseline)
    gained = sum(1 for r in rows if r[1] > baseline + 1e-9)
    logger.info("{}: {}/{} values beat {:.4f}%", constant, gained, len(rows), baseline)
    if unchanged:
        logger.warning(
            "{} value(s) scored *exactly* baseline — there the change is inert, "
            "so it is not really being tested",
            len(unchanged),
        )
    logger.info("VERDICT: {} — {}", verdict, explanation)


if __name__ == "__main__":
    app()
