#!/usr/bin/env python3
"""Turn each mechanism off in turn and ask whether the agent misses it.

Eleven *additions* have been measured and reverted. Nothing has ever been
removed. The explorer accumulated its mechanisms over months, each gated on the
25 public games — which the competition builds to be out-of-distribution from
the scored set — and several were never re-measured after later changes moved
the ground under them. `_inert`'s docstring asks for exactly that after the
chrome mask, which shipped in #29 and was never followed up.

A removal that costs nothing is worth taking: it is one less mechanism tuned on
a set that does not predict the score, and one less thing to reason about. A
removal that costs something is evidence the mechanism earns its place, which we
do not currently have for any of them.

Judged by `ab.py`'s verdict — mean RHAE against two pooled standard deviations,
plus per-game frequency and depth — against the unchanged agent over the same
seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "sia-oss/bench"
sys.path.insert(0, str(BENCH))

import ab as AB  # noqa: E402
import sweep as SW  # noqa: E402

EXPLORER = REPO / "src/tgaer/agents/arc_agi3_explorer.py"
TOGGLES = (
    "USE_PROBE",
    "USE_AFFORDANCE",
    "USE_NAV",
    "USE_INERT",
    "USE_CHURN_MASK",
    "USE_FRONTIER",
    "USE_GOAL_INDUCTION",
)

app = typer.Typer(add_completion=False)


def baseline(label: str, seeds: int) -> list[dict]:
    """Reuse an existing unchanged-agent sweep rather than re-running it."""
    runs = []
    for s in range(seeds):
        path = BENCH / "runs" / f"{label}{s}" / "results.json"
        if not path.exists():
            raise typer.BadParameter(f"no baseline run at {path}")
        data = json.loads(path.read_text())
        runs.append(
            {
                "rhae": float(data["rhae"]),
                "levels": {d["game"]: d["levels_completed"] for d in data["details"]},
            }
        )
    return runs


@app.command()
def main(
    toggles: str = typer.Option(",".join(TOGGLES), help="Constants to switch off."),
    seeds: int = typer.Option(5),
    max_steps: int = typer.Option(600),
    base_label: str = typer.Option("postmerge-s", help="Existing baseline run prefix."),
) -> None:
    """Switch each mechanism off over N seeds and report what it was worth."""
    base = baseline(base_label, seeds)
    import statistics as st

    bm = st.mean([r["rhae"] for r in base])
    logger.info("baseline {} seeds, mean {:.4f}%", seeds, bm)

    original = EXPLORER.read_text()
    results: list[tuple[str, float, bool, list[str]]] = []
    try:
        for name in toggles.split(","):
            SW.set_constant(EXPLORER, name, "False")
            cand = []
            for s in range(seeds):
                run = AB.one_run(EXPLORER, f"ablate-{name.lower()}-s{s}", s, max_steps)
                cand.append(run)
            SW.set_constant(EXPLORER, name, "True")
            cm = st.mean([r["rhae"] for r in cand])
            passed, regressions, _ = AB.verdict(base=base, cand=cand)
            results.append((name, cm - bm, passed, regressions))
            note = "; ".join(regressions[:2]) if regressions else "no game regressed"
            logger.info(
                "{:18} off -> {:.4f}%  ({:+.4f}pp)  {}",
                name,
                cm,
                cm - bm,
                note,
            )
    finally:
        EXPLORER.write_text(original)  # never leave a switched file behind

    logger.info("")
    logger.info("{:18} {:>10} {:>12}  verdict", "mechanism", "delta", "regressions")
    for name, delta, _passed, regs in sorted(results, key=lambda r: r[1]):
        # A mechanism worth keeping is one whose removal *hurts* beyond noise.
        earns = delta < -2 * 0.0198 or regs
        logger.info(
            "{:18} {:+9.4f}pp {:>12}  {}",
            name,
            delta,
            len(regs),
            "earns its place" if earns else "REMOVABLE",
        )


if __name__ == "__main__":
    app()
