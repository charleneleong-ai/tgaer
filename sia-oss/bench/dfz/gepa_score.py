"""Score a warm-suite run against the warm baseline on the tune/holdout split; a level clears only when every repeat does."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import typer
from harness import STALL_REPEATS, EvalRun, Workdir, level_key, parse_episode_id

SPLIT = Path(__file__).resolve().parent / "split.json"
EPISODE = re.compile(r"^ +(\S+): score=[\d.]+, levels=([\d.]+)/", re.M)
REPEAT_SD = 3.7  # episodes cleared, between identical warm-baseline runs

Clears = dict[str, list[int]]


class SuiteScore:
    """Per-level repeat clears for a candidate run and its baseline."""

    def __init__(self, candidate: Clears, baseline: Clears) -> None:
        self.candidate = candidate
        self.baseline = baseline

    @staticmethod
    def clears(summary: str) -> Clears:
        by: Clears = defaultdict(list)
        for eid, levels in EPISODE.findall(summary):
            if (parsed := parse_episode_id(eid)) is not None:
                code, level, _ = parsed
                by[level_key(code, level)].append(int(float(levels) >= level))
        return dict(by)

    @staticmethod
    def cleared(clears: Clears, key: str) -> bool:
        reps = clears.get(key, [])
        return len(reps) == STALL_REPEATS and all(reps)

    @staticmethod
    def episodes(clears: Clears, levels: list[str]) -> int:
        return sum(sum(clears.get(k, [])) for k in levels)

    def line(self, part: str, levels: list[str]) -> str:
        c, b = self.candidate, self.baseline
        n = len(levels) * STALL_REPEATS
        both_b, both_c = (sum(self.cleared(d, k) for k in levels) for d in (b, c))
        out = (
            f"{part:8s} both-repeat clears {both_b} -> {both_c}"
            f"  | episodes {self.episodes(b, levels)}/{n} -> {self.episodes(c, levels)}/{n}"
        )
        if part != "tune":
            return out
        gained = [k for k in levels if self.cleared(c, k) and not self.cleared(b, k)]
        lost = [k for k in levels if self.cleared(b, k) and not self.cleared(c, k)]
        return f"{out}  | gained {gained} lost {lost}"

    @staticmethod
    def total(clears: Clears) -> int:
        return sum(map(sum, clears.values()))

    def pooled_line(self, baseline_rep: Clears, n_levels: int) -> str:
        got = self.total(self.candidate)
        pooled = (self.total(self.baseline) + self.total(baseline_rep)) / 2
        n = n_levels * STALL_REPEATS
        return (
            f"episodes cleared {got}/{n} vs baseline mean {pooled:.1f}/{n} "
            f"(gain {got - pooled:+.1f}; between identical runs sd ~{REPEAT_SD}, so a real gain needs about +7)"
        )


def main(
    candidate: Path = typer.Argument(
        ..., help="Candidate run output dir (holds summary.txt)."
    ),
    workdir: Path | None = typer.Option(None, envvar="DFZ_WORKDIR"),
    baseline: Path | None = typer.Option(
        None, help="Default: the workdir's stallsuite_warm run."
    ),
    baseline_rep: Path | None = typer.Option(
        None, help="Repeat baseline, pooled when present; default stallsuite_warm_rep2."
    ),
    split: Path = typer.Option(SPLIT),
) -> None:
    if workdir is None and (baseline is None or baseline_rep is None):
        raise typer.BadParameter(
            "pass --baseline and --baseline-rep, or set DFZ_WORKDIR"
        )
    wd = Workdir(workdir) if workdir else None
    baseline = baseline or wd.run("stallsuite_warm").root
    baseline_rep = baseline_rep or wd.run("stallsuite_warm_rep2").root
    parts = json.loads(split.read_text())
    clears = [SuiteScore.clears(EvalRun(d).summary()) for d in (candidate, baseline)]
    score = SuiteScore(*clears)
    for part in ("tune", "holdout"):
        print(score.line(part, parts[part]))
    if baseline_rep.exists():
        rep = SuiteScore.clears(EvalRun(baseline_rep).summary())
        print(score.pooled_line(rep, len(parts["tune"]) + len(parts["holdout"])))


if __name__ == "__main__":
    typer.run(main)
