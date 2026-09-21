"""Tests for the bench promotion gate and the constant sweeper.

This tooling decides what ships, and its reason for existing is that the suite
score alone is not trustworthy: the unchanged agent's seed-to-seed spread is
sd ~= 0.030pp, so a perturbation that helps nothing can still read well above a
single baseline rollout. The cases below are the real false positives it has to
reject, taken from the runs that produced them.

Loaded by path because `sia-oss/bench` is a script directory, not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parents[1] / "sia-oss/bench"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab = _load("ab")
sweep = _load("sweep")


def _arm(rhaes: list[float], games: dict[str, int], seeds: int = 5) -> list[dict]:
    """One arm: per-seed runs where ``games`` says how many seeds each scores in."""
    return [
        {
            "rhae": rhaes[s],
            "levels": {g: (1 if s < n else 0) for g, n in games.items()},
        }
        for s in range(seeds)
    ]


# The unchanged agent: five seeds, two distinct scores, the five games that
# always clear. Mean 0.1561, sd 0.0280 — the real measured baseline.
FLAT = [0.1868, 0.1868, 0.1357, 0.1357, 0.1357]
ALWAYS = {"lp85": 5, "sp80": 5, "ls20": 5, "ar25": 5, "tu93": 5}
BASE = _arm(FLAT, ALWAYS)


class TestVerdict:
    def test_a_game_losing_its_seeds_fails_even_when_rhae_rises(self) -> None:
        """The real effect-ranked-clicks run: the headline read positive while
        sp80 went 5/5 seeds to 0/5. That trade will not reproduce."""
        cand = _arm([x + 0.2 for x in FLAT], {**ALWAYS, "sp80": 0})
        passed, regressions, _ = ab.verdict(cand=cand, base=BASE)
        assert not passed
        assert regressions == ["sp80: scores in 5 seeds -> 0"]

    def test_one_flipped_seed_is_noise_and_not_a_regression(self) -> None:
        """A single seed flipping is exactly what this tool exists to see through."""
        cand = _arm([x + 0.2 for x in FLAT], {**ALWAYS, "sp80": 4})
        passed, regressions, _ = ab.verdict(cand=cand, base=BASE)
        assert passed and not regressions

    def test_a_gain_inside_the_noise_floor_does_not_pass(self) -> None:
        """-0.0044pp across 5 seeds: the measured result of a change whose single
        rollout read +0.2032pp."""
        cand = _arm([x - 0.0044 for x in FLAT], ALWAYS)
        passed, regressions, _ = ab.verdict(cand=cand, base=BASE)
        assert not passed and not regressions

    def test_a_clean_gain_outside_the_noise_passes_and_is_reported(self) -> None:
        cand = _arm([x + 0.5 for x in FLAT], {**ALWAYS, "cd82": 5})
        passed, _, improvements = ab.verdict(cand=cand, base=BASE)
        assert passed
        assert len(improvements) == 1 and "cd82" in improvements[0]

    def test_an_identical_arm_does_not_pass(self) -> None:
        passed, regressions, _ = ab.verdict(cand=BASE, base=BASE)
        assert not passed and not regressions

    def test_a_game_missing_from_every_candidate_run_is_a_regression(self) -> None:
        """A crashed game yields no scorecard row; that must not read as 'equal'."""
        cand = _arm([x + 0.5 for x in FLAT], {g: 5 for g in ALWAYS if g != "sp80"})
        passed, regressions, _ = ab.verdict(cand=cand, base=BASE)
        assert not passed and any("sp80" in r for r in regressions)


class TestSweepStability:
    """`STABLE_FRACTION` is the line between a trend and a lucky spike. The
    numbers below are the real sweeps: the click-repeat limit gained at 1 of 6
    values, the chrome mask at 2 of 11, and both were wrong."""

    @pytest.mark.parametrize(
        ("gains", "total", "expected"),
        [
            (0, 6, "NONE"),
            (1, 6, "SPIKE"),
            (2, 11, "SPIKE"),
            (4, 6, "STABLE"),
            (6, 6, "STABLE"),
        ],
    )
    def test_a_gain_at_few_values_reads_as_a_spike(
        self, gains: int, total: int, expected: str
    ) -> None:
        rows = [(str(i), 0.5 if i < gains else 0.4263, 6) for i in range(total)]
        verdict, _ = sweep.stability_verdict(rows, baseline=0.4263)
        assert verdict == expected


class TestSetConstant:
    def test_rewrites_only_the_named_constant(self, tmp_path: Path) -> None:
        f = tmp_path / "m.py"
        f.write_text("A = 1\nB = 2\n# A = 99 in a comment\n")
        sweep.set_constant(f, "A", "64")
        assert f.read_text() == "A = 64\nB = 2\n# A = 99 in a comment\n"

    def test_an_absent_constant_raises_rather_than_silently_measuring_nothing(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "m.py"
        f.write_text("A = 1\n")
        with pytest.raises(SystemExit, match="NOPE"):
            sweep.set_constant(f, "NOPE", "1")
