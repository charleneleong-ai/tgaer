"""Tests for the bench promotion gate and the constant sweeper.

This tooling decides what ships, and its whole reason for existing is that the
suite score alone is not trustworthy: the rollout is deterministic and chaotic,
so a perturbation that helps nothing can still read well above baseline. The
cases below are the real false positives it has to reject, taken from the runs
that produced them.

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


gate = _load("gate")
sweep = _load("sweep")


def _results(rhae: float, games: dict[str, tuple[int, float]]) -> dict[str, Any]:
    """A results.json shaped payload: ``game -> (levels_completed, env_score)``."""
    return {
        "rhae": rhae,
        "details": [
            {"game": g, "levels_completed": lv, "env_score": es}
            for g, (lv, es) in games.items()
        ],
    }


BASE = _results(
    0.4263, {"lp85": (1, 0.02778), "sp80": (1, 0.00196), "ls20": (1, 0.00374)}
)


class TestVerdict:
    def test_a_lost_level_fails_even_when_rhae_rises(self) -> None:
        """The real `affordance-first` run: sc25 improved, sp80 lost its only
        level. A trade like that nets positive locally and will not reproduce."""
        cand = _results(
            0.4025, {"lp85": (1, 0.02778), "sp80": (0, 0.0), "ls20": (1, 0.00374)}
        )
        passed, regressions, _ = gate.verdict(cand, BASE)
        assert not passed
        assert regressions == ["sp80: 1 -> 0 levels"]

    def test_a_dropped_env_score_at_equal_levels_still_fails(self) -> None:
        cand = _results(
            0.9, {"lp85": (1, 0.02778), "sp80": (1, 0.0005), "ls20": (1, 0.00374)}
        )
        passed, regressions, _ = gate.verdict(cand, BASE)
        assert not passed and "sp80" in regressions[0]

    def test_a_clean_gain_passes_and_is_reported(self) -> None:
        cand = _results(
            0.586, {"lp85": (3, 0.04056), "sp80": (1, 0.00196), "ls20": (1, 0.00374)}
        )
        passed, _, improvements = gate.verdict(cand, BASE)
        assert passed
        assert len(improvements) == 1 and "lp85" in improvements[0]

    def test_no_regression_but_flat_rhae_does_not_pass(self) -> None:
        passed, regressions, _ = gate.verdict(BASE, BASE)
        assert not passed and not regressions

    def test_a_game_missing_from_the_candidate_is_a_regression(self) -> None:
        """A crashed game yields no scorecard row; that must not read as 'equal'."""
        cand = _results(0.9, {"lp85": (1, 0.02778), "ls20": (1, 0.00374)})
        passed, regressions, _ = gate.verdict(cand, BASE)
        assert not passed and "sp80" in regressions[0]


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
