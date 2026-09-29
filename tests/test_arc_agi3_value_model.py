"""Tests for the value-model validation split.

The gate a representation change will be judged on, so its failure modes are
the ones that flatter a model: a fold that trains on the level it is scored on,
and a chance floor set too low.

Loaded by path because `sia-oss/bench` is a script directory, not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "sia-oss/bench"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vm = _load("value_model")


def _steps(levels: list[int]) -> list[dict[str, Any]]:
    return [{"level": lv, "sig": i} for i, lv in enumerate(levels)]


class TestClearedSegments:
    """Only levels that cleared have a winning state to back-label."""

    def test_the_level_in_progress_at_the_end_is_dropped(self) -> None:
        segs = vm.cleared_segments(_steps([0, 0, 0, 1, 1, 1, 2, 2, 2]))
        assert [[s["level"] for s in seg] for seg in segs] == [[0, 0, 0], [1, 1, 1]]

    def test_a_segment_too_short_to_rank_is_dropped(self) -> None:
        segs = vm.cleared_segments(_steps([0, 0, 0, 1, 2, 2, 2]))
        assert [seg[0]["level"] for seg in segs] == [0]


class TestTop1:
    """Expected hits of the best-predicted action; its floor is chance by construction."""

    TRUE = np.array([1.0, 1.0, 2.0, 3.0])

    @pytest.mark.parametrize(
        ("pred", "hits"),
        [
            # Two of four actions are optimal, so a constant prediction hits half
            # the time — not the quarter 1/size gives, which overstates a model.
            ([0.0, 0.0, 0.0, 0.0], 0.5),
            ([1.0, 1.0, 2.0, 3.0], 1.0),
            ([3.0, 9.0, 3.0, 9.0], 0.5),
            ([9.0, 9.0, 0.0, 9.0], 0.0),
        ],
    )
    def test_ties_split_the_pick(self, pred: list[float], hits: float) -> None:
        assert vm.top1(np.array(pred), self.TRUE, np.zeros(4, int)) == (hits, 1)

    def test_a_single_action_state_is_not_ranked(self) -> None:
        true = np.array([1.0, 2.0, 4.0])
        assert vm.top1(np.zeros(3), true, np.array([0, 0, 1])) == (0.5, 1)


@pytest.mark.parametrize(
    ("deltas", "p"),
    [
        ([0.2, 1.0, 0.1, 3.0, 0.5, 2.0], 1 / 64),
        ([1.0, 0.0, 0.0], 0.5),  # zero deltas carry no direction
        ([], 1.0),
    ],
)
def test_sign_test(deltas: list[float], p: float) -> None:
    assert vm.sign_test(deltas) == pytest.approx(p)


def test_back_label_falls_toward_the_winning_state() -> None:
    """Every fold's labels come from here, so a wrong distance poisons them all."""
    seg = [{"sig": s} for s in ("a", "b", "c", "win")]
    assert vm.back_label(seg) == {0: 3.0, 1: 2.0, 2: 1.0}


def test_a_repeated_trajectory_counts_once() -> None:
    """All five tu93 seeds play identically, and a copy is not independent evidence."""
    same = [(1, 15.0, 35, 10.9)]
    episodes = {
        ("tu93", 0): same,
        ("tu93", 1): list(same),
        ("lp85", 0): [(1, 80.0, 156, 76.0)],
    }
    assert list(vm.distinct(episodes)) == [("tu93", 0), ("lp85", 0)]
