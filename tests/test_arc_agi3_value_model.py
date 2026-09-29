"""Tests for the value-model bench: its validation split and the models it compares.

The split is the gate a representation change is judged on, so its failure modes
are the ones that flatter a model: a fold that trains on the level it is scored
on, and a chance floor set too low. The conv encoder's tests skip without torch,
which lives in the optional ``bench`` group that CI does not install.

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


def _table(y: list[float], g: list[int]) -> Any:
    n = len(y)
    return vm.Table(
        np.zeros((n, 1)),
        np.array(y, float),
        np.array(g),
        np.zeros((n, 2, 2)),
        [("act", 1)] * n,
    )


def _oracle(train: list[Any], test: Any) -> np.ndarray:
    return test.y


def _blind(train: list[Any], test: Any) -> np.ndarray:
    return np.zeros_like(test.y)


def _toy(n: int, seed: int) -> tuple[np.ndarray, list[tuple], np.ndarray, np.ndarray]:
    """Step up or down toward a goal 2-3 cells from the avatar.

    The pair's rows are drawn first and only then is it decided which colour sits
    on top, so the colour histogram, centroid and spread — every global statistic
    the baseline reads — are distributed identically either way. Only which colour
    is above the other says which step is right.
    """
    rng = np.random.default_rng(seed)
    grids, prims, y, g = [], [], [], []
    for state in range(n):
        board = np.zeros((12, 12), np.int16)
        gap = int(rng.integers(2, 4))
        top, c = int(rng.integers(1, 11 - gap)), int(rng.integers(1, 11))
        above = bool(rng.integers(2))  # is the goal the upper cell?
        board[top, c], board[top + gap, c] = (3, 2) if above else (2, 3)
        for act, toward in ((1, above), (2, not above)):  # 1 steps up, 2 steps down
            grids.append(board)
            prims.append(("act", act))
            y.append(1.0 if toward else 3.0)
            g.append(state)
    return np.stack(grids), prims, np.array(y), np.array(g)


@pytest.fixture(scope="module")
def ge() -> Any:
    pytest.importorskip("torch")
    return _load("grid_encoder")


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


class TestCachedRollout:
    """A trajectory is replayed from disk until the seed, budget or agent changes."""

    @pytest.fixture
    def calls(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
        explorer = tmp_path / "explorer.py"
        explorer.write_text("v1")
        made: list[tuple] = []

        def rollout(game_id: str, max_steps: int, seed: int) -> list[dict[str, Any]]:
            made.append((game_id, max_steps, seed))
            return [{"level": 0, "game": game_id}]

        monkeypatch.setattr(vm, "CACHE", tmp_path / "cache")
        monkeypatch.setattr(vm, "EXPLORER", explorer)
        monkeypatch.setattr(vm, "rollout", rollout)
        return made

    def test_a_second_call_replays_without_rolling_out(
        self, calls: list[tuple]
    ) -> None:
        first = vm.cached_rollout("lp85", 0, 600)
        assert vm.cached_rollout("lp85", 0, 600) == first
        assert calls == [("lp85", 600, 0)]
        assert not list(vm.CACHE.glob("*.tmp"))

    @pytest.mark.parametrize(
        ("seed", "budget", "agent", "recorder"),
        [
            (1, 600, "v1", False),
            (0, 700, "v1", False),
            (0, 600, "v2", False),
            (0, 600, "v1", True),
        ],
    )
    def test_any_change_is_a_miss(
        self,
        calls: list[tuple],
        monkeypatch: pytest.MonkeyPatch,
        seed: int,
        budget: int,
        agent: str,
        recorder: bool,
    ) -> None:
        vm.cached_rollout("lp85", 0, 600)
        vm.EXPLORER.write_text(agent)
        if recorder:

            def rollout(
                game_id: str, max_steps: int, seed: int
            ) -> list[dict[str, Any]]:
                calls.append(("rewritten recorder", max_steps, seed))
                return []

            monkeypatch.setattr(vm, "rollout", rollout)
        vm.cached_rollout("lp85", seed, budget)
        assert len(calls) == 2


class TestScoreFold:
    """Every model is scored on the same states, against the same floor."""

    TEST = _table([1, 1, 2, 3, 5, 9], [0, 0, 0, 0, 1, 1])

    def test_a_perfect_model_hits_every_state_and_a_blind_one_scores_chance(
        self,
    ) -> None:
        models = {"oracle": _oracle, "blind": _blind}
        fold = vm.score_fold(1, [_table([1.0], [0])], self.TEST, models)
        assert fold == (1, 2, {"chance": 1.0, "oracle": 2.0, "blind": 1.0})

    @pytest.mark.parametrize(
        ("train", "test"),
        [([], TEST), ([_table([], [])], TEST), ([_table([1.0], [0])], _table([], []))],
    )
    def test_nothing_to_train_on_or_rank_is_unscorable(
        self, train: list, test: Any
    ) -> None:
        assert vm.score_fold(1, train, test, {"oracle": _oracle}) is None


class TestGridEncoder:
    """The conv sees where things are, which the scalar baseline cannot."""

    def test_a_click_marks_its_cell_and_a_simple_action_its_id(self, ge: Any) -> None:
        grids = np.zeros((2, 6, 6), np.int16)
        planes, ids = ge.encode(grids, [("click", 3, 5), ("act", 4)])
        assert planes[0, ge.COLOURS].nonzero().tolist() == [[3, 5]]
        assert planes[1, ge.COLOURS].sum() == 0
        assert ids.tolist() == [0, 4]

    def test_it_learns_a_relation_no_global_statistic_can_express(
        self, ge: Any
    ) -> None:
        grids, prims, y, _ = _toy(150, seed=1)
        test_grids, test_prims, test_y, test_g = _toy(50, seed=2)
        pred = ge.fit_predict(grids, prims, y, test_grids, test_prims, steps=300)
        hits, seen = vm.top1(pred, test_y, test_g)
        assert hits / seen >= 0.9  # chance is 0.5
