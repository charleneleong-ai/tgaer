"""Tests for the official ARC-AGI-3 RHAE metric as `evaluate.py` implements it.

RHAE is what the Kaggle leaderboard reports and what `supervisor.py` promotes
on, so the formula has to match the ARC-AGI-3 Technical Report §4.1 exactly:
squared per-level efficiency capped at 1.15, a level-index-weighted mean per
environment, itself capped by the weighted fraction of levels completed, then
averaged across environments. The cases below are the report's own worked
examples plus the suite's real baseline games.

This lives apart from `test_arc_agi3_sia.py` because that module imports
`sia_adapter`, which currently fails to import (it references `game_key` and
`run_levels`, neither of which exists in `arc_agi3_score_local`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# The grader is a standalone script — it runs inside the generation sandbox
# with no guarantee `tgaer` is importable — so load it by path.
EVALUATE_PY = (
    Path(__file__).resolve().parents[1]
    / "sia-oss/tasks/arc-agi3/data/public/evaluate.py"
)
_spec = importlib.util.spec_from_file_location("sia_evaluate", EVALUATE_PY)
grader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grader)

LP85_BASELINES = [17, 38, 31, 16, 41, 60, 26, 159]
LS20_BASELINES = [22, 123, 73, 84, 96, 192, 186]


def _level(level: int, actions: float) -> dict[str, float]:
    return {"level": level, "our_actions_median": actions}


class TestLevelScore:
    @pytest.mark.parametrize(
        ("baseline", "actions", "expected"),
        [
            (10, 100, 0.01),  # report §4.2: 10/100 -> 1%
            (20, 2, 1.15),  # report §4.2: a 10x exploit is capped at 1.15
            (17, 10, 1.15),  # lp85 L1 sits at the cap
            (22, 68, (22 / 68) ** 2),  # ls20 L1 at 3.1x -> ~10%
        ],
    )
    def test_is_squared_efficiency_capped_at_115(
        self, baseline: int, actions: int, expected: float
    ) -> None:
        assert grader.level_score(baseline, actions) == pytest.approx(expected)


class TestEnvironmentScore:
    @pytest.mark.parametrize(
        ("completed", "expected_cap"),
        [(5, 15 / 15), (4, 10 / 15), (3, 6 / 15), (0, 0.0)],
    )
    def test_cap_is_the_weighted_fraction_of_levels_completed(
        self, completed: int, expected_cap: float
    ) -> None:
        """Report §4.2's worked 5-level example: 4 of 5 caps at 10/15."""
        levels = {lvl: _level(lvl, 10) for lvl in range(1, completed + 1)}
        env = grader.environment_score(levels, [10] * 5)
        assert env["cap"] == pytest.approx(expected_cap)
        assert env["env_score"] == pytest.approx(expected_cap)
        assert env["levels_completed"] == completed

    def test_cap_binds_when_efficiency_would_exceed_it(self) -> None:
        """lp85: level 1 of 8 at the 1.15 cap still only earns 1/36, not
        1.15/36 — this is why the old unweighted score was so misleading."""
        env = grader.environment_score({1: _level(1, 10)}, LP85_BASELINES)
        assert env["weighted_efficiency"] == pytest.approx(1.15 / 36)
        assert env["env_score"] == pytest.approx(1 / 36)

    def test_efficiency_binds_when_below_the_cap(self) -> None:
        env = grader.environment_score({1: _level(1, 68)}, LS20_BASELINES)
        assert env["env_score"] == pytest.approx((22 / 68) ** 2 / 28)
        assert env["env_score"] < env["cap"]

    def test_later_levels_weigh_more_than_earlier_ones(self) -> None:
        """w_l = l, so clearing L2 at the same efficiency triples the total."""
        baselines = [10, 10, 10]
        only_l1 = grader.environment_score({1: _level(1, 20)}, baselines)
        l1_and_l2 = grader.environment_score(
            {1: _level(1, 20), 2: _level(2, 20)}, baselines
        )
        assert l1_and_l2["weighted_efficiency"] == pytest.approx(
            3 * only_l1["weighted_efficiency"]
        )

    def test_uncompleted_levels_score_zero_and_do_not_raise(self) -> None:
        env = grader.environment_score({}, LP85_BASELINES)
        assert env["env_score"] == 0.0 and env["levels_completed"] == 0


class TestLevelsCompleted:
    def test_levels_are_sequential_so_a_gap_stops_the_count(self) -> None:
        assert grader.levels_completed({1: _level(1, 5), 3: _level(3, 5)}) == 1

    def test_a_missing_level_one_means_nothing_completed(self) -> None:
        assert grader.levels_completed({2: _level(2, 5)}) == 0


class TestTotal:
    def test_is_the_mean_over_environments_as_a_percent(self) -> None:
        assert grader.rhae_percent([1 / 36, 0.0, 0.0, 0.0]) == pytest.approx(
            100 * (1 / 36) / 4
        )


class TestFindSubmissionFile:
    def test_prefers_submission_json_over_a_newer_sibling(self, tmp_path: Path) -> None:
        """run_9 wrote baseline.json *after* candidate.json; picking by mtime
        graded the baseline and reported a false no-op."""
        results = tmp_path / "results"
        results.mkdir()
        (results / "submission.json").write_text("{}")
        newer = results / "baseline.json"
        newer.write_text("{}")
        import os

        os.utime(newer, (10**9, 10**9))  # far future relative to submission.json
        assert grader.find_submission_file(tmp_path).name == "submission.json"

    def test_falls_back_to_the_newest_json_when_unnamed(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        results.mkdir()
        (results / "candidate.json").write_text("{}")
        assert grader.find_submission_file(tmp_path).name == "candidate.json"

    def test_no_results_dir_returns_none(self, tmp_path: Path) -> None:
        assert grader.find_submission_file(tmp_path) is None
