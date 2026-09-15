"""Tests for the SIA harness: `sia_adapter.py`'s pure logic and `sia_eval.py`'s
checkers.

Neither game play nor the adapter subprocess is exercised here — that needs a
live ARC-AGI-3 API key and the starter checkout, and is covered by running the
harness for real (`sia_eval.py --only <case>`). What is covered is the part
that decides whether a run passes: parsing the game id out of a case's prose,
collapsing repeats into the median SIA optimises against, and the checkers
`sia evals run` actually grades on.
"""

from __future__ import annotations

import pytest

from tgaer.agents import sia_adapter as adapter
from tgaer.agents import sia_eval as ev


class TestParseGame:
    @pytest.mark.parametrize(
        "text",
        [
            "Play lp85 and report what it scored.",
            "PLAY LP85.",
            "  lp85  ",
            "lp85",
        ],
    )
    def test_finds_the_game_id_regardless_of_case_or_surrounding_prose(
        self, text: str
    ) -> None:
        assert adapter.parse_game(text) == "lp85"

    def test_repeated_mentions_of_the_same_game_are_not_ambiguous(self) -> None:
        assert adapter.parse_game("Play lp85. Yes, lp85.") == "lp85"

    @pytest.mark.parametrize(
        "text",
        ["", "please play a game", "score the agent"],
    )
    def test_no_game_id_raises_rather_than_guessing(self, text: str) -> None:
        with pytest.raises(ValueError, match="no ARC-AGI-3 game id"):
            adapter.parse_game(text)

    def test_two_different_game_ids_raise_rather_than_picking_one(self) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            adapter.parse_game("Play lp85 then tu93.")


class TestLevelSummary:
    """The reported ratio is a median across repeats, not a single draw — a
    single cleared level has failed to reproduce across three repeats before."""

    def test_actions_are_the_median_across_repeats_that_cleared_it(self) -> None:
        rows = [
            {"game": "lp85", "level": 1, "baseline": 17, "actions": a}
            for a in (10, 12, 40)
        ]
        (summary,) = adapter.level_summary(rows, repeats=3)
        assert summary["our_actions_median"] == 12.0
        assert summary["cleared_in_repeats"] == 3

    def test_a_level_only_some_repeats_cleared_reports_the_smaller_count(self) -> None:
        rows = [{"game": "lp85", "level": 1, "baseline": 17, "actions": 10}]
        (summary,) = adapter.level_summary(rows, repeats=3)
        assert summary["cleared_in_repeats"] == 1

    def test_the_score_formula_is_reproduced_exactly(self) -> None:
        """min((baseline / actions)^2 * 100, 115) — the ARC-AGI-3 scoring rule
        this whole project optimises against. baseline=17/actions=10 pushes the
        raw ratio past the 115 cap (289), so this also covers the clamp."""
        rows = [{"game": "lp85", "level": 1, "baseline": 17, "actions": 10}]
        (summary,) = adapter.level_summary(rows, repeats=1)
        assert summary["level_score"] == 115.0

    def test_the_score_formula_is_uncapped_below_115(self) -> None:
        rows = [{"game": "tu93", "level": 1, "baseline": 19, "actions": 383}]
        (summary,) = adapter.level_summary(rows, repeats=1)
        assert summary["level_score"] == round((19 / 383) ** 2 * 100, 2)

    def test_no_cleared_levels_summarises_to_an_empty_list(self) -> None:
        assert adapter.level_summary([], repeats=3) == []


class TestVerdict:
    def test_names_the_slowest_cleared_level_not_the_first(self) -> None:
        levels = [
            {
                "level": 1,
                "ratio_vs_baseline": 2.0,
                "our_actions_median": 20,
                "human_baseline_actions": 10,
                "level_score": 25.0,
            },
            {
                "level": 2,
                "ratio_vs_baseline": 8.0,
                "our_actions_median": 80,
                "human_baseline_actions": 10,
                "level_score": 1.5,
            },
        ]
        assert "level 2" in adapter.verdict(levels, "tu93")

    def test_no_cleared_levels_says_so_plainly(self) -> None:
        assert adapter.verdict([], "sk48") == (
            "No level of sk48 was cleared, so it scored nothing."
        )


class TestCheckLevels:
    """`check_levels` is what `sia evals run` grades every case on — it has to
    fail a run for the exact reason a human reading the scorecard would."""

    LEVEL = {
        "level": 1,
        "ratio_vs_baseline": 3.0,
        "cleared_in_repeats": 3,
        "our_actions_median": 60,
        "human_baseline_actions": 20,
        "level_score": 11.1,
    }

    def test_require_levels_passes_when_cleared_in_every_repeat(self) -> None:
        assert (
            ev.check_levels({1: self.LEVEL}, {"require_levels": [1]}, repeats=3) == ""
        )

    def test_require_levels_fails_a_level_never_cleared(self) -> None:
        assert "never cleared" in ev.check_levels(
            {}, {"require_levels": [1]}, repeats=3
        )

    def test_require_levels_fails_a_level_only_sometimes_cleared(self) -> None:
        flaky = {**self.LEVEL, "cleared_in_repeats": 1}
        reason = ev.check_levels({1: flaky}, {"require_levels": [1]}, repeats=3)
        assert "1 of 3 repeats" in reason

    def test_max_ratio_fails_a_level_over_the_cap(self) -> None:
        reason = ev.check_levels({1: self.LEVEL}, {"max_ratio": 2.0}, repeats=3)
        assert "cap 2.0x" in reason

    def test_max_ratio_passes_a_level_at_the_cap(self) -> None:
        assert ev.check_levels({1: self.LEVEL}, {"max_ratio": 3.0}, repeats=3) == ""

    def test_min_levels_any_counts_cleared_levels_regardless_of_reliability(
        self,
    ) -> None:
        flaky = {**self.LEVEL, "cleared_in_repeats": 1}
        assert ev.check_levels({1: flaky}, {"min_levels_any": 1}, repeats=3) == ""

    def test_min_levels_any_fails_when_too_few_cleared(self) -> None:
        assert "needed at least 2" in ev.check_levels(
            {1: self.LEVEL}, {"min_levels_any": 2}, repeats=3
        )


class TestCheck:
    CARD = {
        "game": "lp85",
        "repeats": 3,
        "faults": [],
        "levels": [
            {
                "level": 1,
                "ratio_vs_baseline": 1.0,
                "cleared_in_repeats": 3,
                "our_actions_median": 10,
                "human_baseline_actions": 10,
                "level_score": 100.0,
            },
        ],
    }

    def test_a_clean_run_passes(self) -> None:
        correct, reason = ev.check(self.CARD, {"require_levels": [1]})
        assert correct and "L1" in reason

    def test_an_adapter_error_fails_regardless_of_the_check_spec(self) -> None:
        correct, reason = ev.check({"error": "env-unavailable"}, {})
        assert not correct and "env-unavailable" in reason

    def test_an_empty_scorecard_fails_rather_than_vacuously_passing(self) -> None:
        """An empty `{}` satisfies no require_levels/max_ratio condition
        (nothing to violate) — this guards against that silently passing."""
        correct, reason = ev.check({}, {"require_levels": [1]})
        assert not correct and "no scorecard" in reason

    def test_a_nonempty_fault_counter_fails_even_if_the_ratio_check_would_pass(
        self,
    ) -> None:
        broken = {**self.CARD, "faults": ["image_unavailable=12"]}
        correct, reason = ev.check(broken, {"require_levels": [1]})
        assert not correct and "fault counters" in reason


class TestParseScorecard:
    def test_extracts_the_json_block_after_the_marker(self) -> None:
        output = f'some prose\n\n{adapter.SCORECARD_MARKER}\n{{"game": "lp85"}}'
        assert ev.parse_scorecard(output) == {"game": "lp85"}

    def test_no_marker_present_returns_empty_rather_than_raising(self) -> None:
        assert ev.parse_scorecard("no scorecard here") == {}

