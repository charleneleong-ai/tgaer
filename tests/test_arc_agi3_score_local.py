"""Tests for the local scoring harness.

The harness exists to make agent comparisons trustworthy, so the checks that
decide whether a run is readable at all — which features fired, which fault
counters moved, whether the seed reached the model — need to be right, or
every measurement taken with it is suspect.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from rich.table import Table

from tgaer.evaluation import arc_agi3_score_local as sl


class TestRender:
    ROWS = [
        {
            "game": "sk48",
            "levels_completed": 0,
            "actions": 201,
            "state": "GameState.NOT_FINISHED",
            "seconds": 1.0,
        }
    ]

    def test_a_backend_without_token_stats_does_not_kill_the_run(self) -> None:
        """The HTTP backend counts no tokens. Crashing on the decorative stat
        line skipped the inert-feature check downstream of it, so a run that
        had silently lost a feature reported nothing at all."""
        sl.render(self.ROWS, "test", 0.0, 0, backend=object())


class TestResolveGames:
    AVAILABLE = ["sk48", "tn36", "ls20", "vc33"]

    @pytest.mark.parametrize(
        ("selector", "expected"),
        [
            ("competition", ["sk48", "tn36"]),
            ("all", AVAILABLE),
            ("ls20,vc33", ["ls20", "vc33"]),
            ("ls20", ["ls20"]),
        ],
    )
    def test_selectors(self, selector: str, expected: list[str]) -> None:
        assert sl.resolve_games(selector, self.AVAILABLE) == expected

    def test_unknown_game_is_rejected(self) -> None:
        with pytest.raises(Exception, match="unknown game id"):
            sl.resolve_games("nope42", self.AVAILABLE)


class TestFeatureAssertions:
    """A feature that never fires must not look like one that did not help.

    Vision shipped inert twice and undo once; each run still printed a tidy
    zero, which is indistinguishable from a feature that ran and was useless.
    """

    class Agent:
        SEND_IMAGE = True
        PROBE_ACTIONS = True
        REPL_STEPS = 0
        EXPLOIT_REPEATS = 8
        MECHANIC_NOTES = False

    def test_flags_an_enabled_feature_that_never_fired(self) -> None:
        inert = sl.inert_features(
            self.Agent, {"probe": 5, "exploit": 3, "forward_predicted": 40}
        )
        assert inert == ["image_sent"]

    def test_silent_when_every_enabled_feature_fired(self) -> None:
        assert (
            sl.inert_features(
                self.Agent,
                {"image_sent": 900, "probe": 5, "exploit": 3, "forward_predicted": 40},
            )
            == []
        )

    def test_disabled_features_are_not_required_to_fire(self) -> None:
        """REPL_STEPS is 0 here, so repl_call being absent is correct."""
        assert "repl_call" not in sl.inert_features(
            self.Agent,
            {"image_sent": 1, "probe": 1, "exploit": 1, "forward_predicted": 1},
        )


class TestFaultCounters:
    """Counters that mean a path was reached that never should be.

    inert_features asks whether something never happened; this asks whether
    something happened that shouldn't have. Both are invisible in the score and
    look identical in a results table, and the second is what would have caught
    vision shipping inert — image_unavailable read 900 on that run and was
    noticed only by somebody reading the numbers by hand.
    """

    def test_a_clean_run_reports_nothing(self) -> None:
        assert sl.faults({"image_sent": 900, "probe": 5}, actions=1000) == []

    @pytest.mark.parametrize("counter", sl.FAULT_COUNTERS)
    def test_any_fault_counter_is_reported(self, counter: str) -> None:
        """One occurrence is enough: these are defects, not preferences."""
        assert sl.faults({counter: 1}, actions=1000) == [f"{counter}=1"]

    def test_the_vision_failure_that_went_unnoticed_is_caught(self) -> None:
        found = sl.faults({"image_unavailable": 900}, actions=1000)
        assert found and "image_unavailable=900" in found[0]

    def test_occasional_degradation_is_tolerated(self) -> None:
        """A stray random fallback is normal; the agent is not broken."""
        assert sl.faults({"random_fallback": 10}, actions=1000) == []

    def test_sustained_degradation_is_reported(self) -> None:
        found = sl.faults({"random_fallback": 300}, actions=1000)
        assert found and "30% of actions" in found[0]

    def test_ratios_need_actions_to_divide_by(self) -> None:
        """A run that took no actions must not raise on the division."""
        assert sl.faults({"random_fallback": 5}, actions=0) == []


class TestForwardModelIsAsserted:
    def test_a_forward_model_that_never_predicts_fails_the_run(self) -> None:
        """It observes every transition, so predicting nothing all run means it
        is not being fed — the shape chrome detection failed in for five
        revisions while nothing asked whether it fired."""
        agent = TestFeatureAssertions.Agent
        assert "forward_predicted" in sl.inert_features(
            agent, {"image_sent": 1, "probe": 1, "exploit": 1}
        )

    def test_a_working_forward_model_passes(self) -> None:
        agent = TestFeatureAssertions.Agent
        assert (
            sl.inert_features(
                agent,
                {"image_sent": 1, "probe": 1, "exploit": 1, "forward_predicted": 40},
            )
            == []
        )


class TestSeeding:
    """A run that records a seed must have used it.

    An A/B whose three "seeds" differed only by label produced byte-identical
    results in the arm whose decisions were mostly deterministic, and no
    variance at all to compare the other arm against.
    """

    def test_each_game_gets_its_own_stream(self) -> None:
        """Seeding every game identically made the concurrent games draw the
        same numbers at the same index, so a run contributed one correlated
        draw rather than one per game."""
        import random as _random

        a = _random.Random("7:sk48").random()
        b = _random.Random("7:cn04").random()
        again = _random.Random("7:sk48").random()
        assert a != b, "different games must not share a stream"
        assert a == again, "the same game and seed must still reproduce"


class TestAgentSelection:
    """The harness has to be able to score the agent that actually scores.

    It loaded `MyAgent` unconditionally, so the explorer — 4 games and 5 levels
    on the roster against 0 for the LLM agent — could only be measured through
    a Kaggle kernel build. With the public score no longer tracking local
    results, this harness is the instrument, so it must reach both.
    """

    def test_both_agents_are_selectable(self) -> None:
        assert set(sl.AGENT_CLASSES) == {"myagent", "explorer"}

    def test_the_explorer_loads_from_the_working_tree(self) -> None:
        assert sl.load_agent_class(None, "explorer").__name__ == "ExplorerAgent"

    def test_the_llm_agent_stays_the_default(self) -> None:
        assert sl.load_agent_class(None).__name__ == "MyAgent"

    def test_an_unknown_agent_is_rejected(self) -> None:
        with pytest.raises(SystemExit, match="Unknown --agent"):
            sl.load_agent_class(None, "nope")

    def test_model_features_are_not_expected_of_a_model_free_agent(self) -> None:
        """The explorer inherits SEND_IMAGE and friends but never calls a model,
        so demanding they fire fails every healthy run."""

        class Explorer:
            USES_MODEL = False
            SEND_IMAGE = True
            PROBE_ACTIONS = True
            REPL_STEPS = 0
            EXPLOIT_REPEATS = 8

        assert sl.inert_features(Explorer, {}) == []

    def test_the_llm_agent_is_still_checked(self) -> None:
        assert "image_sent" in sl.inert_features(
            TestFeatureAssertions.Agent,
            {"probe": 1, "exploit": 1, "forward_predicted": 1},
        )


class TestLevelBreakdown:
    """The per-level ratio is the number the metric actually rewards."""

    @staticmethod
    def run(score: float, actions: int) -> SimpleNamespace:
        return SimpleNamespace(
            score=score,
            level_scores=[115.0, 0.0],
            level_actions=[actions, 900],
            level_baseline_actions=[17, 40],
        )

    @classmethod
    def card(cls, *runs: SimpleNamespace) -> SimpleNamespace:
        runs = runs or (cls.run(2.778, 10),)
        env = SimpleNamespace(
            id="lp85-abc123", score=max(r.score for r in runs), runs=list(runs)
        )
        return SimpleNamespace(environments=[env])

    def test_only_cleared_levels_are_reported_with_their_baseline_ratio(self) -> None:
        """Level 2 scored zero, and listing it would bury the levels whose
        action count can still be improved."""
        (row,) = sl.level_breakdown(self.card())
        assert (row["game"], row["level"], row["baseline"], row["actions"]) == (
            "lp85",
            1,
            17,
            10,
        )

    def test_the_best_play_of_a_game_is_the_one_reported(self) -> None:
        """The scorecard scores a game as max() over its plays, so a weaker
        play must not be the one whose ratios get optimised."""
        (row,) = sl.level_breakdown(self.card(self.run(0.1, 100), self.run(2.778, 10)))
        assert row["actions"] == 10

    @pytest.mark.parametrize("card", [SimpleNamespace(environments=[]), object()])
    def test_a_run_that_cleared_nothing_is_not_an_error(self, card: object) -> None:
        assert sl.level_breakdown(card) == []

    def test_every_rendered_column_has_its_own_header(self) -> None:
        """The headers were `(..., "score", "game")` over a level score and a
        game score, so the one table this reporting exists to print was
        unreadable while the suite stayed green."""
        rendered: list[str] = []
        with mock.patch.object(sl.console, "print", rendered.append):
            sl.render_levels(sl.level_breakdown(self.card()))
        table = next(r for r in rendered if isinstance(r, Table))
        headers = [str(col.header) for col in table.columns]
        assert len(set(headers)) == len(headers) == 7


class TestLevelMetrics:
    """W&B is where runs get compared, so the real objective has to reach it."""

    BREAKDOWN = [
        {
            "game": "lp85",
            "level": 1,
            "baseline": 17,
            "actions": 10,
            "level_score": 115.0,
        },
        {
            "game": "tu93",
            "level": 3,
            "baseline": 34,
            "actions": 886,
            "level_score": 0.15,
        },
    ]

    def test_the_ratio_spread_is_reported_as_scalars(self) -> None:
        metrics = sl.level_metrics(self.BREAKDOWN)
        assert (metrics["level_ratio/best"], metrics["level_ratio/worst"]) == (
            0.59,
            26.06,
        )

    def test_each_level_is_addressable_on_its_own(self) -> None:
        assert sl.level_metrics(self.BREAKDOWN)["level/tu93/3/ratio"] == 26.06

    def test_a_run_that_cleared_nothing_logs_no_ratios(self) -> None:
        """An empty dict still lets the caller splat it into tracker.log."""
        assert sl.level_metrics([]) == {}
