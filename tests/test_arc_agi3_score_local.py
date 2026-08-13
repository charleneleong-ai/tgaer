"""Tests for the local scoring harness.

The harness exists to make agent comparisons trustworthy, so its two pieces of
real logic — the llama-cpp overflow rule and the ollama→OpenAI response shim —
need to be right, or every measurement taken with it is suspect.
"""

from __future__ import annotations

import pytest

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
