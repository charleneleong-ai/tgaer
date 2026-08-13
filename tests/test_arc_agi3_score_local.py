"""Tests for the local scoring harness.

The harness exists to make agent comparisons trustworthy, so its two pieces of
real logic — the llama-cpp overflow rule and the ollama→OpenAI response shim —
need to be right, or every measurement taken with it is suspect.
"""

from __future__ import annotations

from typing import Any

import pytest

from tgaer.evaluation import arc_agi3_score_local as sl


def backend(n_ctx: int = 4096) -> sl.OllamaBackend:
    return sl.OllamaBackend(model="test", n_ctx=n_ctx, host="http://localhost:11434")


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


def stub_post(
    monkeypatch: pytest.MonkeyPatch, be: sl.OllamaBackend, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any]) -> FakeResponse:  # noqa: A002
        sent.append(json)
        return FakeResponse(payload)

    monkeypatch.setattr(be.client, "post", fake_post)
    return sent


class TestOverflowRule:
    """Mirrors llama_cpp/llama.py:1336 — `>= n_ctx` raises, below it passes."""

    @pytest.mark.parametrize(
        ("used", "raises"), [(4095, False), (4096, True), (9000, True)]
    )
    def test_boundary(
        self, monkeypatch: pytest.MonkeyPatch, used: int, raises: bool
    ) -> None:
        be = backend(4096)
        stub_post(
            monkeypatch, be, {"prompt_eval_count": used, "message": {"content": "hi"}}
        )
        if raises:
            with pytest.raises(sl.ContextOverflow):
                be.create_chat_completion([{"role": "user", "content": "x"}])
        else:
            be.create_chat_completion([{"role": "user", "content": "x"}])
        assert be.prompt_tokens == [used]
        assert be.overflows == int(raises)
        assert be.calls == 1

    def test_missing_token_count_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defaulting to 0 here would silently report 'no overflows' forever."""
        be = backend()
        stub_post(monkeypatch, be, {"message": {"content": "hi"}})
        with pytest.raises(KeyError):
            be.create_chat_completion([{"role": "user", "content": "x"}])

    def test_runs_with_headroom_so_counts_are_untruncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ollama truncates to num_ctx; asking for exactly n_ctx would hide overflows."""
        be = backend(4096)
        sent = stub_post(
            monkeypatch, be, {"prompt_eval_count": 10, "message": {"content": "a"}}
        )
        be.create_chat_completion([{"role": "user", "content": "x"}])
        assert sent[0]["options"]["num_ctx"] > be.n_ctx


class TestResponseShim:
    """The agent parses llama-cpp's shape; ollama's differs in two ways."""

    def test_tool_arguments_become_a_json_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ollama returns a dict; the agent calls json.loads on it."""
        be = backend()
        stub_post(
            monkeypatch,
            be,
            {
                "prompt_eval_count": 10,
                "message": {
                    "tool_calls": [
                        {"function": {"name": "MOUSE", "arguments": {"x": 3, "y": 4}}}
                    ]
                },
            },
        )
        out = be.create_chat_completion([{"role": "user", "content": "x"}])
        call = out["choices"][0]["message"]["tool_calls"][0]
        assert isinstance(call["function"]["arguments"], str)
        import json

        assert json.loads(call["function"]["arguments"]) == {"x": 3, "y": 4}

    def test_plain_content_has_no_tool_calls_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        be = backend()
        stub_post(
            monkeypatch,
            be,
            {"prompt_eval_count": 10, "message": {"content": "action(['UP'])"}},
        )
        message = be.create_chat_completion([{"role": "user", "content": "x"}])[
            "choices"
        ][0]["message"]
        assert message["content"] == "action(['UP'])"
        assert "tool_calls" not in message

    def test_thinking_is_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """qwen3 reasons by default, which would blow past max_tokens every turn."""
        be = backend()
        sent = stub_post(
            monkeypatch, be, {"prompt_eval_count": 10, "message": {"content": "a"}}
        )
        be.create_chat_completion([{"role": "user", "content": "x"}])
        assert sent[0]["think"] is False


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

    def test_a_backend_with_token_stats_still_reports_them(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The guard must not silently swallow the overflow count it exists to show."""
        be = backend(4096)
        be.prompt_tokens = [10, 9000]
        sl.render(self.ROWS, "test", 0.0, 0, backend=be)
        assert "context overflows" in capsys.readouterr().out


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

    @pytest.mark.parametrize(("seed", "present"), [(None, False), (0, True), (7, True)])
    def test_ollama_forwards_the_seed(
        self, monkeypatch: pytest.MonkeyPatch, seed: int | None, present: bool
    ) -> None:
        """0 is falsy and valid: `if self.seed:` would drop it silently."""
        be = sl.OllamaBackend(model="test", n_ctx=4096, host="http://x", seed=seed)
        sent = stub_post(
            monkeypatch, be, {"prompt_eval_count": 10, "message": {"content": "a"}}
        )
        be.create_chat_completion([{"role": "user", "content": "x"}])
        assert ("seed" in sent[0]["options"]) is present
        if present:
            assert sent[0]["options"]["seed"] == seed

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
