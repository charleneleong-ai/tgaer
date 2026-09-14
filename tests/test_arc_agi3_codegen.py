"""Tests for the LLM-as-programmer harness.

The whole point of this module is that generated code is untrusted: it may be
prose, may not compile, may throw, may be confidently wrong. What has to hold is
that every one of those cases falls back to the explorer instead of taking the
run down, and that a policy only earns real actions by reproducing transitions
we already recorded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tgaer.agents import arc_agi3_codegen as cg

GRID = np.array([[0, 1], [2, 3]])
OTHER = np.array([[9, 9], [9, 9]])


def _t(action: int, *, changed: bool = True, level: int = 0, after: int | None = None) -> cg.Transition:
    return cg.Transition(
        grid=GRID,
        action=action,
        click=None,
        next_grid=OTHER if changed else GRID,
        level=level,
        level_after=level if after is None else after,
    )


def _evidence(transitions: list[cg.Transition]) -> cg.GameEvidence:
    return cg.GameEvidence(game="lp85", available_actions=[1, 2, 6], transitions=transitions)


class TestExtractCode:
    @pytest.mark.parametrize(
        "reply",
        [
            "here you go\n```python\ndef policy(): pass\n```\ndone",
            "```\ndef policy(): pass\n```",
        ],
    )
    def test_finds_the_block_with_or_without_a_language_tag(self, reply: str) -> None:
        assert cg.extract_code(reply).strip() == "def policy(): pass"

    def test_prose_only_returns_none_rather_than_guessing(self) -> None:
        assert cg.extract_code("I think you should move left a lot.") is None

    def test_takes_the_first_block_when_the_model_writes_several(self) -> None:
        reply = "```python\nfirst = 1\n```\nand\n```python\nsecond = 2\n```"
        assert "first" in cg.extract_code(reply)


class TestCompilePolicy:
    def test_returns_the_callable_for_valid_code(self) -> None:
        assert callable(cg.compile_policy("def policy(g, a, m):\n    return a[0]\n"))

    @pytest.mark.parametrize(
        "code",
        [
            "def policy(:\n",  # syntax error
            "raise RuntimeError('boom')",  # throws at module level
            "policy = 42",  # defined but not callable
            "def other(): pass",  # no policy at all
        ],
    )
    def test_every_kind_of_bad_code_yields_none_not_an_exception(self, code: str) -> None:
        assert cg.compile_policy(code) is None

    def test_numpy_is_available_to_generated_code(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return int(np.max(g))\n")
        assert policy(GRID, [1], {}) == 3


class TestAgreement:
    def test_a_policy_that_reproduces_observed_actions_scores_high(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return a[0]\n")
        assert cg.agreement(policy, [_t(1), _t(2)]) == 1.0

    def test_a_throwing_policy_scores_zero_rather_than_propagating(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    raise ValueError('x')\n")
        assert cg.agreement(policy, [_t(1)]) == 0.0

    def test_an_all_none_policy_earns_nothing(self) -> None:
        """Deferring is allowed but must not be a way to pass validation."""
        policy = cg.compile_policy("def policy(g, a, m):\n    return None\n")
        assert cg.agreement(policy, [_t(1), _t(2)]) == 0.0

    def test_actions_that_changed_nothing_do_not_count_as_agreement(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return a[0]\n")
        assert cg.agreement(policy, [_t(1, changed=False)]) == 0.0

    def test_no_held_out_transitions_scores_zero(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return a[0]\n")
        assert cg.agreement(policy, []) == 0.0


class TestValidate:
    def test_a_good_policy_is_usable_and_says_why(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return a[0]\n")
        usable, reason = cg.validate(policy, _evidence([_t(1), _t(2), _t(1), _t(2)]))
        assert usable and "agreement" in reason

    def test_a_deferring_policy_is_rejected_below_the_threshold(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return None\n")
        usable, reason = cg.validate(policy, _evidence([_t(1), _t(2), _t(1), _t(2)]))
        assert not usable and "agreement 0.00" in reason

    def test_no_evidence_means_not_usable(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return a[0]\n")
        usable, reason = cg.validate(policy, _evidence([]))
        assert not usable and "no warmup" in reason


class TestRequestPolicy:
    """Every backend failure mode has to degrade to the explorer, not raise."""

    class _Backend:
        def __init__(self, reply: str | Exception) -> None:
            self.reply = reply

        def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

    def test_a_dead_backend_is_reported_not_raised(self) -> None:
        policy, reason = cg.request_policy(
            self._Backend(ConnectionError("refused")), _evidence([_t(1)])
        )
        assert policy is None and "backend failed" in reason

    def test_prose_only_is_reported(self) -> None:
        policy, reason = cg.request_policy(self._Backend("no code here"), _evidence([_t(1)]))
        assert policy is None and "no python block" in reason

    def test_uncompilable_code_is_reported(self) -> None:
        policy, reason = cg.request_policy(
            self._Backend("```python\ndef policy(:\n```"), _evidence([_t(1)])
        )
        assert policy is None and "did not compile" in reason

    def test_a_valid_policy_comes_back_usable(self) -> None:
        reply = "```python\ndef policy(g, a, m):\n    return a[0]\n```"
        policy, _ = cg.request_policy(
            self._Backend(reply), _evidence([_t(1), _t(2), _t(1), _t(2)])
        )
        assert policy is not None and policy(GRID, [7], {}) == 7


class TestGameEvidence:
    def test_action_effects_separate_changing_from_winning_actions(self) -> None:
        ev = _evidence([_t(1), _t(1, changed=False), _t(2, level=0, after=1)])
        effects = ev.action_effects()
        assert effects[1] == {"tried": 2, "changed": 1, "advanced": 0}
        assert effects[2]["advanced"] == 1

    def test_summary_reports_effects_without_dumping_the_grid(self) -> None:
        summary = _evidence([_t(1), _t(2)]).summary()
        assert "action 1:" in summary and "grid shape" in summary
        # a raw frame dump would blow the context and bury the signal
        assert len(summary) < 2000

    def test_summary_survives_an_empty_warmup(self) -> None:
        assert "warmup steps observed: 0" in _evidence([]).summary()


def test_as_json_is_loadable(monkeypatch: Any) -> None:
    import json

    assert json.loads(cg.as_json(_evidence([_t(1)])))["steps"] == 1
