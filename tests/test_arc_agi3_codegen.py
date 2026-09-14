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


def _t(
    action: int,
    *,
    changed: bool = True,
    level: int = 0,
    after: int | None = None,
    click: tuple[int, int] | None = None,
) -> cg.Transition:
    return cg.Transition(
        grid=GRID,
        action=action,
        click=click,
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


class TestAsClick:
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            (("click", 3, 4), (3, 4)),
            (["click", 3, 4], (3, 4)),
            (6, None),
            (("click", 3), None),
            (("move", 3, 4), None),
            (("click", "x", 4), None),
        ],
    )
    def test_recognises_only_well_formed_clicks(self, choice: Any, expected: Any) -> None:
        assert cg.as_click(choice) == expected


class TestDeadEvidence:
    def test_a_cell_clicked_repeatedly_to_no_effect_is_dead(self) -> None:
        """The lp85 failure, made checkable: 542 clicks on one unchanging cell."""
        ev = _evidence([_t(6, changed=False, click=(18, 20)) for _ in range(5)])
        assert ev.dead_clicks() == {(18, 20)}

    def test_one_no_op_is_not_enough_to_call_a_cell_dead(self) -> None:
        assert _evidence([_t(6, changed=False, click=(1, 1))]).dead_clicks() == set()

    def test_a_cell_that_ever_worked_is_never_dead(self) -> None:
        ev = _evidence(
            [_t(6, changed=False, click=(1, 1)), _t(6, changed=False, click=(1, 1)),
             _t(6, changed=True, click=(1, 1))]
        )
        assert ev.dead_clicks() == set() and ev.live_clicks() == {(1, 1)}

    def test_dead_and_live_actions_split_on_whether_anything_changed(self) -> None:
        ev = _evidence([_t(1, changed=False), _t(1, changed=False), _t(2, changed=True)])
        assert ev.dead_actions() == {1} and ev.live_actions() == {2}


class TestProductivity:
    """Falsification, not imitation — a policy is judged on avoiding what the
    warmup proved useless, never on reproducing the explorer's moves."""

    def _held(self, n: int = 8) -> list[cg.Transition]:
        return [_t(1, click=(9, 9)) for _ in range(n)]

    def test_a_policy_clicking_a_known_dead_cell_scores_zero(self) -> None:
        ev = _evidence(
            [_t(6, changed=False, click=(18, 20)) for _ in range(4)]
            + [_t(6, changed=True, click=(5, 5))]
        )
        policy = cg.compile_policy(
            "def policy(g, a, m):\n    return ('click', 18, 20)\n"
        )
        result = cg.productivity(policy, self._held(), ev, [6])
        assert result.judged == 8 and result.score == 0.0

    def test_a_policy_clicking_a_cell_known_to_work_scores_one(self) -> None:
        ev = _evidence([_t(6, changed=True, click=(5, 5))])
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 5, 5)\n")
        assert cg.productivity(policy, self._held(), ev, [6]).score == 1.0

    def test_choices_with_no_evidence_are_skipped_not_guessed(self) -> None:
        ev = _evidence([_t(6, changed=True, click=(5, 5))])
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 40, 40)\n")
        result = cg.productivity(policy, self._held(), ev, [6])
        assert result.committed == 8 and result.judged == 0

    def test_proposing_an_unavailable_action_is_always_wrong(self) -> None:
        ev = _evidence([_t(1, changed=True)])
        policy = cg.compile_policy("def policy(g, a, m):\n    return 99\n")
        result = cg.productivity(policy, self._held(), ev, [1])
        assert result.judged == 8 and result.score == 0.0

    def test_deferring_is_recorded_as_no_commitment(self) -> None:
        ev = _evidence([_t(1, changed=True)])
        policy = cg.compile_policy("def policy(g, a, m):\n    return None\n")
        result = cg.productivity(policy, self._held(), ev, [1])
        assert result.committed == 0 and result.commit_rate == 0.0

    def test_a_throwing_policy_yields_nothing_rather_than_propagating(self) -> None:
        ev = _evidence([_t(1, changed=True)])
        policy = cg.compile_policy("def policy(g, a, m):\n    raise ValueError('x')\n")
        result = cg.productivity(policy, self._held(), ev, [1])
        assert result.judged == 0 and result.score == 0.0

    def test_the_policy_is_not_shown_the_answer(self) -> None:
        """It receives the real action list, not the action the explorer took —
        the earlier version passed `[t.action]`, which leaked the answer."""
        seen: list[list[int]] = []

        def policy(grid: Any, available: list[int], memory: dict[str, Any]) -> Any:
            seen.append(list(available))
            return None

        cg.productivity(policy, self._held(2), _evidence([_t(1)]), [1, 2, 6])
        assert seen == [[1, 2, 6], [1, 2, 6]]


class TestValidate:
    """Three bars, each catching a different way a policy can be useless."""

    LIVE = [_t(6, changed=True, click=(5, 5)) for _ in range(20)]

    def test_a_productive_policy_is_usable(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 5, 5)\n")
        usable, reason = cg.validate(policy, _evidence(self.LIVE))
        assert usable and "productivity 1.00" in reason

    def test_a_policy_that_defers_on_everything_is_rejected(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return None\n")
        usable, reason = cg.validate(policy, _evidence(self.LIVE))
        assert not usable and "defers too often" in reason

    def test_a_policy_we_have_no_evidence_about_is_rejected(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 63, 63)\n")
        usable, reason = cg.validate(policy, _evidence(self.LIVE))
        assert not usable and "too little evidence" in reason

    def test_a_policy_picking_dead_cells_is_rejected(self) -> None:
        evidence = _evidence(
            self.LIVE + [_t(6, changed=False, click=(18, 20)) for _ in range(20)]
        )
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 18, 20)\n")
        usable, reason = cg.validate(policy, evidence)
        assert not usable and "known-dead" in reason

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

    def test_empty_content_names_the_reasoning_trap_specifically(self) -> None:
        """A reasoning model burns the budget thinking and returns "" — which
        must not be misread as a bad prompt. Measured on Qwen3.8-27B."""
        policy, reason = cg.request_policy(self._Backend("   "), _evidence([_t(1)]))
        assert policy is None and "thinking" in reason

    def test_prose_only_is_reported(self) -> None:
        policy, reason = cg.request_policy(self._Backend("no code here"), _evidence([_t(1)]))
        assert policy is None and "no python block" in reason

    def test_uncompilable_code_is_reported(self) -> None:
        policy, reason = cg.request_policy(
            self._Backend("```python\ndef policy(:\n```"), _evidence([_t(1)])
        )
        assert policy is None and "did not compile" in reason

    def test_a_valid_policy_comes_back_usable(self) -> None:
        reply = "```python\ndef policy(g, a, m):\n    return ('click', 5, 5)\n```"
        evidence = _evidence([_t(6, changed=True, click=(5, 5)) for _ in range(20)])
        policy, reason = cg.request_policy(self._Backend(reply), evidence)
        assert policy is not None, reason
        assert cg.as_click(policy(GRID, [6], {})) == (5, 5)


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
