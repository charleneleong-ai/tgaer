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


def _grid(fill: int) -> np.ndarray:
    return np.full((2, 2), fill, dtype=np.int16)


def _t(
    action: int,
    *,
    changed: bool = True,
    level: int = 0,
    after: int | None = None,
    click: tuple[int, int] | None = None,
    grid: np.ndarray | None = None,
    next_grid: np.ndarray | None = None,
) -> cg.Transition:
    base = GRID if grid is None else grid
    return cg.Transition(
        grid=base,
        action=action,
        click=click,
        next_grid=(OTHER if changed else base) if next_grid is None else next_grid,
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


class TestOutcomes:
    """`cyclic` is the outcome change-detection cannot see, and the reason this
    class exists: lp85 spent 542 actions changing the board and going nowhere."""

    def test_an_option_that_only_revisits_old_states_is_cyclic_not_novel(self) -> None:
        a, b = _grid(1), _grid(2)
        ev = _evidence(
            [
                _t(6, click=(1, 1), grid=a, next_grid=b),  # first sighting of b
                _t(6, click=(1, 1), grid=b, next_grid=a),  # back to a — a loop
                _t(6, click=(1, 1), grid=a, next_grid=b),
            ]
        )
        row = ev.outcomes()[("click", 1, 1)]
        assert row["novel"] == 1 and row["cyclic"] == 2

    def test_a_cycling_option_counts_as_unproductive_despite_always_changing(
        self,
    ) -> None:
        a, b = _grid(1), _grid(2)
        ev = _evidence(
            [_t(6, click=(1, 1), grid=a, next_grid=b)]
            + [_t(6, click=(1, 1), grid=b, next_grid=a) for _ in range(3)]
        )
        # it changed the board every single time, and still goes nowhere
        assert all(t.changed for t in ev.transitions)
        assert ("click", 1, 1) in ev.unproductive_options()

    def test_an_option_reaching_new_states_is_progressive(self) -> None:
        ev = _evidence(
            [_t(6, click=(2, 2), grid=_grid(i), next_grid=_grid(i + 1)) for i in range(4)]
        )
        assert ("click", 2, 2) in ev.progressive_options()

    def test_clearing_a_level_is_progressive_whatever_the_board_did(self) -> None:
        ev = _evidence([_t(6, click=(3, 3), level=0, after=1)])
        assert ("click", 3, 3) in ev.progressive_options()

    def test_an_option_that_never_changes_anything_is_unproductive(self) -> None:
        ev = _evidence([_t(6, changed=False, click=(18, 20)) for _ in range(4)])
        assert ("click", 18, 20) in ev.unproductive_options()

    def test_one_sighting_is_too_few_to_condemn_an_option(self) -> None:
        assert _evidence([_t(6, changed=False, click=(1, 1))]).unproductive_options() == set()


class TestChromeMask:
    def test_a_cell_animating_every_step_is_masked_out_of_state_identity(self) -> None:
        """Without this lp85 reads 95% novel while looping on a single cell."""
        rows = []
        for i in range(30):
            before, after = _grid(0).copy(), _grid(0).copy()
            before[0, 0], after[0, 0] = i % 3, (i + 1) % 3  # a ticking counter
            rows.append(_t(6, click=(1, 1), grid=before, next_grid=after))
        ev = _evidence(rows)
        mask = ev.chrome_mask()
        assert mask is not None and bool(mask[0, 0])
        # with the animation masked, those two boards are the same state
        a, b = rows[0].grid, rows[0].next_grid
        assert ev.settled_key(a, mask) == ev.settled_key(b, mask)

    def test_too_little_evidence_masks_nothing(self) -> None:
        assert _evidence([_t(6) for _ in range(3)]).chrome_mask() is None

    def test_a_quiet_board_masks_nothing(self) -> None:
        rows = [_t(6, changed=False, click=(1, 1)) for _ in range(30)]
        assert _evidence(rows).chrome_mask() is None


class TestProductivity:
    """Judged on progress, never on reproducing the explorer."""

    def _held(self, n: int = 8) -> list[cg.Transition]:
        return [_t(1, click=(9, 9)) for _ in range(n)]

    def _progressive(self) -> cg.GameEvidence:
        return _evidence(
            [_t(6, click=(5, 5), grid=_grid(i), next_grid=_grid(i + 1)) for i in range(6)]
        )

    def _cycling(self) -> cg.GameEvidence:
        a, b = _grid(1), _grid(2)
        return _evidence(
            [_t(6, click=(18, 20), grid=a, next_grid=b)]
            + [_t(6, click=(18, 20), grid=b, next_grid=a) for _ in range(5)]
        )

    def test_a_policy_choosing_a_progressive_option_scores_one(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 5, 5)\n")
        assert cg.productivity(policy, self._held(), self._progressive(), [6]).score == 1.0

    def test_a_policy_choosing_a_cycling_option_scores_zero(self) -> None:
        """The headline fix: this option changes the board every time."""
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 18, 20)\n")
        result = cg.productivity(policy, self._held(), self._cycling(), [6])
        assert result.judged == 8 and result.score == 0.0

    def test_choices_with_no_evidence_are_skipped_not_guessed(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 40, 40)\n")
        result = cg.productivity(policy, self._held(), self._progressive(), [6])
        assert result.committed == 8 and result.judged == 0

    def test_proposing_an_unavailable_action_is_always_wrong(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return 99\n")
        result = cg.productivity(policy, self._held(), self._progressive(), [6])
        assert result.judged == 8 and result.score == 0.0

    def test_a_malformed_answer_counts_against_the_policy(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return 'left'\n")
        result = cg.productivity(policy, self._held(), self._progressive(), [6])
        assert result.judged == 8 and result.score == 0.0

    def test_deferring_is_recorded_as_no_commitment(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return None\n")
        result = cg.productivity(policy, self._held(), self._progressive(), [6])
        assert result.committed == 0 and result.commit_rate == 0.0

    def test_a_throwing_policy_yields_nothing_rather_than_propagating(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    raise ValueError('x')\n")
        result = cg.productivity(policy, self._held(), self._progressive(), [6])
        assert result.judged == 0 and result.score == 0.0

    def test_the_policy_is_not_shown_the_answer(self) -> None:
        """It receives the real action list, not the explorer's chosen action."""
        seen: list[list[int]] = []

        def policy(grid: Any, available: list[int], memory: dict[str, Any]) -> Any:
            seen.append(list(available))
            return None

        cg.productivity(policy, self._held(2), self._progressive(), [1, 2, 6])
        assert seen == [[1, 2, 6], [1, 2, 6]]


class TestValidate:
    """Three bars, each catching a different way a policy can be useless."""

    LIVE = [
        _t(6, click=(5, 5), grid=_grid(i), next_grid=_grid(i + 1)) for i in range(20)
    ]

    def test_a_constant_policy_is_rejected_however_good_its_one_choice(self) -> None:
        """The failure both earlier bars missed: one known-good option replayed
        on every state scores a perfect 1.00 and is a loop by construction."""
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 5, 5)\n")
        usable, reason = cg.validate(policy, _evidence(self.LIVE))
        assert not usable and "ignores the board" in reason

    def test_a_board_sensitive_policy_is_usable(self) -> None:
        policy = cg.compile_policy(
            "def policy(g, a, m):\n    return ('click', 5, int(g[0, 0]) % 2 + 5)\n"
        )
        usable, reason = cg.validate(policy, _evidence(self.LIVE))
        assert usable and "productivity 1.00" in reason

    def test_a_policy_that_defers_on_everything_is_rejected(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return None\n")
        usable, reason = cg.validate(policy, _evidence(self.LIVE))
        assert not usable and "defers too often" in reason

    def test_a_policy_choosing_untried_cells_is_allowed_through(self) -> None:
        """Trying an untouched cell is what a good policy on an unsolved level
        does. Requiring the policy to stay inside the explorer's experience
        while asking it to beat the explorer is incoherent, and rejected five
        games for it. Unjudgeable choices go to the gate instead."""
        policy = cg.compile_policy(
            "def policy(g, a, m):\n    return ('click', 63, int(g[0, 0]) % 2 + 60)\n"
        )
        usable, reason = cg.validate(policy, _evidence(self.LIVE))
        assert usable and "0 judged" in reason

    def test_a_policy_mixing_good_and_disproved_choices_is_still_rejected(self) -> None:
        """Loosening the evidence bar must not let a disproved option back in."""
        # interleaved so the held-out half carries *both* kinds of board and
        # the policy actually reaches its bad branch
        rows: list[cg.Transition] = []
        for i in range(40):
            if i % 2:
                rows.append(_t(6, click=(18, 20), grid=_grid(i), next_grid=_grid(i)))
            else:
                rows.append(_t(6, click=(5, 5), grid=_grid(i), next_grid=_grid(i + 1)))
        policy = cg.compile_policy(
            "def policy(g, a, m):\n"
            "    return ('click', 18, 20) if int(g[0, 0]) % 2 else ('click', 5, 5)\n"
        )
        usable, reason = cg.validate(policy, _evidence(rows))
        assert not usable and "disproved" in reason

    def test_a_policy_picking_cycling_cells_is_rejected(self) -> None:
        evidence = _evidence(
            self.LIVE + [_t(6, changed=False, click=(18, 20)) for _ in range(20)]
        )
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 18, 20)\n")
        usable, reason = cg.validate(policy, evidence)
        assert not usable and "disproved" in reason

    def test_no_evidence_means_not_usable(self) -> None:
        policy = cg.compile_policy("def policy(g, a, m):\n    return a[0]\n")
        usable, reason = cg.validate(policy, _evidence([]))
        assert not usable and "no warmup" in reason


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


def test_as_json_is_loadable() -> None:
    import json

    assert json.loads(cg.as_json(_evidence([_t(1)])))["steps"] == 1


class TestRefinePolicy:
    """The REPL loop: ask, replay, critique, ask again — all offline."""

    class _Scripted:
        """A backend that returns a fixed sequence and records what it was told."""

        def __init__(self, replies: list[str]) -> None:
            self.replies = replies
            self.seen: list[list[dict[str, str]]] = []

        def chat(self, messages: list[dict[str, str]], max_tokens: int = 2048) -> str:
            self.seen.append([dict(m) for m in messages])
            return self.replies[min(len(self.seen) - 1, len(self.replies) - 1)]

    GOOD = (
        "```python\ndef policy(g, a, m):\n"
        "    return ('click', 5, int(g[0, 0]) % 2 + 5)\n```"
    )
    CONSTANT = "```python\ndef policy(g, a, m):\n    return ('click', 5, 5)\n```"
    BROKEN = "```python\ndef policy(:\n```"

    def _evidence(self) -> cg.GameEvidence:
        return _evidence(
            [
                _t(6, click=(5, 5 + i % 2), grid=_grid(i), next_grid=_grid(i + 1))
                for i in range(20)
            ]
        )

    def test_a_good_first_answer_stops_the_loop(self) -> None:
        backend = self._Scripted([self.GOOD])
        policy, reason = cg.refine_policy(backend, self._evidence(), rounds=4)
        assert policy is not None and "round 1" in reason
        assert len(backend.seen) == 1  # no wasted rounds

    def test_a_rejected_answer_is_retried_with_the_reason(self) -> None:
        backend = self._Scripted([self.CONSTANT, self.GOOD])
        policy, reason = cg.refine_policy(backend, self._evidence(), rounds=4)
        assert policy is not None and "round 2" in reason
        followup = backend.seen[1][-1]["content"]
        assert "rejected" in followup and "same answer on every board" in followup

    def test_a_compile_error_is_handed_back_verbatim(self) -> None:
        backend = self._Scripted([self.BROKEN, self.GOOD])
        policy, _ = cg.refine_policy(backend, self._evidence(), rounds=4)
        assert policy is not None
        assert "SyntaxError" in backend.seen[1][-1]["content"]

    def test_it_gives_up_after_the_round_limit(self) -> None:
        backend = self._Scripted([self.CONSTANT])
        policy, reason = cg.refine_policy(backend, self._evidence(), rounds=3)
        assert policy is None and "after 3 rounds" in reason
        assert len(backend.seen) == 3

    def test_the_conversation_accumulates_rather_than_restarting(self) -> None:
        backend = self._Scripted([self.CONSTANT])
        cg.refine_policy(backend, self._evidence(), rounds=3)
        assert [len(m) for m in backend.seen] == [2, 4, 6]

    def test_a_dead_backend_stops_immediately(self) -> None:
        class Dead:
            def chat(self, messages: list[dict[str, str]], max_tokens: int = 2048) -> str:
                raise ConnectionError("refused")

        policy, reason = cg.refine_policy(Dead(), self._evidence(), rounds=4)
        assert policy is None and "backend failed on round 1" in reason

    def test_empty_content_names_the_reasoning_trap(self) -> None:
        policy, reason = cg.refine_policy(self._Scripted(["  "]), self._evidence())
        assert policy is None and "thinking" in reason


class TestCritique:
    def test_it_cites_a_concrete_board_and_the_option_record(self) -> None:
        """Aggregates say a policy failed; an instance says how."""
        a, b = _grid(1), _grid(2)
        evidence = _evidence(
            [_t(6, click=(18, 20), grid=a, next_grid=b)]
            + [_t(6, click=(18, 20), grid=b, next_grid=a) for _ in range(9)]
        )
        policy = cg.compile_policy("def policy(g, a, m):\n    return ('click', 18, 20)\n")
        text = cg.critique(policy, evidence, "picks known-dead options")
        assert "held-out board" in text and "repeats" in text

    def test_a_throwing_policy_is_reported_with_its_exception(self) -> None:
        evidence = _evidence([_t(6, click=(1, 1)) for _ in range(6)])
        policy = cg.compile_policy("def policy(g, a, m):\n    raise KeyError('nope')\n")
        assert "KeyError" in cg.critique(policy, evidence, "threw")


class TestReviewRegressions:
    """One test per defect found reviewing this module. Each of these shipped."""

    def _live(self, n: int = 20) -> cg.GameEvidence:
        return _evidence(
            [_t(6, click=(5, 5), grid=_grid(i), next_grid=_grid(i + 1)) for i in range(n)]
        )

    @pytest.mark.parametrize("answer", ["[1, 2]", "{'a': 1}", "np.array([1, 2])"])
    def test_an_unhashable_answer_is_reported_not_raised(self, answer: str) -> None:
        """`("act", [1,2]) in some_set` raises TypeError, and this runs inside
        the game loop where `play` does not catch — one malformed answer would
        abort the whole suite instead of falling back to the explorer."""
        policy = cg.compile_policy(f"def policy(g, a, m):\n    return {answer}\n")
        text = cg.critique(policy, self._live(), "test")
        assert "not an action or a click" in text

    def test_numpy_integers_count_as_actions(self) -> None:
        """The prompt tells the model to use numpy, so it returns np.int64 —
        which is not an `int`. Both bars misfired: every answer scored malformed
        and they all collapsed to one bucket, so a board-sensitive policy was
        rejected as a constant."""
        assert cg.as_action_id(np.int64(6)) == 6
        evidence = _evidence(
            [_t(6, grid=_grid(i), next_grid=_grid(i + 1)) for i in range(20)]
        )
        policy = cg.compile_policy(
            "def policy(g, a, m):\n    return np.int64(a[int(g[0, 0]) % len(a)])\n"
        )
        result = cg.productivity(policy, cg.held_out_of(evidence), evidence, [6])
        assert result.judged > 0 and result.score == 1.0

    def test_booleans_are_not_action_one(self) -> None:
        assert cg.as_action_id(True) is None

    def test_a_death_respawn_is_not_counted_as_progress(self) -> None:
        """A respawn lands on an unseen board, so without the terminal flag a
        lethal click reads as `novel` and gets promoted."""
        rows = [
            cg.Transition(
                grid=_grid(1), action=6, click=(9, 9), next_grid=_grid(99),
                level=0, level_after=0, terminal=True,
            )
            for _ in range(4)
        ]
        evidence = _evidence(rows)
        assert ("click", 9, 9) not in evidence.progressive_options()
        assert ("click", 9, 9) in evidence.unproductive_options()

    def test_memory_persists_across_a_validation_pass(self) -> None:
        """The prompt promises `memory` persists; a policy relying on it was
        being handed a fresh dict per call and judged a constant."""
        evidence = self._live()
        policy = cg.compile_policy(
            "def policy(g, a, m):\n"
            "    m['n'] = m.get('n', 0) + 1\n"
            "    return ('click', 5, 5 + m['n'] % 2)\n"
        )
        assert cg.distinct_choices(policy, cg.held_out_of(evidence), evidence) == 2
