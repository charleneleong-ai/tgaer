"""Tests for the replay effect model.

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


re_ = _load("replay_effects")


def _grid(
    avatar_rc: tuple[int, int] | None = None,
    extra: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    g = np.full((10, 10), 3, dtype=np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    if avatar_rc is not None:
        g[avatar_rc] = 12
    for rc, v in (extra or {}).items():
        g[rc] = v
    return g


class TestBucket:
    @pytest.mark.parametrize(
        ("n", "want"), [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (7, 3), (8, 4)]
    )
    def test_log2_buckets_with_zero_reserved(self, n: int, want: int) -> None:
        assert re_.bucket(n) == want


class TestEffectSignature:
    def test_identical_frames_have_zero_churn(self) -> None:
        g = _grid((2, 2))
        assert re_.effect_signature(g, g.copy(), False, 12)[4] == 0

    def test_avatar_translation_is_recorded(self) -> None:
        assert re_.effect_signature(_grid((2, 2)), _grid((3, 2)), False, 12)[1] == (
            1,
            0,
        )

    def test_avatar_delta_is_none_on_a_click_game(self) -> None:
        # lp85/m0r0/s5i5/g50t never pin an avatar; this must not crash.
        eff = re_.effect_signature(_grid(None, {(2, 2): 5}), _grid(None), False, None)
        assert eff[1] is None
        assert eff[3] == frozenset({5})

    def test_appeared_and_vanished_are_symmetric(self) -> None:
        eff = re_.effect_signature(
            _grid((2, 2), {(5, 5): 7}), _grid((2, 2), {(5, 5): 9}), False, 12
        )
        assert eff[2] == frozenset({9})
        assert eff[3] == frozenset({7})

    def test_level_advance_is_carried_through(self) -> None:
        g = _grid((2, 2))
        assert re_.effect_signature(g, g.copy(), True, 12)[0] is True

    def test_the_signature_is_hashable(self) -> None:
        g = _grid((2, 2))
        assert len({re_.effect_signature(g, g.copy(), False, 12)}) == 1

    def test_a_real_in_field_change_registers_as_churn(self) -> None:
        # The caller passes _settled frames, so chrome is already masked to a
        # constant; what must hold here is that a genuine change is not lost.
        # The load-bearing check that settled frames are actually passed lives in
        # TestHarnessGlue, because this function cannot enforce its own caller.
        a = _grid((2, 2))
        assert re_.effect_signature(a, _grid((2, 3)), False, 12)[4] > 0


class TestEffectModel:
    @staticmethod
    def _eff(churn: int) -> Any:
        return (False, None, frozenset(), frozenset(), churn)

    def test_an_empty_model_predicts_nothing(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        assert m.predict(("sig", 6, 5, 2)) == (None, -1)

    def test_it_predicts_the_mode_at_the_specific_key(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        for _ in range(3):
            m.observe(("s1", 6, 5, 2), self._eff(4))
        m.observe(("s1", 6, 5, 2), self._eff(9))
        got, level = m.predict(("s2", 6, 5, 2))
        assert got == self._eff(4)
        assert level == 0

    def test_a_rule_learned_at_one_state_fires_at_another(self) -> None:
        # This is the whole point: cross-state generalisation.
        m = re_.EffectModel(re_.class_keys)
        m.observe(("s1", 6, 5, 2), self._eff(4))
        assert m.predict(("s2", 6, 5, 2))[0] == self._eff(4)

    def test_it_backs_off_to_colour_then_action(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        m.observe(("s1", 6, 5, 2), self._eff(4))
        assert m.predict(("s1", 6, 5, 9))[1] == 1
        assert m.predict(("s1", 6, 77, 9))[1] == 2

    def test_an_unresolvable_click_target_degrades_to_the_action(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        m.observe(("s1", 6, 5, 2), self._eff(4))
        got, level = m.predict(("s1", 6, None, None))
        assert got == self._eff(4)
        assert level == 0  # (action,) is the only key this ctx has

    def test_the_state_keyed_arm_does_not_generalise_across_states(self) -> None:
        m = re_.EffectModel(re_.state_keys)
        m.observe(("s1", 6, 5, 2), self._eff(4))
        assert m.predict(("s2", 6, 5, 2)) == (None, -1)

    def test_the_marginal_arm_ignores_the_key_entirely(self) -> None:
        m = re_.EffectModel(re_.marginal_keys)
        m.observe(("s1", 1, None, None), self._eff(4))
        assert m.predict(("s9", 6, 77, 3))[0] == self._eff(4)

    def test_seen_reports_whether_the_specific_key_is_known(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        assert not m.seen(("s1", 6, 5, 2))
        m.observe(("s1", 6, 5, 2), self._eff(4))
        assert m.seen(("s1", 6, 5, 2))


class TestPrequential:
    @staticmethod
    def _eff(churn: int) -> Any:
        return (False, None, frozenset(), frozenset(), churn)

    def test_the_first_transition_is_a_miss_not_a_crash(self) -> None:
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))
        assert p.n == 1
        assert p.report()["class"] == 0.0

    def test_a_deterministic_rule_is_learned_after_first_sighting(self) -> None:
        p = re_.Prequential()
        for _ in range(11):
            p.step(("s1", 6, 5, 2), self._eff(4))
        assert p.report()["class"] == pytest.approx(10 / 11)

    def test_an_inconsistent_rule_does_not_beat_its_own_purity(self) -> None:
        p = re_.Prequential()
        for i in range(100):
            p.step(("s1", 6, 5, 2), self._eff(4 if i % 4 else 9))
        assert p.report()["class"] <= 0.76

    def test_class_beats_state_when_the_rule_generalises(self) -> None:
        p = re_.Prequential()
        for i in range(20):
            p.step((f"s{i}", 6, 5, 2), self._eff(4))
        r = p.report()
        assert r["class"] > r["state"]
        assert r["state"] == 0.0

    def test_a_single_effect_game_gives_marginal_parity_and_no_zero_division(
        self,
    ) -> None:
        p = re_.Prequential()
        for i in range(10):
            p.step((f"s{i}", 1, None, None), self._eff(0))
        r = p.report()
        assert r["marginal"] == pytest.approx(r["class"])

    def test_the_inert_arm_only_judges_whether_anything_changed(self) -> None:
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))
        p.step(("s1", 6, 5, 2), self._eff(9))
        r = p.report()
        assert r["inert"] > r["class"]

    def test_first_sighting_accuracy_is_reported_separately(self) -> None:
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))
        p.step(("s1", 6, 5, 2), self._eff(4))
        assert 0.0 <= p.report()["first_sighting"] <= 1.0

    def test_backoff_levels_are_reported_as_shares(self) -> None:
        p = re_.Prequential()
        for _ in range(4):
            p.step(("s1", 6, 5, 2), self._eff(4))
        r = p.report()
        assert sum(r[k] for k in ("backoff_0", "backoff_1", "backoff_2")) <= 1.0
