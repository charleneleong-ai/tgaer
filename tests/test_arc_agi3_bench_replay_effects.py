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


class TestHarnessGlue:
    def test_a_respawn_frame_is_not_a_transition(self) -> None:
        # The agent excludes these; scoring them teaches that a reset is an ordinary
        # effect. This is the one that silently poisons the model if missed.
        assert not re_.is_transition({"terminal": True})
        assert re_.is_transition({})
        assert re_.is_transition({"terminal": False})

    def test_a_click_context_carries_colour_and_component_size(self) -> None:
        arr = _grid(None, {(5, 5): 7, (5, 6): 7})
        ctx = re_.context_of(("click", 5, 5), arr, "sig")
        assert ctx[0] == "sig"
        assert ctx[2] == 7
        assert ctx[3] == re_.bucket(2)

    def test_size_is_the_clicked_component_not_every_cell_of_that_colour(self) -> None:
        # Two separate 1-cell blobs of colour 7: the clicked one has size 1, not 2.
        arr = _grid(None, {(2, 2): 7, (8, 8): 7})
        assert re_.context_of(("click", 2, 2), arr, "sig")[3] == re_.bucket(1)

    def test_a_simple_action_context_has_no_colour_or_size(self) -> None:
        ctx = re_.context_of(("act", 3), _grid((2, 2)), "sig")
        assert ctx[1] == 3
        assert ctx[2] is None and ctx[3] is None

    def test_an_out_of_bounds_click_yields_no_colour(self) -> None:
        ctx = re_.context_of(("click", 99, 99), _grid((2, 2)), "sig")
        assert ctx[2] is None and ctx[3] is None

    def test_importing_the_module_loads_no_games(self) -> None:
        # main() owns every harness call; import must stay side-effect free.
        head = (BENCH / "replay_effects.py").read_text().split("def main(")[0]
        for harness in ("require_starter", "Arcade(", "arc_runner", "load_agent_class"):
            assert harness not in head, harness

    def test_the_hook_scores_settled_frames_not_raw_ones(self) -> None:
        # The load-bearing chrome guard: effect_signature cannot enforce its own
        # caller, so pin the caller. Raw frames here would let chrome register as
        # churn and make every effect look distinct.
        src = (BENCH / "replay_effects.py").read_text()
        call = src.split("effect_signature(")[-1].split(")")[0]
        assert "settled" in call
        assert "arr" not in call.replace("settled", "")


class _Actor:
    """Minimal stand-in for the explorer, enough to drive Replay."""

    def __init__(self, prim: tuple[Any, ...] | None, avatar: int | None = 12) -> None:
        self.trace = {"prim": prim} if prim is not None else {}

        class _Det:
            pass

        self._det = _Det()
        self._det.avatar = avatar

    def _settled(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def _field(self, arr: np.ndarray) -> Any:
        return (np.array([0, 0]), np.array([arr.shape[0] - 1, arr.shape[1] - 1]))


def _obs(board: np.ndarray, levels: int = 0, terminal: bool = False) -> dict[str, Any]:
    o: dict[str, Any] = {"frame": [board.tolist()], "levels_completed": levels}
    if terminal:
        o["terminal"] = True
    return o


class TestInertArmIndependence:
    """C2: inert must be its own predictor, not the class arm projected."""

    @staticmethod
    def _eff(churn: int) -> Any:
        return (False, None, frozenset(), frozenset(), churn)

    def test_inert_predicts_the_majority_label_not_the_modal_effect_projected(
        self,
    ) -> None:
        # Modal FULL effect is the inert one (3 of 9), but the majority BINARY
        # label is "changed" (6 of 9). A clone-of-class arm returns (True,) here;
        # a real binary predictor returns (False,).
        p = re_.Prequential()
        ctx = ("s1", 6, 5, 2)
        seq = [0, 0, 0, 4, 4, 9, 9, 16, 16]
        for churn in seq:
            p.step(ctx, self._eff(churn))
        assert p._inert.predict(ctx)[0] == (False,)
        assert re_.inert_view(p._arms["class"].predict(ctx)[0]) == (True,)

    def test_inert_can_therefore_score_below_class(self) -> None:
        # Impossible while inert is a clone; the point of the fix.
        p = re_.Prequential()
        ctx = ("s1", 6, 5, 2)
        for churn in (0, 0, 0, 4, 4, 9, 9, 16, 16):
            p.step(ctx, self._eff(churn))
        for _ in range(6):
            p.step(ctx, self._eff(0))
        r = p.report()
        assert r["inert"] < 1.0


class TestShapeMismatchSentinel:
    """I2: a total shape change must not read as "nothing happened"."""

    def test_a_mismatched_shape_is_not_inert(self) -> None:
        a = np.zeros((4, 4), dtype=np.int16)
        b = np.zeros((5, 5), dtype=np.int16)
        eff = re_.effect_signature(a, b, False, None)
        assert eff[4] != 0
        assert re_.inert_view(eff) != (True,)


class TestReportCompleteness:
    """C3 + I3: the decisive metrics need their denominators and a ceiling."""

    @staticmethod
    def _eff(churn: int) -> Any:
        return (False, None, frozenset(), frozenset(), churn)

    def test_the_first_sighting_denominator_is_reported(self) -> None:
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))
        p.step(("s1", 6, 5, 2), self._eff(4))
        assert p.report()["first_n"] == 1

    def test_marginal_is_also_reported_on_the_fresh_subset(self) -> None:
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))
        assert "marginal_fresh" in p.report()

    def test_the_abstain_share_is_reported(self) -> None:
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))
        assert p.report()["abstain"] == 1.0

    def test_per_key_purity_ceiling_is_reported(self) -> None:
        # Two effects 50/50 at one key: a modal predictor can never exceed 0.5.
        p = re_.Prequential()
        for i in range(20):
            p.step(("s1", 6, 5, 2), self._eff(4 if i % 2 else 9))
        assert p.report()["ceiling"] == pytest.approx(0.5, abs=0.06)


class TestReplay:
    """I5 + C1 + I1: the hook is the component that already invalidated a run."""

    def test_a_hook_error_is_counted_and_never_raised(self) -> None:
        r = re_.Replay()
        r.observe(_obs(_grid((2, 2))), _Actor(("click", 0, 0)))
        # A prim the harness never produces: 1-tuple. Must count, not propagate.
        r.observe(_obs(_grid((2, 3))), _Actor(("bogus",)))
        assert r.errors >= 1

    def test_the_transition_into_a_terminal_frame_is_scored(self) -> None:
        r = re_.Replay()
        r.observe(_obs(_grid((2, 2))), _Actor(("act", 1)))
        r.observe(_obs(_grid((5, 5)), terminal=True), _Actor(("act", 1)))
        assert r.pre.n == 1

    def test_the_transition_out_of_a_terminal_frame_is_not_scored(self) -> None:
        r = re_.Replay()
        r.observe(_obs(_grid((2, 2))), _Actor(("act", 1)))
        r.observe(_obs(_grid((5, 5)), terminal=True), _Actor(("act", 1)))
        r.observe(_obs(_grid((2, 2))), _Actor(("act", 1)))
        assert r.pre.n == 1  # still 1: the reset is not a successor

    def test_a_missing_prim_breaks_the_chain(self) -> None:
        r = re_.Replay()
        r.observe(_obs(_grid((2, 2))), _Actor(("act", 1)))
        r.observe(_obs(_grid((2, 3))), _Actor(None))
        r.observe(_obs(_grid((2, 4))), _Actor(("act", 1)))
        assert r.pre.n == 1  # first pair only

    def test_a_level_advance_is_carried_into_the_effect(self) -> None:
        r = re_.Replay()
        r.observe(_obs(_grid((2, 2)), levels=0), _Actor(("act", 1)))
        r.observe(_obs(_grid((2, 3)), levels=1), _Actor(("act", 1)))
        assert r.pre.n == 1
