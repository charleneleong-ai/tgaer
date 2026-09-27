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
