from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics


def _grid(avatar_rc: tuple[int, int], avatar: int = 12) -> np.ndarray:
    """10x10 green field, yellow(4) wall border, one avatar cell."""
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    g[avatar_rc] = avatar
    return g


class TestAvatarDetection:
    def test_pins_after_two_consistent_action_deltas_across_actions(self):
        det = EmpiricalSemantics()
        # action 1 moves avatar down a row (twice, consistent); action 2 moves it
        # right (distinct outcome) -> controllable -> pins to 12.
        det.observe(_grid((2, 2)), 1, _grid((3, 2)), 0)
        det.observe(_grid((3, 2)), 2, _grid((3, 3)), 0)
        det.observe(_grid((3, 3)), 1, _grid((4, 3)), 0)
        assert det.avatar == 12

    def test_action_independent_distractor_never_pins(self):
        det = EmpiricalSemantics()
        # value 7 drifts (0,+1) every step REGARDLESS of action -> same Δ across
        # actions -> not controllable -> never the avatar.
        def g(col: int) -> np.ndarray:
            a = np.full((10, 10), 3, dtype=int)
            a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 4
            a[5, col] = 7
            return a

        det.observe(g(2), 1, g(3), 0)
        det.observe(g(3), 2, g(4), 0)
        det.observe(g(4), 1, g(5), 0)
        assert det.avatar is None

    def test_single_frame_does_not_pin(self):
        det = EmpiricalSemantics()
        det.observe(_grid((2, 2)), 1, _grid((3, 2)), 0)
        assert det.avatar is None
