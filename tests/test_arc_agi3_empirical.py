from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_grid import LS20_DEFAULT, Semantics, cells  # noqa: E402
from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics


def _grid(avatar_rc: tuple[int, int], avatar: int = 12) -> np.ndarray:
    """10x10 green field, yellow(4) wall border, one avatar cell."""
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    g[avatar_rc] = avatar
    return g


def _with(avatar_rc: tuple[int, int], extra: dict[tuple[int, int], int]) -> np.ndarray:
    g = _grid(avatar_rc)
    for rc, v in extra.items():
        g[rc] = v
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


class TestKeyDetection:
    def test_value_vanishing_under_avatar_pins_key(self):
        det = EmpiricalSemantics()
        det._avatar = 12  # avatar already known
        prev = _with((3, 3), {(3, 4): 0})  # key(0) adjacent-right of avatar
        cur = _grid((3, 4))  # avatar steps onto it; key gone
        det.observe(prev, 1, cur, 0)
        assert det.keys == (0,)

    def test_value_vanishing_far_from_avatar_is_not_key(self):
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _with((3, 3), {(8, 8): 0})  # value 0 nowhere near avatar
        cur = _grid((3, 4))  # 0 vanished but not under avatar
        det.observe(prev, 1, cur, 0)
        assert det.keys == ()


class TestDoorDetection:
    def test_level_increment_pins_avatar_adjacent_value_as_door(self):
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _with((3, 3), {(3, 4): 9})  # door(9) adjacent-right
        cur = _grid((3, 4))  # avatar reaches door; levels ticks 0->1
        det.observe(prev, 1, cur, 1)
        assert det.door == 9

    def test_no_increment_no_door(self):
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _with((3, 3), {(3, 4): 9})
        cur = _grid((3, 4))
        det.observe(prev, 1, cur, 0)  # levels stayed 0
        assert det.door is None


class TestSemanticsMerge:
    def test_unpinned_falls_back_to_cold_start(self):
        det = EmpiricalSemantics()
        assert det.semantics(LS20_DEFAULT) == LS20_DEFAULT

    def test_pinned_avatar_overrides_disagreeing_cold_start(self):
        det = EmpiricalSemantics()
        det._avatar = 5  # evidence says avatar is 5, cold-start said 12
        sem = det.semantics(LS20_DEFAULT)
        assert sem.avatar == 5
        assert sem.walls == LS20_DEFAULT.walls  # walls untouched

    def test_pinned_keys_and_door_override(self):
        det = EmpiricalSemantics()
        det._keys = {7}
        det._door = 8
        sem = det.semantics(Semantics(12, (0, 1), 9, (4,), "navigate"))
        assert sem.keys == (7,)
        assert sem.door == 8


from tgaer.agents.arc_agi3_empirical import EmpiricalPlannerAgent  # noqa: E402
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction  # noqa: E402


class _FakeSci:
    def __init__(self, sem):
        self.sem = sem
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return self.sem


def _obs(avatar_rc=(2, 2), levels=0):
    g = _grid(avatar_rc)
    g[5, 5] = 0
    g[7, 7] = 9
    return {
        "frame": [g.tolist()],
        "available_actions": [1, 2, 3, 4],
        "levels_completed": levels,
        "state": "NOT_FINISHED",
    }


class TestAgentIntegration:
    def test_emits_legal_action(self):
        agent = EmpiricalPlannerAgent(scientist=_FakeSci(LS20_DEFAULT))
        act = agent.act(_obs())
        assert isinstance(act, ArcAction) and act.id in (1, 2, 3, 4)

    def test_cold_start_queried_once_per_episode(self):
        sci = _FakeSci(LS20_DEFAULT)
        agent = EmpiricalPlannerAgent(scientist=sci)
        for _ in range(5):
            agent.act(_obs())
        assert sci.calls == 1

    def test_absent_scientist_falls_back_to_ls20(self):
        agent = EmpiricalPlannerAgent(scientist=None, api_base=None)
        act = agent.act(_obs())  # no VL, no crash, still plays
        assert act.id in (1, 2, 3, 4)
