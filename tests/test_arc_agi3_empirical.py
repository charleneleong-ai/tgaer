from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_empirical import EmpiricalPlannerAgent
from tgaer.agents.arc_agi3_grid import LS20_DEFAULT, Semantics
from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction
from tgaer.evaluation import dispatch


def _grid2(avatar_rcs: list[tuple[int, int]], avatar: int = 12) -> np.ndarray:
    """10x10 green field, yellow(4) wall border, multi-cell avatar."""
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    for rc in avatar_rcs:
        g[rc] = avatar
    return g


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


class TestMultiCellAvatar:
    """FIX 1: adjacency measured against nearest avatar cell, not centroid."""

    def test_key_adjacent_to_multicell_footprint_edge_pins(self):
        # Avatar occupies (3,3) and (3,4); key at (3,5) is 1 cell from footprint edge.
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _grid2([(3, 3), (3, 4)])
        prev[3, 5] = 0  # key adjacent to right edge of avatar
        cur = _grid2([(3, 3), (3, 4)])  # key vanishes (picked up)
        det.observe(prev, 1, cur, 0)
        assert det.keys == (0,)

    def test_key_two_cells_beyond_footprint_does_not_pin(self):
        # Avatar occupies (3,3) and (3,4); key at (3,6) is 2 cells from footprint.
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _grid2([(3, 3), (3, 4)])
        prev[3, 6] = 0  # too far from avatar footprint
        cur = _grid2([(3, 3), (3, 4)])  # key vanishes anyway
        det.observe(prev, 1, cur, 0)
        assert det.keys == ()

    def test_door_adjacent_to_multicell_footprint_edge_pins(self):
        # Avatar at (3,3)+(3,4); door at (3,5) vanishes on level increment.
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _grid2([(3, 3), (3, 4)])
        prev[3, 5] = 9  # door adjacent to right edge
        cur = _grid2([(3, 3), (3, 4)])  # door gone (level bumped)
        det.observe(prev, 1, cur, 1)
        assert det.door == 9

    def test_door_two_cells_beyond_footprint_does_not_pin(self):
        # Avatar at (3,3)+(3,4); door at (3,6) vanishes but is too far.
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _grid2([(3, 3), (3, 4)])
        prev[3, 6] = 9
        cur = _grid2([(3, 3), (3, 4)])
        det.observe(prev, 1, cur, 1)
        assert det.door is None


class TestDoorVanishRequirement:
    """FIX 2: door must have vanished (prev→cur) on level-increment step."""

    def test_non_vanishing_decoration_not_pinned_as_door(self):
        # On level-increment, decoration tile (8) is adjacent but stays in cur.
        # Actual door tile (9) is adjacent AND vanishes.
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _with((3, 3), {(3, 4): 8, (4, 3): 9})  # deco(8) + door(9) adjacent
        # cur: deco(8) persists, door(9) gone
        cur = _with((3, 3), {(3, 4): 8})
        det.observe(prev, 1, cur, 1)
        assert det.door == 9  # pinned to vanished tile, not static decoration

    def test_only_non_vanishing_tiles_present_no_pin(self):
        # Only decoration tile adjacent; nothing vanishes -> no door pinned.
        det = EmpiricalSemantics()
        det._avatar = 12
        prev = _with((3, 3), {(3, 4): 8})
        cur = _with((3, 3), {(3, 4): 8})  # nothing changed except level
        det.observe(prev, 1, cur, 1)
        assert det.door is None


class TestAvatarSpritePreference:
    """FIX 3: sprite-preferring tiebreak over array-index order."""

    def test_small_sprite_preferred_over_lower_indexed_blob(self):
        # Value 0 (lower index, 2×2 blob) and value 12 (higher index, 1-cell sprite)
        # both satisfy controllability simultaneously on step 3. With the old code,
        # value 0 is iterated first in set-order and incorrectly pinned as the avatar.
        # The fix must pin 12 (the sprite), not 0 (the blob).
        det = EmpiricalSemantics()

        def _scene(
            blob_r: int, blob_c: int, sprite_r: int, sprite_c: int
        ) -> np.ndarray:
            g = np.full((10, 10), 3, dtype=int)
            g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
            g[blob_r : blob_r + 2, blob_c : blob_c + 2] = 0  # 2×2 blob (lower index)
            g[sprite_r, sprite_c] = 12  # single-cell sprite (higher index)
            return g

        # a1: both move down (+1 row); count 1 for each.
        det.observe(_scene(4, 4, 2, 2), 1, _scene(5, 4, 3, 2), 0)
        # a2: both move right (+1 col); distinct delta established.
        det.observe(_scene(5, 4, 3, 2), 2, _scene(5, 5, 3, 3), 0)
        # a1 again: count 2 for action 1 → blob(0) satisfies controllability first
        # in set iteration order [0,3,4,12]; old code pins 0 (bug), fix pins 12.
        det.observe(_scene(5, 5, 3, 3), 1, _scene(6, 5, 4, 3), 0)
        assert det.avatar == 12


class TestAgentIntegrationExtended:
    """FIX 4: high-value integration tests for EmpiricalPlannerAgent."""

    def test_evidence_overrides_wrong_cold_start_avatar(self):
        # Cold start claims avatar=5 (wrong). True avatar is 12 and moves.
        # We drive the agent through 4 frames with controlled prev_action overrides
        # so the detector sees transitions: (f0,a1,f1), (f1,a2,f2), (f2,a1,f3).
        # After 3 transitions avatar 12 satisfies controllability → overrides cold start.
        wrong_cold = Semantics(
            avatar=5, keys=(0, 1), door=9, walls=(4, 11), verb="navigate"
        )
        sci = _FakeSci(wrong_cold)
        agent = EmpiricalPlannerAgent(scientist=sci)

        frames = [_grid((2, 2)), _grid((3, 2)), _grid((3, 3)), _grid((4, 3))]
        inject_actions = [1, 2, 1]  # actions we want the detector to "see"

        for i, f in enumerate(frames):
            obs = {
                "frame": [f.tolist()],
                "available_actions": [1, 2, 3, 4],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
            }
            agent.act(obs)
            # Override prev_action so next observe() sees our controlled action.
            if i < len(inject_actions):
                agent._prev_action = inject_actions[i]

        assert agent._det.avatar == 12

    def test_pinned_role_persists_across_level_bump(self):
        # Pin a key role, then bump the level; detector's pinned key must survive.
        agent = EmpiricalPlannerAgent(scientist=None, api_base=None)
        agent._det._avatar = 12
        agent._det._keys = {0}  # manually pin a key role
        agent._levels = 0

        # Act with level=1 (bump) — controller resets, but detector keeps pins
        obs = {
            "frame": [_grid((2, 2)).tolist()],
            "available_actions": [1, 2, 3, 4],
            "levels_completed": 1,
            "state": "NOT_FINISHED",
        }
        agent.act(obs)

        assert 0 in agent._det._keys  # key role survived the level bump

    def test_blocked_avatar_no_crash_uses_legal_action(self):
        # Avatar never moves (surrounded by walls) — detector never pins avatar.
        # Agent must keep emitting legal actions from cold-start LS20.
        agent = EmpiricalPlannerAgent(scientist=None, api_base=None)
        # Walled-in grid: avatar at (5,5) surrounded by walls on all sides
        g = np.full((10, 10), 3, dtype=int)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
        g[5, 5] = 12
        g[4, 5] = g[6, 5] = g[5, 4] = g[5, 6] = 4  # wall all four neighbours

        for _ in range(5):
            obs = {
                "frame": [g.tolist()],
                "available_actions": [1, 2, 3, 4],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
            }
            act = agent.act(obs)
            assert isinstance(act, ArcAction) and act.id in (1, 2, 3, 4)

        assert agent._det.avatar is None  # never pinned due to no movement


class TestDispatchWiring:
    def test_empirical_kind_registered(self):
        assert dispatch._ARC_AGI3_AGENTS["empirical"] is EmpiricalPlannerAgent
