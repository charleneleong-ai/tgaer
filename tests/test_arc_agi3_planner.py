# tests/test_arc_agi3_planner.py
from __future__ import annotations

from collections import Counter

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    LS20_DEFAULT,
    avatar_is_sprite,
    components,
    field_box,
    find_role,
)
from tgaer.agents.arc_agi3_planner import PlannerArcAgi3Agent
from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics


# A small synthetic LS20-style board: green floor (3), a 1x1 darkred avatar (12)
# top-left, a black/blue key (0/1) mid-board, a maroon door (9) bottom-right,
# yellow wall border (4). 1-cell avatar so the move lattice is unit steps.
def _board() -> np.ndarray:
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4  # wall border
    g[2, 2] = 12  # avatar
    g[5, 5] = 0  # key marker
    g[7, 7] = 9  # door
    return g


def _obs(board: np.ndarray, levels: int = 0, actions=(1, 2, 3, 4)) -> dict:
    return {
        "frame": [board.tolist()],
        "available_actions": list(actions),
        "levels_completed": levels,
        "state": "NOT_FINISHED",
    }


class TestGeometry:
    def test_components_separates_disjoint_clusters(self):
        arr = np.full((6, 6), 3, dtype=int)
        arr[1, 1] = arr[1, 2] = 0
        arr[4, 4] = 0
        comps = components(arr, (0,))
        assert sorted(len(c) for c in comps) == [1, 2]

    def test_keys_and_door_found_inside_field(self):
        board = _board()
        box = field_box(board)
        assert len(find_role(board, LS20_DEFAULT.keys, box)) == 1
        assert find_role(board, (LS20_DEFAULT.door,), box)

    def test_avatar_sprite_accepts_piece_rejects_structure(self):
        board = _board()
        assert avatar_is_sprite(board, LS20_DEFAULT.avatar)  # one compact in-field cell
        assert not avatar_is_sprite(board, 4)  # wall border: large, structural


class TestPlannerNavigates:
    def test_learns_a_move_vector_after_one_real_move(self):
        # Drive the agent once, then feed back a board where the avatar moved
        # down one row; the agent must record action->delta for that action.
        a = PlannerArcAgi3Agent()
        a.act(_obs(_board()))
        moved = _board()
        moved[2, 2] = 3
        moved[3, 2] = 12  # avatar shifted down one row
        a.act(_obs(moved))
        assert a._ctl.delta  # learned at least one action->vector
        assert next(iter(a._ctl.delta.values())).shape == (2,)

    def test_resets_phase_on_new_level(self):
        a = PlannerArcAgi3Agent()
        a.phase = "door"
        a.act(_obs(_board(), levels=1))  # level changed 0->1
        assert a._ctl.phase == "key"
        assert a._levels == 1


class TestAvatarSelection:
    """What actually picks the avatar, once controllability has filtered.

    `_observe_motion` screens candidates with `_is_controllable` first, so the
    sprite tiebreak only ever sees values that already respond to actions.
    Measured across all 25 games: it split that set 0 times and changed the
    pick 0 times — every avatar was decided by smallest footprint. The guard
    reads as a safeguard against picking a wall, and there is no run in which
    it does that.
    """

    @staticmethod
    def _detector(deltas: dict[int, dict[int, object]]):
        det = EmpiricalSemantics()
        for value, per_action in deltas.items():
            slot = det._deltas.setdefault(value, {})
            for action, counts in per_action.items():
                slot.setdefault(action, Counter()).update(counts)
        return det

    def test_the_smallest_controllable_value_wins(self):
        """Two values both controllable and both compact: size decides."""
        det = self._detector(
            {
                5: {1: {(1, 0): 3}, 2: {(-1, 0): 3}},
                7: {1: {(1, 0): 3}, 2: {(-1, 0): 3}},
            }
        )
        arr = np.full((20, 20), 3, dtype=int)
        arr[2, 2] = 5  # 1 cell
        arr[5:7, 5:7] = 7  # 4 cells
        prev = arr.copy()
        prev[2, 2], prev[1, 2] = 3, 5
        det._observe_motion(prev, 1, arr)
        assert det.avatar == 5

    def test_an_uncontrollable_value_is_never_the_avatar(self):
        """A wall is excluded before size or shape is consulted at all."""
        det = self._detector(
            {
                5: {1: {(1, 0): 3}, 2: {(-1, 0): 3}},  # controllable
                11: {1: {(0, 0): 5}},  # never moves
            }
        )
        arr = np.full((20, 20), 3, dtype=int)
        arr[2, 2] = 5
        arr[0, :] = 11
        prev = arr.copy()
        prev[2, 2], prev[1, 2] = 3, 5
        det._observe_motion(prev, 1, arr)
        assert det.avatar == 5
