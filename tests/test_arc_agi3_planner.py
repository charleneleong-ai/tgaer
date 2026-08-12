# tests/test_arc_agi3_planner.py
from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    LS20_DEFAULT,
    avatar_is_sprite,
    components,
    field_box,
    find_role,
)
from tgaer.agents.arc_agi3_planner import PlannerArcAgi3Agent


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
