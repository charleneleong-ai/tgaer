# tests/test_arc_agi3_grid.py
from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    LS20_DEFAULT,
    KeyDoorController,
    Semantics,
    avatar_is_sprite,
    field_box,
    find_role,
)
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction


def test_find_role_filters_to_field_box():
    arr = np.full((10, 10), 3, dtype=int)
    arr[5, 5] = 9
    arr[0, 0] = 9  # a second door cell; both are inside the field box here
    box = field_box(arr)
    found = find_role(arr, (9,), box)
    assert any(abs(c[0] - 5) < 1 and abs(c[1] - 5) < 1 for c in found)


def test_find_role_excludes_centroid_outside_field():
    """The field is the modal colour's extent, not a fixed colour.

    This previously used a 3x3 green blob in a field of 0 and relied on
    field_box keying on GREEN. That prior is what blinded tn36, where all 88
    components fell outside a box drawn around an unrelated green decoration,
    so the scenario is rebuilt to identify the field the way the code now does:
    the surround is split across two colours so neither outnumbers the field.
    """
    arr = np.full((30, 30), 1, dtype=int)
    arr[20:, :] = 2  # surround split so the field stays the modal colour
    arr[0:20, 0:20] = 8  # 399 field cells against 299 + 200 of surround
    arr[10, 10] = 9  # door inside the field
    arr[29, 29] = 9  # stray door cell far outside (pad=4 will not reach)
    box = field_box(arr)
    found = find_role(arr, (9,), box)
    assert any(abs(c[0] - 10) < 1 and abs(c[1] - 10) < 1 for c in found)
    assert not any(c[0] > 25 and c[1] > 25 for c in found)


def _ld_board():
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    g[2, 2] = 12
    g[5, 5] = 0
    g[7, 7] = 9
    return g


class TestKeyDoorController:
    def test_on_new_level_resets_phase_keeps_delta(self):
        c = KeyDoorController()
        c.delta = {1: np.array([1, 0])}
        c.phase = "door"
        c.on_new_level()
        assert c.phase == "key" and 1 in c.delta

    def test_learn_records_delta_on_real_move(self):
        c = KeyDoorController()
        c.step(_ld_board(), LS20_DEFAULT, [1, 2, 3, 4])  # records prev_tl + action
        moved = _ld_board()
        moved[2, 2] = 3
        moved[3, 2] = 12
        c.learn(moved, LS20_DEFAULT)
        assert c.delta

    def test_press_verb_emits_interaction_when_adjacent(self):
        # avatar already next to the door; press verb -> interaction action, not a step
        g = np.full((6, 6), 3, dtype=int)
        g[2, 2] = 12
        g[2, 3] = 9  # door directly to the right (adjacent)
        sem = Semantics(avatar=12, keys=(), door=9, walls=(4,), verb="press")
        c = KeyDoorController()
        c.delta = {
            1: np.array([1, 0]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, -1]),
        }
        act = c.step(g, sem, [1, 2, 3, 4, 5])
        assert act.id == 5  # keyboard interaction preferred

    def test_press_verb_emits_interaction_adjacent_to_key(self):
        # keys present; avatar adjacent to nearest key -> targets key, emits interaction
        g = np.full((6, 6), 3, dtype=int)
        g[2, 2] = 12  # avatar
        g[2, 3] = 0  # key directly to the right (adjacent, cover=1)
        g[5, 5] = 9  # door far away
        sem = Semantics(avatar=12, keys=(0,), door=9, walls=(4,), verb="press")
        c = KeyDoorController()
        c.delta = {
            1: np.array([1, 0]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, -1]),
        }
        act = c.step(g, sem, [1, 2, 3, 4, 5])
        assert act.id == 5  # interaction emitted at the key, not a movement

    def test_press_verb_navigates_when_target_distant(self):
        # no keys; door is far away -> controller navigates (move), does NOT press
        g = np.full((8, 8), 3, dtype=int)
        g[1, 1] = 12  # avatar top-left
        g[6, 6] = 9  # door far away (cover >> 1)
        sem = Semantics(avatar=12, keys=(), door=9, walls=(4,), verb="press")
        c = KeyDoorController()
        c.delta = {
            1: np.array([1, 0]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, -1]),
        }
        act = c.step(g, sem, [1, 2, 3, 4, 5])
        assert act.id in (1, 2, 3, 4)  # moves toward target; does NOT press yet


class TestClickVerb:
    def _sem(self):
        return Semantics(avatar=12, keys=(0,), door=9, walls=(4,), verb="click")

    def test_clicks_key_at_col_row_convention(self):
        # key at array cell [3][5] -> ArcAction(id=6, x=col=5, y=row=3)
        g = np.full((8, 8), 3, dtype=int)
        g[3, 5] = 0  # key
        g[6, 6] = 9  # door
        act = KeyDoorController().step(g, self._sem(), [6])
        assert act.id == 6 and act.x == 5 and act.y == 3

    def test_clicks_door_once_keys_gone(self):
        g = np.full((8, 8), 3, dtype=int)
        g[6, 2] = 9  # door only, no keys
        act = KeyDoorController().step(g, self._sem(), [6])
        assert act.id == 6 and act.x == 2 and act.y == 6

    def test_falls_back_when_action6_absent(self):
        g = np.full((8, 8), 3, dtype=int)
        g[3, 5] = 0
        act = KeyDoorController().step(g, self._sem(), [1, 2, 3, 4])
        assert act.id in (1, 2, 3, 4)  # no ACTION6 -> keyboard fallback, no crash

    def test_falls_back_when_no_target(self):
        g = np.full((8, 8), 3, dtype=int)  # no key, no door
        act = KeyDoorController().step(g, self._sem(), [6])
        assert isinstance(act, ArcAction)  # never crashes; centre/keyboard fallback


def test_sprite_size_is_judged_against_the_grid_not_the_field_box():
    """The box moves; the meaning of "small" must not move with it.

    field_box now takes the modal colour's extent, which varied 2255 -> 4096
    on ls20 alone, so a threshold scaled by it silently loosened from 68 cells
    to 123. Scaling by the grid keeps one verdict per board.
    """
    arr = np.full((40, 40), 5, dtype=int)
    arr[26:, :] = 1  # 560 cells
    arr[0:26, 25:] = 2  # 390 cells; colour 5 keeps 630 and stays modal
    arr[2:6, 2:7] = 12  # a 20-cell sprite, 3.1% of the 650-cell box, 1.3% of grid

    lo, hi = field_box(arr)
    box_area = (int(hi[0]) - int(lo[0]) + 1) * (int(hi[1]) - int(lo[1]) + 1)
    assert box_area < arr.size, "the box must be a sub-region for this to bite"
    assert avatar_is_sprite(arr, 12), (
        f"20 cells is 1.3% of the grid but {20 / box_area:.1%} of the box; "
        "judging against the box rejects a sprite for where the box happens to fall"
    )
