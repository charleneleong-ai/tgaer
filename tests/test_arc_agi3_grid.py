# tests/test_arc_agi3_grid.py
from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    LS20_DEFAULT,
    KeyDoorController,
    Semantics,
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
    arr = np.zeros((12, 12), dtype=int)  # background 0, NOT green
    arr[5:8, 5:8] = 3  # small green play-field
    arr[6, 6] = 9  # door inside the field
    arr[0, 0] = 9  # stray door cell far outside (pad=4 won't reach)
    box = field_box(arr)
    found = find_role(arr, (9,), box)
    assert any(abs(c[0] - 6) < 1 and abs(c[1] - 6) < 1 for c in found)  # inside found
    assert not any(c[0] < 1 and c[1] < 1 for c in found)  # (0,0) excluded


def test_ls20_default_is_navigate():
    assert LS20_DEFAULT.verb == "navigate"
    assert isinstance(LS20_DEFAULT, Semantics)


def _ld_board():
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    g[2, 2] = 12
    g[5, 5] = 0
    g[7, 7] = 9
    return g


class TestKeyDoorController:
    def test_step_returns_directional_action(self):
        c = KeyDoorController()
        act = c.step(_ld_board(), LS20_DEFAULT, [1, 2, 3, 4])
        assert isinstance(act, ArcAction) and act.id in (1, 2, 3, 4)

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
