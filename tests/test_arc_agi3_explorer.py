# tests/test_arc_agi3_explorer.py
from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_explorer import (
    ExplorerArcAgi3Agent,
    StateGraph,
    click_targets,
    frame_signature,
    proposals,
)
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction


# A 10x10 green (3) play-field with a yellow (4) wall border, a 1-cell avatar
# (12). The border sits OUTSIDE the green field box, so signatures key on the
# interior only. Extra blobs are painted by `extra` = {value: [(r, c), ...]}.
def _board(avatar=(2, 2), extra: dict | None = None) -> np.ndarray:
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    g[avatar] = 12
    for val, cellset in (extra or {}).items():
        for r, c in cellset:
            g[r, c] = val
    return g


def _obs(board: np.ndarray, levels: int = 0, actions=(1, 2, 3, 4)) -> dict:
    return {
        "frame": [board.tolist()],
        "available_actions": list(actions),
        "levels_completed": levels,
        "state": "NOT_FINISHED",
    }


class TestFrameSignature:
    def test_moved_avatar_changes_signature(self):
        assert frame_signature(_board(avatar=(2, 2))) != frame_signature(
            _board(avatar=(5, 5))
        )

    def test_change_outside_field_box_is_ignored(self):
        # The (0,0) corner is wall border, outside the green field box — a HUD-like
        # change there must not alter the signature (status-bar masking).
        b = _board()
        b[0, 0] = 7
        assert frame_signature(b) == frame_signature(_board())


class TestClickTargets:
    def test_orders_components_by_size(self):
        # value 5 is a 3-cell blob, value 7 a single cell — the larger, more
        # salient blob's centroid must come first.
        board = _board(extra={5: [(4, 4), (4, 5), (5, 4)], 7: [(8, 8)]})
        targets = click_targets(board)
        big = (round(np.mean([4, 4, 5])), round(np.mean([4, 5, 4])))
        assert targets[0] == big

    def test_excludes_out_of_field_walls(self):
        # The wall border (value 4) is outside the field box and must not appear.
        board = _board(extra={5: [(4, 4)]})
        for r, c in click_targets(board):
            assert board[r, c] != 4


class TestProposals:
    def test_simple_actions_become_act_primitives(self):
        assert proposals(_board(), [1, 2, 3]) == [("act", 1), ("act", 2), ("act", 3)]

    def test_action6_expands_into_click_targets(self):
        board = _board(extra={5: [(4, 4)]})
        prims = proposals(board, [6])
        assert prims and all(p[0] == "click" for p in prims)
        # every click primitive carries an in-field (row, col)
        assert all(len(p) == 3 for p in prims)


class TestStateGraph:
    def test_register_sets_untested_once(self):
        g = StateGraph()
        g.register("A", [("act", 1), ("act", 2)])
        g.take("A", ("act", 1))
        g.register("A", [("act", 1), ("act", 2)])  # re-sighting must not resurrect
        assert g.untested_at("A") == [("act", 2)]

    def test_path_empty_when_start_has_untested(self):
        g = StateGraph()
        g.register("A", [("act", 1)])
        assert g.path_to_frontier("A") == []

    def test_path_routes_to_nearest_frontier(self):
        g = StateGraph()
        g.register("A", [("act", 1)])
        g.connect("A", ("act", 1), "B")
        g.take("A", ("act", 1))
        g.register("B", [("act", 2)])  # A exhausted, B has untested
        assert g.path_to_frontier("A") == [("act", 1)]

    def test_path_none_when_no_frontier_reachable(self):
        g = StateGraph()
        g.register("A", [("act", 1)])
        g.take("A", ("act", 1))
        assert g.path_to_frontier("A") is None


class TestExplorerLoop:
    def test_emits_a_legal_action(self):
        act = ExplorerArcAgi3Agent().act(_obs(_board()))
        assert isinstance(act, ArcAction) and act.id in (1, 2, 3, 4)

    def test_tries_each_untested_action_before_repeating(self):
        # Feeding the SAME board (avatar never moves in this stub) means the
        # signature is constant; the explorer must cycle through all four
        # untested directional actions before any repeat.
        agent = ExplorerArcAgi3Agent()
        board = _board()
        ids = [agent.act(_obs(board)).id for _ in range(4)]
        assert sorted(ids) == [1, 2, 3, 4]

    def test_new_level_resets_exploration(self):
        # Exhaust the node, then bump levels_completed: the fresh level must be
        # explored again rather than the agent giving up on the (same) signature.
        agent = ExplorerArcAgi3Agent()
        board = _board()
        for _ in range(4):
            agent.act(_obs(board, levels=0))
        ids = [agent.act(_obs(board, levels=1)).id for _ in range(4)]
        assert sorted(ids) == [1, 2, 3, 4]

    def test_click_game_targets_a_component_not_blind_center(self):
        board = _board(extra={5: [(4, 4), (4, 5)]})
        act = ExplorerArcAgi3Agent().act(_obs(board, actions=(6,)))
        assert act.id == 6 and act.x is not None and act.y is not None
