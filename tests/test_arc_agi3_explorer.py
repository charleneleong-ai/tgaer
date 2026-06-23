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


def _obs(
    board: np.ndarray, levels: int = 0, actions=(1, 2, 3, 4), terminal: bool = False
) -> dict:
    obs = {
        "frame": [board.tolist()],
        "available_actions": list(actions),
        "levels_completed": levels,
        "state": "NOT_FINISHED",
    }
    if terminal:  # a respawn frame after death — the prior action was fatal
        obs["terminal"] = True
    return obs


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

    def test_goal_value_clicks_come_first(self):
        # value 7 is the bigger (more salient) blob; value 5 is a known goal. The
        # goal-value click must be proposed before the larger blind salience pick.
        board = _board(extra={5: [(6, 3)], 7: [(2, 2), (2, 3), (2, 4)]})
        prims = proposals(board, [6], goal_values={5})
        assert prims[0] == ("click", 6, 3)


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


class TestTerminalAvoidance:
    def test_avoids_fatal_action_in_fallback(self):
        # act1 dies, act2 is safe. Once both are taken and the node is re-entered
        # with nothing untested, the fallback must reuse the safe act2 — not act1,
        # which is the first proposal it would otherwise default to.
        agent = ExplorerArcAgi3Agent()
        board = _board()
        agent.act(_obs(board, actions=(1, 2)))  # take act1
        agent.act(_obs(board, actions=(1, 2), terminal=True))  # act1 fatal; take act2
        act = agent.act(_obs(board, actions=(1, 2)))  # fallback must skip fatal act1
        assert act.id == 2

    def test_map_survives_death_respawn(self):
        # Build start --act2--> frontier T on level 1, then die: the respawn drops
        # levels 1->0, which must NOT wipe the map. The agent should route along
        # the surviving start--act2-->T edge, not restart by re-exploring act1.
        agent = ExplorerArcAgi3Agent()
        start, other = _board(avatar=(2, 2)), _board(avatar=(5, 5))
        agent.act(_obs(start, levels=1, actions=(1, 2)))  # take act1 at start
        agent.act(_obs(start, levels=1, actions=(1, 2)))  # take act2 (start exhausted)
        agent.act(_obs(other, levels=1, actions=(1, 2)))  # arrived at frontier T
        act = agent.act(_obs(start, levels=0, actions=(1, 2), terminal=True))  # respawn
        assert act.id == 2  # routes along the surviving start--act2-->T edge


class TestWinInduction:
    def test_advancing_click_value_is_learned_and_re_clicked(self):
        # Click value 5; that advances the level. On the new level a bigger value-7
        # blob would win blind salience, but the induced goal re-clicks value 5.
        agent = ExplorerArcAgi3Agent()
        a1 = agent.act(_obs(_board(extra={5: [(4, 6)]}), levels=0, actions=(6,)))
        assert a1.id == 6 and (a1.y, a1.x) == (4, 6)  # clicked value 5
        nxt = _board(extra={5: [(6, 3)], 7: [(2, 2), (2, 3), (2, 4)]})
        a2 = agent.act(_obs(nxt, levels=1, actions=(6,)))  # level++ → induce value 5
        assert a2.id == 6 and (a2.y, a2.x) == (6, 3)  # re-clicks value 5, not blob 7

    def test_navigate_win_induces_no_click_goal(self):
        # A level-up after a directional act must not crash or learn a click goal.
        agent = ExplorerArcAgi3Agent()
        board = _board()
        agent.act(_obs(board, levels=0, actions=(1, 2)))  # took an ("act", _)
        agent.act(_obs(board, levels=1, actions=(1, 2)))  # level++ after a non-click
        assert agent._goal_values == set()
