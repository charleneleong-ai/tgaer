# tests/test_arc_agi3_explorer.py
from __future__ import annotations

from collections import Counter

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


class TestNavInduction:
    def test_act_feeds_chosen_action_effect_to_detector(self):
        # The explorer must learn its own action→motion stream: after a step where
        # the avatar moved down (1,0), the detector records chosen-action → that Δ.
        agent = ExplorerArcAgi3Agent()
        a = agent.act(_obs(_board(avatar=(2, 2)), actions=(1, 2, 3, 4)))
        agent.act(_obs(_board(avatar=(3, 2)), actions=(1, 2, 3, 4)))  # avatar +1 row
        assert agent._det._deltas.get(12, {}).get(a.id, Counter())[(1, 0)] == 1

    def test_act_induces_door_on_directional_levelup(self):
        # Avatar already pinned; a directional level-up with a value vanishing under
        # the avatar pins that value as the navigate goal (door).
        agent = ExplorerArcAgi3Agent()
        agent._det._avatar = 12
        agent.act(_obs(_board(avatar=(3, 3), extra={9: [(3, 4)]}), actions=(1, 2)))
        agent.act(_obs(_board(avatar=(3, 4)), levels=1, actions=(1, 2)))  # door gone
        assert agent._det.door == 9

    def test_no_motion_learned_across_death_respawn(self):
        # A respawn frame is not a real successor — motion across the death jump
        # must not pollute the detector's move map.
        agent = ExplorerArcAgi3Agent()
        agent.act(_obs(_board(avatar=(2, 2)), actions=(1, 2)))
        agent.act(_obs(_board(avatar=(8, 8)), actions=(1, 2), terminal=True))
        assert agent._det._deltas == {}


_LATTICE = {
    1: Counter({(1, 0): 2}),  # down
    2: Counter({(-1, 0): 2}),  # up
    3: Counter({(0, 1): 2}),  # right
    4: Counter({(0, -1): 2}),  # left
}


def _seed_lattice(agent):
    """Pre-induce avatar(12) + a full 4-direction move lattice, but NO goal yet —
    the state a directed bootstrap starts from (it can move, but hasn't won)."""
    agent._det._avatar = 12
    # Deep-copy the Counters: act() mutates them in place, so a shared seed would
    # leak counts across tests.
    agent._det._deltas = {12: {a: Counter(c) for a, c in _LATTICE.items()}}


def _seed_nav(agent, door=9):
    """Pre-induce avatar(12), a full 4-direction move lattice, and a door value."""
    _seed_lattice(agent)
    agent._det._door = door


class TestNavExploit:
    def test_navigates_toward_induced_door(self):
        # Door(9) sits 4 cells to the right; the Planner's first move is right (3),
        # not blind exploration.
        agent = ExplorerArcAgi3Agent()
        _seed_nav(agent)
        board = _board(avatar=(2, 2), extra={9: [(2, 6)]})
        assert agent.act(_obs(board, actions=(1, 2, 3, 4))).id == 3

    def test_routes_around_blocked_cell(self):
        # A wall on the cell directly right forces a detour — never step 3 into it.
        agent = ExplorerArcAgi3Agent()
        _seed_nav(agent)
        agent._blocked.add((2, 3))
        board = _board(avatar=(2, 2), extra={9: [(2, 6)]})
        assert agent.act(_obs(board, actions=(1, 2, 3, 4))).id != 3

    def test_refused_move_records_blocked_cell(self):
        # The nav move steps right but the avatar does not budge → that destination
        # cell is an (empirically learned) wall.
        agent = ExplorerArcAgi3Agent()
        _seed_nav(agent)
        board = _board(avatar=(2, 2), extra={9: [(2, 6)]})
        agent.act(_obs(board, actions=(1, 2, 3, 4)))  # emits right (3)
        agent.act(_obs(board, actions=(1, 2, 3, 4)))  # avatar stayed → refused
        assert (2, 3) in agent._blocked


class TestDirectedBootstrap:
    """Before any win, steer toward salient objects so the FIRST level-up is
    sought, not stumbled into — the fix for the cold-start bootstrap gap."""

    def test_seeks_nearest_affordance_before_any_win(self):
        # No goal induced yet, but avatar+lattice are known: head toward the only
        # salient object (4 cells right), not blind exploration (which picks down).
        agent = ExplorerArcAgi3Agent()
        _seed_lattice(agent)
        board = _board(avatar=(2, 2), extra={5: [(2, 6)]})
        assert agent.act(_obs(board, actions=(1, 2, 3, 4))).id == 3

    def test_steps_directly_onto_adjacent_affordance(self):
        # The Planner stops one cell short; an adjacent object must still be entered
        # (the pickup), so the move that lands exactly on it is emitted.
        agent = ExplorerArcAgi3Agent()
        _seed_lattice(agent)
        board = _board(avatar=(2, 2), extra={5: [(2, 3)]})
        assert agent.act(_obs(board, actions=(1, 2, 3, 4))).id == 3

    def test_skips_the_already_induced_door(self):
        # Once the door is induced it is _nav_move's job; affordance-seeking must
        # skip it (else the two fight). With the door the only object, affordance
        # has nothing to seek and _nav_move drives the step instead.
        agent = ExplorerArcAgi3Agent()
        _seed_nav(agent)  # door=9 induced
        board = _board(avatar=(2, 2), extra={9: [(2, 6)]})
        lattice = agent._det.move_lattice()
        assert agent._nav_affordance(board, [1, 2, 3, 4], lattice) is None  # skipped
        assert agent.act(_obs(board, actions=(1, 2, 3, 4))).id == 3  # _nav_move drives

    def test_key_pickup_clears_blocked_walls(self):
        # A pickup may unlock a previously-refused door, so stale wall memory is
        # dropped when a new key is learned.
        agent = ExplorerArcAgi3Agent()
        _seed_lattice(agent)
        agent._blocked.add((7, 7))
        agent.act(
            _obs(_board(avatar=(2, 2), extra={5: [(2, 3)]}), actions=(1, 2, 3, 4))
        )
        agent.act(_obs(_board(avatar=(2, 3)), actions=(1, 2, 3, 4)))  # key 5 collected
        assert 5 in agent._det.keys and agent._blocked == set()


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


class _Ls20Sim:
    """Minimal ls20-like navigate game: the avatar (12) walks to a door (9);
    reaching it clears the door for one frame (value vanishes → induction fires),
    then loads the next level. Exposes only pixels — no semantics."""

    MOVES = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}

    def __init__(self, doors: list[tuple[int, int]], size: int = 8) -> None:
        self.doors = doors
        self.size = size
        self.levels = 0
        self.avatar = (1, 1)
        self.door: tuple[int, int] | None = doors[0]

    def obs(self) -> dict:
        g = np.full((self.size, self.size), 3, dtype=int)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
        if self.door is not None:
            g[self.door] = 9
        g[self.avatar] = 12
        return {
            "frame": [g.tolist()],
            "available_actions": [1, 2, 3, 4],
            "levels_completed": self.levels,
            "state": "NOT_FINISHED",
        }

    def step(self, action: int) -> None:
        if self.door is None:  # the cleared frame was seen → spawn the next level
            self.avatar = (1, 1)
            self.door = (
                self.doors[self.levels] if self.levels < len(self.doors) else None
            )
            return
        d = self.MOVES.get(action)
        if d is None:
            return
        nr, nc = self.avatar[0] + d[0], self.avatar[1] + d[1]
        if not (0 < nr < self.size - 1 and 0 < nc < self.size - 1):
            return  # wall — move refused
        self.avatar = (nr, nc)
        if (nr, nc) == self.door:
            self.door = None  # door consumed; value 9 momentarily vanishes
            self.levels += 1


class TestLs20WithoutHardcodedSemantics:
    """The Phase-3 generalization proof: a fresh explorer carrying NO LS20 colour
    prior solves a multi-level navigate game purely from induced avatar + door."""

    def test_explore_then_exploit_solves_every_level(self):
        sim = _Ls20Sim([(6, 6), (1, 6), (6, 1)])
        agent = ExplorerArcAgi3Agent()
        cleared: dict[int, int] = {}
        for s in range(400):
            prev = sim.levels
            sim.step(agent.act(sim.obs()).id)
            if sim.levels > prev:
                cleared[sim.levels] = s
            if sim.levels == len(sim.doors):
                break

        assert sim.levels == 3  # all levels solved, no hardcoded semantics
        assert agent._det.avatar == 12 and agent._det.door == 9  # induced from pixels
        # Level 1 bootstraps from a cold start (pin avatar, then seek the goal);
        # later levels exploit the induced goal directly, so each costs fewer steps.
        assert cleared[2] - cleared[1] < cleared[1]


class _Ls20LockSim:
    """Minimal LockSmith-like game: the avatar (12) must collect a key (5) before
    the door (9) will open. Stepping onto a locked door is refused; once the key is
    held, reaching the door clears the level (both values vanish for one frame, so
    key- and door-induction fire). Exposes only pixels — no semantics."""

    MOVES = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}

    def __init__(self, levels: list[tuple], size: int = 8) -> None:
        self._spec = levels  # [(key_cell, door_cell), ...]
        self.size = size
        self.levels = 0
        self._load(0)

    def _load(self, i: int) -> None:
        self.avatar = (1, 1)
        self.key, self.door = self._spec[i] if i < len(self._spec) else (None, None)

    def obs(self) -> dict:
        g = np.full((self.size, self.size), 3, dtype=int)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
        if self.key is not None:
            g[self.key] = 5
        if self.door is not None:
            g[self.door] = 9
        g[self.avatar] = 12
        return {
            "frame": [g.tolist()],
            "available_actions": [1, 2, 3, 4],
            "levels_completed": self.levels,
            "state": "NOT_FINISHED",
        }

    def step(self, action: int) -> None:
        if self.door is None and self.key is None:  # cleared frame seen → next level
            self._load(self.levels)
            return
        d = self.MOVES.get(action)
        if d is None:
            return
        nr, nc = self.avatar[0] + d[0], self.avatar[1] + d[1]
        if not (0 < nr < self.size - 1 and 0 < nc < self.size - 1):
            return  # border wall — refused
        if (nr, nc) == self.door and self.key is not None:
            return  # door locked until the key is collected — refused
        self.avatar = (nr, nc)
        if (nr, nc) == self.key:
            self.key = None  # key collected; value 5 momentarily vanishes
        elif (nr, nc) == self.door:
            self.door = None  # door consumed; value 9 vanishes
            self.levels += 1


class TestDirectedLockBootstrap:
    """The cold-start fix: a fresh explorer with NO hardcoded semantics manufactures
    its first win by seeking the key→door affordance, then exploits on later levels.
    Sized so blind exploration (cost ∝ area) cannot finish in budget but directed
    seeking (cost ∝ path length) can — the fix is load-bearing, not incidental."""

    def _solve(self, blind: bool, budget: int):
        size, m = 20, 18
        sim = _Ls20LockSim([((1, 2), (1, m)), ((m, 2), (m, m))], size=size)
        agent = ExplorerArcAgi3Agent()
        if blind:
            agent._nav_affordance = lambda *a, **k: None  # disable the bootstrap
        for s in range(budget):
            sim.step(agent.act(sim.obs()).id)
            if sim.levels == 2:
                return s + 1, agent
        return None, agent

    def test_directed_bootstrap_solves_a_large_locked_game_blind_cannot(self):
        budget = 250  # directed solves in ~58 steps; blind needs ~668
        steps, agent = self._solve(blind=False, budget=budget)
        blind_steps, _ = self._solve(blind=True, budget=budget)
        assert steps is not None  # both locked levels solved by directed bootstrap
        assert blind_steps is None  # blind exploration cannot, in the same budget
        assert agent._det.door == 9  # door induced from the first directed win
        assert 5 in agent._det.keys  # key affordance learned while seeking


class _StridePhantomSim:
    """A stride-5 navigate game seeded with off-stride 'phantom' decoys (value 7)
    that affordance-seeking can *approach* but never *reach*: the move lattice steps
    5 cells, the decoys sit 2 off it. Two decoys straddle the avatar so the nearest
    flips every step — a steering limit cycle. The real key(5)→door(9) is reachable
    on-stride to the side, so escaping requires ignoring the unreachable decoys.
    Reproduces the live ls20 failure the Phase 5 telemetry diagnostic localised."""

    MOVES = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}

    def __init__(self, size: int = 39) -> None:
        self.size = size
        self.levels = 0
        self._load()

    def _load(self) -> None:
        self.avatar = (15, 15)  # ≡0 mod 5 → reachable lattice
        self.phantoms = [(13, 15), (18, 15)]  # ≡3 mod 5 → off-stride, straddle avatar
        self.key: tuple[int, int] | None = (15, 25)
        self.door: tuple[int, int] | None = (15, 30)
        self._havekey = False

    def obs(self) -> dict:
        g = np.full((self.size, self.size), 3, dtype=int)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
        for p in self.phantoms:
            g[p] = 7
        if not self._havekey and self.key is not None:
            g[self.key] = 5
        if self.door is not None:
            g[self.door] = 9
        g[self.avatar] = 12
        return {
            "frame": [g.tolist()],
            "available_actions": [1, 2, 3, 4],
            "levels_completed": self.levels,
            "state": "NOT_FINISHED",
        }

    def step(self, action: int) -> None:
        if self.door is None and self._havekey:  # cleared frame seen → next level
            self.levels += 1
            self._load()
            return
        d = self.MOVES.get(action)
        if d is None:
            return
        nr, nc = self.avatar[0] + d[0], self.avatar[1] + d[1]
        if not (0 < nr < self.size - 1 and 0 < nc < self.size - 1):
            return  # border wall — refused
        if (nr, nc) == self.door and not self._havekey:
            return  # door locked until the key is collected
        self.avatar = (nr, nc)
        if (nr, nc) == self.key:
            self._havekey = True
            self.key = None
        elif (nr, nc) == self.door:
            self.door = None


class TestAffordancePhantomEscape:
    """Phase 6: affordance-seeking must skip unreachable off-stride targets rather
    than oscillate between them — the live ls20 limit cycle (Phase 5 diagnostic)."""

    def test_explorer_escapes_phantom_cycle_and_solves(self):
        sim = _StridePhantomSim()
        agent = ExplorerArcAgi3Agent()
        for _ in range(120):
            sim.step(agent.act(sim.obs()).id)
            if sim.levels == 2:
                break
        assert sim.levels == 2  # escaped the decoy cycle, solved key→door both levels


class _StraddleDecoySim:
    """Two reachable decoys (7) straddle the avatar, and a far per-frame blinker cell
    (3↔6) makes every frame a new signature at a repeated position — defeating
    signature-keyed memory exactly as real ls20 does (live: 741 signatures, 30 avatar
    cells). Greedy nearest-target affordance ping-pongs between the decoys (stepping
    onto one occludes it, promoting the other) — an up↔down cycle in avatar-POSITION
    space that only position-keyed memory can break. The door (9) sits straight up past
    the cycle, so only an avatar that breaks out reaches it."""

    MOVES = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}

    def __init__(self, size: int = 12) -> None:
        self.size = size
        self.levels = 0
        self._blink = False
        self._load()

    def _load(self) -> None:
        self.avatar = (5, 2)
        self.decoys = [(4, 2), (6, 2)]  # straddle the avatar vertically, adjacent
        # The door sits up the straddle axis, past the upper decoy: a trapped avatar
        # ping-ponging in place never advances toward it; one that refuses to step
        # back walks straight up to it (as on real ls20, where breaking the cycle let
        # the frontier explorer traverse and clear a level).
        self.door = (2, 2) if self.levels < 2 else None

    def obs(self) -> dict:
        g = np.full((self.size, self.size), 3, dtype=int)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
        g[1, 9] = 6 if self._blink else 3  # far per-frame blinker → signature churn
        for d in self.decoys:
            g[d] = 7
        if self.door is not None:
            g[self.door] = 9
        g[self.avatar] = 12
        return {
            "frame": [g.tolist()],
            "available_actions": [1, 2, 3, 4],
            "levels_completed": self.levels,
            "state": "NOT_FINISHED",
        }

    def step(self, action: int) -> None:
        self._blink = not self._blink  # churn the signature every step
        if self.door is None:  # cleared frame seen → next level
            self._load()
            return
        d = self.MOVES.get(action)
        if d is None:
            return
        nr, nc = self.avatar[0] + d[0], self.avatar[1] + d[1]
        if not (0 < nr < self.size - 1 and 0 < nc < self.size - 1):
            return  # border wall — refused
        self.avatar = (nr, nc)
        if (nr, nc) == self.door:
            self.door = None  # reached → level clears, value 9 vanishes for a frame
            self.levels += 1


class TestAffordanceEscapesPositionCycle:
    """Affordance won't step the avatar onto a recently-occupied cell, so it escapes a
    position-space cycle even when per-frame churn keeps every signature distinct."""

    def test_explorer_escapes_position_cycle_and_solves(self):
        sim = _StraddleDecoySim()
        agent = ExplorerArcAgi3Agent()
        for _ in range(150):
            sim.step(agent.act(sim.obs()).id)
            if sim.levels == 2:
                break
        assert sim.levels == 2  # broke the position cycle and reached the door twice


class TestTrace:
    def test_probe_branch_tagged_first(self):
        # First directional step seeds the lattice → the probe branch fires.
        agent = ExplorerArcAgi3Agent()
        agent.act(_obs(_board(avatar=(2, 2)), actions=(1, 2, 3, 4)))
        assert agent.trace["branch"] == "probe"
        assert agent.trace["step"] == 1
        assert agent.trace["avatar"] is None  # not yet pinned on first frame

    def test_choose_branch_when_no_induction(self):
        # Click-only game: no avatar/lattice ever, so selection falls to frontier.
        agent = ExplorerArcAgi3Agent()
        board = _board(extra={5: [(4, 4)]})
        agent.act(_obs(board, actions=(6,)))
        assert agent.trace["branch"] == "choose"
        assert agent.trace["lattice_size"] == 0
