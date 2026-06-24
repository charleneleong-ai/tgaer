"""Game-agnostic explorer for ARC-AGI-3 — the explore→induce→exploit controller.

Unlike the LS20-tuned ``KeyDoorController``, this agent assumes nothing about a
game's win condition. It builds a directed **state graph** of frame signatures
and drives **frontier-directed exploration**: at each step it takes an untested
action from the current state, or routes along known edges to the nearest state
that still has one. This is the training-free, no-LLM spine that topped the
ARC-AGI-3 2025 preview (algorithmic exploration ≫ frontier LLM ≫ random).

Win-induction is wired for both verbs. Phase 3a (click): when a click advances a
level, the clicked cell's value is learned and re-clicked first on later levels.
Phase 3b (navigate): an ``EmpiricalSemantics`` detector induces the avatar (by
controllability), its move lattice, and the goal value that vanishes under the
avatar on a level-up; once known, ``_nav_move`` BFS-plans to the nearest goal
cell via the ``Planner``, with refused moves recorded as walls (no hardcoded
colours). This re-solves multi-level ``ls20`` without the LS20 semantics prior.

Phase 4 (directed bootstrap) closes the cold-start gap: induction can only fire
*after* a first win, so blind exploration must manufacture one — infeasible on a
real 64×64 grid. Once the avatar and its lattice are induced (which needs only
controllability, not a win), ``_nav_affordance`` steers toward the nearest salient
object — a candidate key/door — so the first win is *sought*, not stumbled into;
its cost is then path-length (linear), not area (quadratic). ``_probe_moves`` seeds
the full lattice first so directed routing never oscillates on a partial one.

Action primitives generalise across verbs: simple actions are ``("act", id)``;
ACTION6 becomes salience-ranked ``("click", row, col)`` targets at component
centroids, so the click-only games the navigate planner cannot touch still get
explored.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np

from tgaer.agents.arc_agi3_grid import Planner, cells, components, field_box, in_field
from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics
from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import COMPLEX_ACTION_ID, ArcAction

# A primitive is the atomic unit of exploration: ("act", id) or ("click", r, c).
Primitive = tuple

_MOVES = (1, 2, 3, 4)  # directional action ids — the moves a lattice is built from


def frame_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Hashable identity of the *play-field* — the region inside the green box,
    so HUD / status-bar churn outside it doesn't fragment the state graph.

    Field detection (``field_box``) keys on the green floor; a later phase should
    replace it with a colour-free field detector for games without one."""
    lo, hi = field_box(arr)
    r0, c0, r1, c1 = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
    sub = arr[r0 : r1 + 1, c0 : c1 + 1]
    return sub.shape, sub.tobytes()


def _background(arr: np.ndarray, box) -> int:
    """The most common in-field cell value — treated as floor and not a target."""
    lo, hi = box
    sub = arr[int(lo[0]) : int(hi[0]) + 1, int(lo[1]) : int(hi[1]) + 1]
    return int(Counter(sub.ravel().tolist()).most_common(1)[0][0]) if sub.size else 0


def _centroid(comp: np.ndarray) -> tuple[int, int]:
    return int(round(comp[:, 0].mean())), int(round(comp[:, 1].mean()))


def click_targets(
    arr: np.ndarray, k: int = 12, max_field_frac: float = 0.25
) -> list[tuple[int, int]]:
    """Salience-ranked click points: centroids of in-field, single-colour,
    non-background components, largest compact object first, capped at ``k``.
    Components spanning more than ``max_field_frac`` of the field are treated as
    structure (walls / floor), not buttons, and dropped.

    The size cutoff is a Phase-1 heuristic; Phase 2 (action-effect classification)
    replaces it with empirical "does clicking here change the frame?" filtering."""
    box = field_box(arr)
    lo, hi = box
    field_area = max(1.0, float((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1)))
    bg = _background(arr, box)
    scored: list[tuple[int, int, int]] = []
    for v in (int(x) for x in np.unique(arr)):
        if v == bg:
            continue
        for c in components(arr, (v,)):  # one colour at a time — never merge objects
            if not in_field(c.mean(0), box) or len(c) > max_field_frac * field_area:
                continue
            cr, cc = _centroid(c)
            if arr[cr, cc] != v:  # centroid off the component → a hollow frame/ring
                continue  # (e.g. the wall border), not a clickable object
            scored.append((len(c), cr, cc))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(r, c) for _, r, c in scored[:k]]


def goal_targets(arr: np.ndarray, goal_values) -> list[tuple[int, int]]:
    """Centroids of components whose value is a learned goal — a click here has
    previously advanced a level, so it is worth trying before blind salience."""
    out: list[tuple[int, int]] = []
    for v in goal_values:
        out.extend(_centroid(comp) for comp in components(arr, (v,)))
    return out


def proposals(arr: np.ndarray, available: list[int], goal_values=()) -> list[Primitive]:
    """Ordered action primitives to try at the current frame. Clicks on learned
    goal values come first; then ACTION6 fans out into salience-ranked click
    targets; every other id is a simple ``act``. Duplicates are dropped."""
    prims: list[Primitive] = []
    if COMPLEX_ACTION_ID in available and goal_values:
        prims.extend(("click", r, c) for r, c in goal_targets(arr, goal_values))
    for a in available:
        if a == COMPLEX_ACTION_ID:
            prims.extend(("click", r, c) for r, c in click_targets(arr))
        else:
            prims.append(("act", a))
    seen: set[Primitive] = set()
    return [p for p in prims if not (p in seen or seen.add(p))]


def to_arc(prim: Primitive) -> ArcAction:
    if prim[0] == "click":
        return ArcAction(id=COMPLEX_ACTION_ID, x=prim[2], y=prim[1])  # x=col, y=row
    return ArcAction(id=prim[1])


class StateGraph:
    """Directed graph of frame signatures. Nodes carry their untested primitives;
    edges record observed ``(signature, primitive) -> signature`` transitions."""

    def __init__(self) -> None:
        self._untested: dict[Any, list[Primitive]] = {}
        self._adj: dict[Any, list[tuple[Primitive, Any]]] = {}

    def register(self, sig: Any, prims: list[Primitive]) -> None:
        """Record a node's untested primitives on first sighting only — a
        re-sighting must never resurrect primitives already taken."""
        if sig not in self._untested:
            self._untested[sig] = list(prims)  # _adj is filled lazily by connect()

    def take(self, sig: Any, prim: Primitive) -> None:
        rest = self._untested.get(sig)
        if rest and prim in rest:
            rest.remove(prim)

    def connect(self, src: Any, prim: Primitive, dst: Any) -> None:
        edges = self._adj.setdefault(src, [])
        if (prim, dst) not in edges:
            edges.append((prim, dst))

    def untested_at(self, sig: Any) -> list[Primitive]:
        return self._untested.get(sig, [])

    def path_to_frontier(self, start: Any) -> list[Primitive] | None:
        """Primitive sequence from ``start`` to the nearest node with untested
        primitives. ``[]`` if ``start`` itself is a frontier; ``None`` if no
        frontier is reachable over known edges."""
        if self.untested_at(start):
            return []
        prev: dict[Any, tuple[Any, Primitive]] = {}
        seen, q = {start}, deque([start])
        while q:
            node = q.popleft()
            for prim, dst in self._adj.get(node, []):
                if dst in seen:
                    continue
                seen.add(dst)
                prev[dst] = (node, prim)
                if self.untested_at(dst):
                    return self._trace(prev, start, dst)
                q.append(dst)
        return None

    @staticmethod
    def _trace(prev, start, node) -> list[Primitive]:
        path: list[Primitive] = []
        while node != start:
            node, prim = prev[node]
            path.append(prim)
        return path[::-1]


class ExplorerArcAgi3Agent(Agent):
    """Frontier-directed explorer. Per level: take an untested primitive at the
    current state, else follow known edges to the nearest state that has one."""

    def __init__(self, seed: int = 0, **_: Any) -> None:
        self._graph = StateGraph()
        self._plan: deque[Primitive] = deque()
        self._prev_sig: Any | None = None
        self._prev_prim: Primitive | None = None
        self._levels = 0
        # Edges that led to GAME_OVER, persistent across deaths and level resets.
        self._fatal: set[tuple[Any, Primitive]] = set()
        # Cell values whose click advanced a level — persistent goal prior.
        self._goal_values: set[int] = set()
        self._prev_arr: np.ndarray | None = None
        # Induces the avatar (controllability), its move lattice, and the navigate
        # goal (value that vanishes under the avatar on a level-up). Persists across
        # level resets — induced roles are the cross-level transfer.
        self._det = EmpiricalSemantics()
        # Cells where a lattice move was refused — emergent walls, no colour prior.
        self._blocked: set[tuple[int, int]] = set()
        # Key count last seen — a fresh pickup may unlock a previously-refused door.
        self._prev_key_n = 0
        # Directional actions already probed once to seed the avatar's move lattice.
        self._probed: set[int] = set()
        self.last_reply: str | None = None

    def _on_new_level(self) -> None:
        self._graph = StateGraph()
        self._plan.clear()
        self._prev_sig = None
        self._prev_prim = None

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        frame = obs.get("frame") or []
        available = obs.get("available_actions") or [1]
        if not frame:
            return to_arc(("act", available[0]))
        arr = np.asarray(frame[-1])
        levels = obs.get("levels_completed", self._levels)

        # Learn avatar / move-lattice / navigate-goal from the prior in-level
        # transition; a death respawn is not a real successor, so skip it.
        learning = (
            not obs.get("terminal")
            and self._prev_arr is not None
            and self._prev_prim is not None
        )
        if learning:
            self._det.observe(self._prev_arr, to_arc(self._prev_prim).id, arr, levels)
            if len(self._det.keys) > self._prev_key_n:  # a pickup may unlock a door
                self._blocked.clear()
        self._prev_key_n = len(self._det.keys)
        lattice = self._det.move_lattice()  # once per step, after the observe update
        if learning:
            self._learn_blocked(arr, lattice)

        # A respawn after death: the action that led here was fatal. Record the
        # edge so it is never repeated, and drop the cross-death link — the frame
        # is a fresh level start, not a normal successor.
        if obs.get("terminal") and self._prev_sig is not None and self._prev_prim:
            self._fatal.add((self._prev_sig, self._prev_prim))
            self._graph.take(self._prev_sig, self._prev_prim)
            self._prev_sig = self._prev_prim = None
        if levels > self._levels:  # genuine progress wipes the per-level map; a
            self._induce_goal()  # but first learn what the winning click targeted
            self._on_new_level()  # death respawn (levels drop) must keep the map
        self._levels = levels

        sig = frame_signature(arr)
        prims = proposals(arr, available, self._goal_values)
        self._graph.register(sig, prims)
        if self._prev_sig is not None and self._prev_prim is not None:
            self._graph.connect(self._prev_sig, self._prev_prim, sig)

        prim = (
            self._probe_moves(available, lattice)  # learn each move's effect first
            or self._nav_affordance(arr, available, lattice)  # directed bootstrap: seek
            or self._nav_move(arr, available, lattice)  # exploit the induced goal
            or self._choose(sig, prims)  # blind frontier exploration
        )
        self._graph.take(sig, prim)
        self._prev_sig, self._prev_prim = sig, prim
        self._prev_arr = arr
        self.last_reply = f"[explorer] {prim}"
        return to_arc(prim)

    def _induce_goal(self) -> None:
        """A level was just completed: if the preceding action clicked a cell,
        learn that cell's value as a goal to re-click on later levels."""
        if (
            self._prev_arr is not None
            and self._prev_prim
            and self._prev_prim[0] == "click"
        ):
            _, r, c = self._prev_prim
            self._goal_values.add(int(self._prev_arr[r, c]))

    def _learn_blocked(self, arr: np.ndarray, lattice: dict[int, np.ndarray]) -> None:
        """A directional move the lattice expected to shift the avatar, but which
        left it put, means the destination cell is a wall — record it so the
        Planner routes around it without any hardcoded wall colours."""
        avatar = self._det.avatar
        if avatar is None or self._prev_arr is None or self._prev_prim[0] != "act":
            return
        d = lattice.get(self._prev_prim[1])
        prev_av, cur_av = cells(self._prev_arr, avatar), cells(arr, avatar)
        if d is None or not len(prev_av) or not len(cur_av):
            return
        if (cur_av.min(0) == prev_av.min(0)).all():  # refused: avatar did not move
            cell = prev_av.min(0) + d
            self._blocked.add((int(cell[0]), int(cell[1])))

    def _probe_moves(
        self, available: list[int], lattice: dict[int, np.ndarray]
    ) -> Primitive | None:
        """Bootstrap: take each directional action once so the avatar's move lattice
        is complete before directed routing relies on it (a partial lattice makes the
        router oscillate). Skip a move whose effect is already known or once tried."""
        for a in available:
            if a in _MOVES and a not in lattice and a not in self._probed:
                self._probed.add(a)
                return ("act", a)
        return None

    def _route(
        self,
        arr: np.ndarray,
        available: list[int],
        lattice: dict[int, np.ndarray],
        av: np.ndarray,
        goal: np.ndarray,
    ) -> Primitive | None:
        """First move that carries the avatar toward ``goal``: step straight onto an
        adjacent goal (the Planner stops a cell short, but a pickup / door-entry must
        actually land), else BFS-plan around known walls. ``None`` if no usable move."""
        tl = av.min(0)
        if int(abs(goal - tl).sum()) == 1:  # adjacent → step straight on
            for a, d in lattice.items():
                if a in available and (tl + d == goal).all():
                    return ("act", a)
        planner = Planner(arr, (av - tl).astype(int), lattice, walls=())
        planner.blocked = self._blocked
        path = planner.path(tl, goal)
        if path and path[0] in available:
            return ("act", path[0])
        return None

    def _nav_move(
        self, arr: np.ndarray, available: list[int], lattice: dict[int, np.ndarray]
    ) -> Primitive | None:
        """Exploit: once the avatar, its move lattice, and the navigate goal (door)
        are induced, route to the nearest goal cell. ``None`` whenever the goal isn't
        known or no path exists — the caller falls back to frontier exploration."""
        avatar, door = self._det.avatar, self._det.door
        if avatar is None or door is None:
            return None
        av, goals = cells(arr, avatar), cells(arr, door)
        if not lattice or not len(av) or not len(goals):
            return None
        tl = av.min(0)
        goal = min(goals, key=lambda g: int(abs(g - tl).sum()))
        return self._route(arr, available, lattice, av, goal)

    def _nav_affordance(
        self, arr: np.ndarray, available: list[int], lattice: dict[int, np.ndarray]
    ) -> Primitive | None:
        """Directed bootstrap: before the goal is induced, steer toward the nearest
        salient object — a candidate key/door — so the *first* win is sought rather
        than stumbled into. Targets exclude the avatar, the already-induced door
        (the real exploit drives that), and known walls. ``None`` when the avatar or
        its lattice isn't known yet, or nothing reachable remains."""
        avatar = self._det.avatar
        if avatar is None or not lattice:
            return None
        av = cells(arr, avatar)
        if not len(av):
            return None
        door = self._det.door
        skip = {tuple(c) for c in cells(arr, door)} if door is not None else set()
        skip |= self._blocked
        tl = av.min(0)
        targets = [
            np.array(t)
            for t in click_targets(arr)
            if arr[t] != avatar and t not in skip
        ]
        for goal in sorted(targets, key=lambda g: int(abs(g - tl).sum())):
            if move := self._route(arr, available, lattice, av, goal):
                return move
        return None

    def _choose(self, sig: Any, prims: list[Primitive]) -> Primitive:
        # Drop a stale route the current frame can no longer execute.
        if self._plan and self._plan[0] not in set(prims):
            self._plan.clear()
        if self._plan:
            return self._plan.popleft()
        # Untested prims are fatal-free by construction (fatal edges are taken out
        # of untested when recorded), so only this last-resort reuse needs to screen
        # them: prefer a primitive not known to end the game.
        if untested := self._graph.untested_at(sig):
            return untested[0]
        path = self._graph.path_to_frontier(sig)
        if path:
            self._plan = deque(path)
            return self._plan.popleft()
        safe = [p for p in prims if (sig, p) not in self._fatal]
        if safe:
            return safe[0]
        return prims[0] if prims else ("act", 1)
