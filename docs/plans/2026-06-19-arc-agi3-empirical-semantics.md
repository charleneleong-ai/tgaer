# ARC-AGI-3 Empirical Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect a grid game's avatar / key / door roles from the action→observation stream the agent already produces, so the key→door controller gets correct semantics on games where the LS20-hardcoded prior and the mislabel-prone VL scientist both fail.

**Architecture:** A pure, stateful detector `EmpiricalSemantics` ingests `(prev_frame, action, cur_frame, levels_completed)` each step and pins roles from evidence — avatar by action-conditioned translation (controllability), key by disappearance-on-contact, door by level-increment-on-contact. A new `EmpiricalPlannerAgent` drives the existing `KeyDoorController` with `detector.semantics(cold_start)`, where the cold-start is an injected VL `Scientist` (optional) falling back to `LS20_DEFAULT`. Evidence-pinned roles always override the cold-start.

**Tech Stack:** Python 3.11, numpy, pytest. Reuses `arc_agi3_grid` geometry helpers and `KeyDoorController`. ruff 0.15.13 lint/format gate.

## Global Constraints

- Python 3.11+, modern syntax; type hints on all function signatures; no redundant docstrings.
- Detector lives in its own module `src/tgaer/agents/arc_agi3_semantics.py`; it imports geometry helpers from `arc_agi3_grid` and holds **no** env or model I/O.
- `planner` and `scientist` agent kinds must remain untouched and behaviourally unchanged.
- Tests: one area file `tests/test_arc_agi3_empirical.py`, split into classes per sub-feature; module-level picklable helpers (no lambdas); parametrize near-duplicates.
- Lint gate before every push: `ruff check <files>` AND `ruff format --check <files>` (ruff 0.15.13).
- Conventional commits; no `Co-Authored-By` trailers. Branch `feat/arc-agi3-empirical-semantics` (already created, spec committed at `aa02b15`).
- Run tests with the repo venv: `cd /workspace/tgaer && .venv/bin/python -m pytest`.

## File Structure

- **Create** `src/tgaer/agents/arc_agi3_semantics.py` — `EmpiricalSemantics` detector (Tasks 1–3).
- **Create** `src/tgaer/agents/arc_agi3_empirical.py` — `EmpiricalPlannerAgent` (Task 4).
- **Create** `configs/experiments/arc_agi3_empirical.yaml` — `empirical` experiment config (Task 5).
- **Modify** `src/tgaer/evaluation/dispatch.py` — register `"empirical"` kind (Task 5).
- **Create** `tests/test_arc_agi3_empirical.py` — all tests (Tasks 1–4).

### Reused interfaces (already exist — exact signatures)

From `tgaer.agents.arc_agi3_grid`:
- `components(arr: np.ndarray, values: tuple[int, ...]) -> list[np.ndarray]` — each element is an `(N,2)` array of `(row,col)` cells of one connected component.
- `field_box(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(lo, hi)` corners from GREEN(3) cells.
- `in_field(centroid: np.ndarray, box, pad: int = 4) -> bool`.
- `cells(arr: np.ndarray, v: int) -> np.ndarray` — all `(row,col)` of value `v`.
- `Semantics(avatar: int, keys: tuple[int, ...], door: int, walls: tuple[int, ...], verb: str)` — frozen dataclass.
- `LS20_DEFAULT = Semantics(avatar=12, keys=(0, 1), door=9, walls=(4, 11), verb="navigate")`.
- `KeyDoorController()` with `.learn(arr, sem) -> None`, `.step(arr, sem, avail: list[int]) -> int`, `.on_new_level() -> None`, `.made_progress() -> bool`, `.delta: dict[int, np.ndarray]`.
- `to_action(action_id: int) -> ArcAction`.

From `tgaer.agents.arc_agi3_scientist`: `Scientist(model=..., api_base=...)` with `.infer(frame) -> Semantics | None`.
From `tgaer.core.agent_base`: `Agent` base with `act(observation) -> ArcAction`.
Observation dict shape: `{"frame": [grid, ...]}` (last is current), `"available_actions": list[int]`, `"levels_completed": int`, `"state": str`.

---

### Task 1: Avatar detection by action-conditioned translation

**Files:**
- Create: `src/tgaer/agents/arc_agi3_semantics.py`
- Test: `tests/test_arc_agi3_empirical.py`

**Interfaces:**
- Consumes: `components`, `field_box`, `in_field` from `arc_agi3_grid`.
- Produces:
  - `EmpiricalSemantics(avatar_confirm: int = 2)` — detector.
  - `EmpiricalSemantics.observe(prev: np.ndarray | None, action: int | None, cur: np.ndarray, levels: int) -> None`.
  - `EmpiricalSemantics.avatar -> int | None` (property; pinned avatar value or `None`).
  - Internal: `_single_in_field_centroid(arr, v) -> np.ndarray | None` (one in-field component of `v`, else None).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arc_agi3_empirical.py
from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics


def _grid(avatar_rc: tuple[int, int], avatar: int = 12) -> np.ndarray:
    """10x10 green field, yellow(4) wall border, one avatar cell."""
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    g[avatar_rc] = avatar
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py::TestAvatarDetection -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tgaer.agents.arc_agi3_semantics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tgaer/agents/arc_agi3_semantics.py
from __future__ import annotations

from collections import Counter

import numpy as np

from tgaer.agents.arc_agi3_grid import components, field_box, in_field

Delta = tuple[int, int]


class EmpiricalSemantics:
    """Derive avatar/key/door roles from the action→observation stream.

    The avatar is the value whose motion is *controllable* — a consistent,
    action-specific translation; a distractor that moves the same way regardless
    of action is not controllable and is never pinned.
    """

    def __init__(self, avatar_confirm: int = 2) -> None:
        self._avatar_confirm = avatar_confirm
        self._avatar: int | None = None
        # value -> action -> Counter of observed integer Δ tuples
        self._deltas: dict[int, dict[int, Counter[Delta]]] = {}

    @property
    def avatar(self) -> int | None:
        return self._avatar

    def observe(
        self, prev: np.ndarray | None, action: int | None, cur: np.ndarray, levels: int
    ) -> None:
        if prev is not None and action is not None:
            self._observe_motion(prev, action, cur)

    def _observe_motion(self, prev: np.ndarray, action: int, cur: np.ndarray) -> None:
        if self._avatar is not None:
            return
        shared = set(np.unique(prev)).intersection(np.unique(cur))
        for v in shared:
            p = _single_in_field_centroid(prev, int(v))
            c = _single_in_field_centroid(cur, int(v))
            if p is None or c is None:
                continue
            delta = (int(round(c[0] - p[0])), int(round(c[1] - p[1])))
            self._deltas.setdefault(int(v), {}).setdefault(action, Counter())[delta] += 1
            if self._is_controllable(int(v)):
                self._avatar = int(v)
                return

    def _is_controllable(self, v: int) -> bool:
        per_action = self._deltas[v]
        # consistency: some action has the SAME non-zero Δ at least avatar_confirm times
        consistent = any(
            d != (0, 0) and n >= self._avatar_confirm
            for counter in per_action.values()
            for d, n in counter.items()
        )
        # controllability: the per-action majority Δ takes ≥2 distinct values
        majorities = {counter.most_common(1)[0][0] for counter in per_action.values()}
        return consistent and len(majorities) >= 2


def _single_in_field_centroid(arr: np.ndarray, v: int) -> np.ndarray | None:
    box = field_box(arr)
    comps = [c for c in components(arr, (v,)) if in_field(c.mean(0), box)]
    return comps[0].mean(0) if len(comps) == 1 else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py::TestAvatarDetection -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /workspace/tgaer
git add src/tgaer/agents/arc_agi3_semantics.py tests/test_arc_agi3_empirical.py
git commit -m "feat(arc-agi3): empirical avatar detection by action-conditioned translation"
```

---

### Task 2: Key detection by disappearance, door by level-increment

**Files:**
- Modify: `src/tgaer/agents/arc_agi3_semantics.py`
- Test: `tests/test_arc_agi3_empirical.py`

**Interfaces:**
- Consumes: `cells` from `arc_agi3_grid`; the pinned `avatar` from Task 1.
- Produces:
  - `EmpiricalSemantics.keys -> tuple[int, ...]` (property; accumulated pinned key values, sorted).
  - `EmpiricalSemantics.door -> int | None` (property).
  - `observe` now also runs key + door detection each step.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_arc_agi3_empirical.py
from tgaer.agents.arc_agi3_grid import cells  # noqa: E402  (top-of-file in practice)


def _with(avatar_rc, extra: dict[tuple[int, int], int]) -> np.ndarray:
    g = _grid(avatar_rc)
    for rc, v in extra.items():
        g[rc] = v
    return g


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py::TestKeyDetection tests/test_arc_agi3_empirical.py::TestDoorDetection -q`
Expected: FAIL — `AttributeError: 'EmpiricalSemantics' object has no attribute 'keys'`.

- [ ] **Step 3: Write minimal implementation**

Add `cells` to the import and track `_levels`, `_keys`, `_door`. Replace `__init__`, `observe`, and add the two detectors + properties:

```python
# import line becomes:
from tgaer.agents.arc_agi3_grid import cells, components, field_box, in_field

# in __init__ add:
        self._keys: set[int] = set()
        self._door: int | None = None
        self._levels: int | None = None

# properties:
    @property
    def keys(self) -> tuple[int, ...]:
        return tuple(sorted(self._keys))

    @property
    def door(self) -> int | None:
        return self._door

# observe becomes:
    def observe(
        self, prev: np.ndarray | None, action: int | None, cur: np.ndarray, levels: int
    ) -> None:
        if prev is not None and action is not None:
            self._observe_motion(prev, action, cur)
            if self._avatar is not None:
                self._observe_key(prev, cur)
                self._observe_door(prev, levels)
        self._levels = levels

    def _avatar_cell(self, arr: np.ndarray) -> np.ndarray | None:
        rc = cells(arr, self._avatar) if self._avatar is not None else np.empty((0, 2))
        return rc.mean(0) if len(rc) else None

    def _observe_key(self, prev: np.ndarray, cur: np.ndarray) -> None:
        av = self._avatar_cell(prev)
        if av is None:
            return
        gone = set(np.unique(prev)) - set(np.unique(cur))
        for v in gone:
            if v in (3, 4, self._avatar):  # floor / wall / avatar are not keys
                continue
            near = any(abs(r - av[0]) + abs(c - av[1]) <= 1 for r, c in cells(prev, int(v)))
            if near:
                self._keys.add(int(v))

    def _observe_door(self, prev: np.ndarray, levels: int) -> None:
        if self._levels is None or levels <= self._levels:
            return
        av = self._avatar_cell(prev)
        if av is None:
            return
        for r, c in np.argwhere(prev != 3):  # non-floor cells adjacent to avatar
            v = int(prev[r, c])
            if v in (4, self._avatar) or v in self._keys:
                continue
            if abs(r - av[0]) + abs(c - av[1]) <= 1:
                self._door = v
                return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py -q`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
cd /workspace/tgaer
git add src/tgaer/agents/arc_agi3_semantics.py tests/test_arc_agi3_empirical.py
git commit -m "feat(arc-agi3): empirical key (disappearance) + door (level-tick) detection"
```

---

### Task 3: `semantics(cold_start)` — pinned roles override, unpinned fall back

**Files:**
- Modify: `src/tgaer/agents/arc_agi3_semantics.py`
- Test: `tests/test_arc_agi3_empirical.py`

**Interfaces:**
- Consumes: `Semantics`, `LS20_DEFAULT` from `arc_agi3_grid`.
- Produces: `EmpiricalSemantics.semantics(cold_start: Semantics) -> Semantics` — pinned avatar/keys/door override the cold-start; any unpinned role is taken from `cold_start`; `walls` and `verb` always from `cold_start` (walls stay the controller's wall-oracle job; verb is out of scope, defaults via cold-start).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_arc_agi3_empirical.py
from tgaer.agents.arc_agi3_grid import LS20_DEFAULT, Semantics  # noqa: E402


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py::TestSemanticsMerge -q`
Expected: FAIL — `AttributeError: 'EmpiricalSemantics' object has no attribute 'semantics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add Semantics to the import:
from tgaer.agents.arc_agi3_grid import Semantics, cells, components, field_box, in_field

# method:
    def semantics(self, cold_start: Semantics) -> Semantics:
        return Semantics(
            avatar=self._avatar if self._avatar is not None else cold_start.avatar,
            keys=self.keys if self._keys else cold_start.keys,
            door=self._door if self._door is not None else cold_start.door,
            walls=cold_start.walls,
            verb=cold_start.verb,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tgaer
git add src/tgaer/agents/arc_agi3_semantics.py tests/test_arc_agi3_empirical.py
git commit -m "feat(arc-agi3): merge empirical pins over cold-start semantics"
```

---

### Task 4: `EmpiricalPlannerAgent` — wire detector + controller + cold-start

**Files:**
- Create: `src/tgaer/agents/arc_agi3_empirical.py`
- Test: `tests/test_arc_agi3_empirical.py`

**Interfaces:**
- Consumes: `EmpiricalSemantics`; `KeyDoorController`, `LS20_DEFAULT`, `to_action` from `arc_agi3_grid`; `Scientist` from `arc_agi3_scientist`; `Agent` base.
- Produces: `EmpiricalPlannerAgent(seed=0, model=..., api_base=..., scientist=None, **_)` with `act(observation) -> ArcAction`. Queries the cold-start scientist once per episode (first frame); detector observes every step; controller is driven by `detector.semantics(cold_start)`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_arc_agi3_empirical.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py::TestAgentIntegration -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tgaer.agents.arc_agi3_empirical'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tgaer/agents/arc_agi3_empirical.py
from __future__ import annotations

from typing import Any

import numpy as np

from tgaer.agents.arc_agi3_grid import (
    KeyDoorController,
    LS20_DEFAULT,
    to_action,
)
from tgaer.agents.arc_agi3_scientist import Scientist
from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics
from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction


class EmpiricalPlannerAgent(Agent):
    """Key→door controller whose semantics are detected from the action stream.

    A VL ``Scientist`` supplies a per-episode cold-start labelling (frame 0);
    ``EmpiricalSemantics`` then pins avatar/key/door from evidence and overrides
    the cold-start. With no VL the cold-start is ``LS20_DEFAULT``.
    """

    def __init__(
        self,
        seed: int = 0,
        model: str = "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        api_base: str | None = "http://localhost:8000/v1",
        scientist: Scientist | None = None,
        **_: Any,
    ) -> None:
        self._sci = scientist or (Scientist(model=model, api_base=api_base) if api_base else None)
        self._ctl = KeyDoorController()
        self._det = EmpiricalSemantics()
        self._cold = LS20_DEFAULT
        self._started = False
        self._levels = -1
        self._prev: np.ndarray | None = None
        self._prev_action: int | None = None
        self.last_reply: str | None = None

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        frame = obs.get("frame") or []
        if not frame:
            return to_action((obs.get("available_actions") or [1])[0])
        arr = np.asarray(frame[-1])
        levels = obs.get("levels_completed", self._levels)

        if not self._started:
            self._started = True
            if self._sci is not None:
                self._cold = self._sci.infer(frame) or LS20_DEFAULT
        if levels != self._levels:
            self._levels = levels
            self._ctl.on_new_level()

        self._det.observe(self._prev, self._prev_action, arr, levels)
        sem = self._det.semantics(self._cold)
        self._ctl.learn(arr, sem)
        aid = self._ctl.step(arr, sem, obs.get("available_actions") or [1])

        self._prev, self._prev_action = arr, aid
        self.last_reply = f"[empirical] act={aid} avatar={sem.avatar}"
        return to_action(aid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py -q`
Expected: PASS (all classes).

- [ ] **Step 5: Commit**

```bash
cd /workspace/tgaer
git add src/tgaer/agents/arc_agi3_empirical.py tests/test_arc_agi3_empirical.py
git commit -m "feat(arc-agi3): EmpiricalPlannerAgent — detector-driven semantics with VL cold-start"
```

---

### Task 5: Wire the `empirical` agent kind + experiment config

**Files:**
- Modify: `src/tgaer/evaluation/dispatch.py:7-25`
- Create: `configs/experiments/arc_agi3_empirical.yaml`
- Test: `tests/test_arc_agi3_empirical.py`

**Interfaces:**
- Consumes: `EmpiricalPlannerAgent`; the `_ARC_AGI3_AGENTS` registry dict in `dispatch.py`.
- Produces: `"empirical"` resolvable via `dispatch.available_kinds()` / agent build path.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_arc_agi3_empirical.py
from tgaer.evaluation import dispatch  # noqa: E402


class TestDispatchWiring:
    def test_empirical_kind_registered(self):
        assert dispatch._ARC_AGI3_AGENTS["empirical"] is EmpiricalPlannerAgent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tgaer && .venv/bin/python -m pytest tests/test_arc_agi3_empirical.py::TestDispatchWiring -q`
Expected: FAIL — `KeyError: 'empirical'`.

- [ ] **Step 3: Write minimal implementation**

In `src/tgaer/evaluation/dispatch.py`, add the import alongside the others (after line 10) and the registry entry (after the `"scientist"` line):

```python
from tgaer.agents.arc_agi3_empirical import EmpiricalPlannerAgent
```

```python
    "scientist": ScientistPlannerAgent,
    "empirical": EmpiricalPlannerAgent,
```

Create `configs/experiments/arc_agi3_empirical.yaml`:

```yaml
experiment_name: "arc_agi3_empirical"
seed: 0

env:
  kind: "arc_agi3"
  game_id: "ls20-9607627b"
  max_actions: 80

agent:
  kind: "empirical"  # key->door controller + stream-detected semantics, VL cold-start
  model: "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
  api_base: "http://localhost:8000/v1"

guards:
  futile_action: false
  repeated_plan: false

evaluation:
  max_steps: 80
  output_dir: "results/arc_agi3/arc_agi3_empirical/"
```

- [ ] **Step 4: Run test + full suite + lint gate**

```bash
cd /workspace/tgaer
.venv/bin/python -m pytest tests/test_arc_agi3_empirical.py -q
.venv/bin/python -m pytest -q   # full suite — no regressions in planner/scientist
ruff check src/tgaer/agents/arc_agi3_semantics.py src/tgaer/agents/arc_agi3_empirical.py src/tgaer/evaluation/dispatch.py tests/test_arc_agi3_empirical.py
ruff format --check src/tgaer/agents/arc_agi3_semantics.py src/tgaer/agents/arc_agi3_empirical.py src/tgaer/evaluation/dispatch.py tests/test_arc_agi3_empirical.py
```
Expected: all PASS / "All checks passed!" / "N files already formatted". If `ruff format --check` reports a file, run `ruff format <file>` and fold into the commit.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tgaer
git add src/tgaer/evaluation/dispatch.py configs/experiments/arc_agi3_empirical.yaml tests/test_arc_agi3_empirical.py
git commit -m "feat(arc-agi3): wire empirical agent kind + experiment config"
```

---

### Task 6 (post-implementation, not auto-run): live A/B vs planner & scientist

Not a code task — a validation step requiring the A100 + vLLM server (out of band, per the live-run protocol). After the unit work lands:

1. Stand up the Qwen3-VL server (`serving/qwen_serve.sh` equivalent, `:8000`, `--limit-mm-per-prompt '{"image": 1}'`).
2. Run all three on `ls20-9607627b`: `planner` (baseline 1/7), `scientist` (1/7 with guard), `empirical` (target ≥1/7, and check it self-corrects a deliberately-wrong cold-start).
3. Run the 25-game roster sweep for `empirical` — the headline metric is **non-zero on any game beyond LS20**, which neither prior agent achieves.
4. Tear down the server; free the A100.

Record results in the PR body (mirrors the scientist PR's live-finding section). This task gates the PR's "Live A/B" checkbox but not the merge of the unit-tested detector.

---

## Self-Review

**Spec coverage:**
- Avatar = action-conditioned translation → Task 1. ✓
- Key = disappearance-on-contact → Task 2. ✓
- Door = level-increment-on-contact → Task 2. ✓
- Cold-start handoff + override rule → Tasks 3, 4. ✓
- Asymmetric thresholds (avatar 2 / key 1 / door 1) → Task 1 `avatar_confirm=2`; key/door pin on first event in Task 2. ✓
- Lifecycle: semantics persist across levels, controller resets per level → Task 4 (`on_new_level` only resets controller; detector state persists). ✓
- New `empirical` kind, own module, scientist/planner untouched → Tasks 1 (module), 5 (wiring). ✓
- Walls stay controller's job → Task 3 (`walls` always from cold-start). ✓
- Press-verb / L1+ lattice / enemy roles out of scope → not planned. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `observe(prev, action, cur, levels)` signature identical across Tasks 1, 2, 4. `semantics(cold_start) -> Semantics` consistent Tasks 3, 4. `EmpiricalPlannerAgent(... scientist=None ...)` consistent Tasks 4, 5. Properties `avatar`/`keys`/`door` consistent. ✓

**Note on `_observe_motion` short-circuit:** once `_avatar` is pinned it returns early (Task 1), so key/door detection in Task 2's `observe` runs only after the avatar is known — matching the spec's evidence ordering (avatar first, then key/door relative to it).
