# ARC-AGI-3 VL Scientist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Qwen3-VL "scientist" that infers each ARC-AGI-3 game's semantics (role→cell-values + interaction verb) once per level and feeds them to the key→door navigation controller, generalising the shipped LS20-only planner across the roster.

**Architecture:** Extract the planner's pure-numpy geometry and per-step navigation logic into a reusable `arc_agi3_grid.py` (`Semantics`, `KeyDoorController`, BFS). The shipped `PlannerArcAgi3Agent` becomes a thin wrapper over the controller with hardcoded `LS20_DEFAULT` semantics. A new `ScientistPlannerAgent` drives the same controller but sources `Semantics` from a `Scientist` (VL inference, once per level), falling back to `LS20_DEFAULT` when the VL server is down or returns invalid output.

**Tech Stack:** Python 3.11+, numpy, litellm (OpenAI-compatible local vLLM serving), pytest. No new dependencies.

## Global Constraints

- Spec of record: `docs/specs/2026-06-18-arc-agi3-scientist-design.md`.
- The shipped `PlannerArcAgi3Agent` must remain behaviour-identical (it holds the 1/7 LS20 result). Task 1 locks this with characterization tests *before* any refactor; every later task keeps them green.
- Default scientist model: `openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`, `api_base=http://localhost:8000/v1`. Model/api_base are constructor args (model-agnostic via litellm `openai/<served>`).
- Degradation: any scientist failure (network / parse / index-not-in-grid / bad verb) returns `None` → controller uses `LS20_DEFAULT`. No live model in tests — inject a fake.
- Only directional actions 1-4 navigate; ACTION6 (`COMPLEX_ACTION_ID=6`) carries `(x, y)`; `GRID_SIZE=64`.
- Run tests with `PYTHONPATH=src uv run pytest`. Lint gate before every push: `uvx ruff@<repo-pin> check .` AND `ruff format --check .`.

---

### Task 1: Characterization tests for the shipped planner (baseline lock)

The shipped `PlannerArcAgi3Agent` has **no committed tests**. Lock its behaviour on a synthetic key→door board and its geometry helpers before touching anything.

**Files:**
- Test: `tests/test_arc_agi3_planner.py` (create)

**Interfaces:**
- Consumes: `PlannerArcAgi3Agent` from `tgaer.agents.arc_agi3_planner`; `_components`, `_keys`, `_door` from the same module (current location).
- Produces: a reusable synthetic-board factory + scripted-clear assertion later tasks re-run unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_arc_agi3_planner.py
from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_planner import PlannerArcAgi3Agent, _components, _door, _keys
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction

# A small synthetic LS20-style board: green floor (3), a 1x1 darkred avatar (12)
# top-left, a black/blue key (0/1) mid-board, a maroon door (9) bottom-right,
# yellow wall border (4). 1-cell avatar so the move lattice is unit steps.
def _board() -> np.ndarray:
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4  # wall border
    g[2, 2] = 12  # avatar
    g[5, 5] = 0   # key marker
    g[7, 7] = 9   # door
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
        comps = _components(arr, (0,))
        assert sorted(len(c) for c in comps) == [1, 2]

    def test_keys_and_door_found_inside_field(self):
        board = _board()
        assert len(_keys(board)) == 1
        assert _door(board) is not None

class TestPlannerNavigates:
    def test_emits_a_legal_directional_action(self):
        act = PlannerArcAgi3Agent().act(_obs(_board()))
        assert isinstance(act, ArcAction) and act.id in (1, 2, 3, 4)

    def test_learns_a_move_vector_after_one_real_move(self):
        # Drive the agent once, then feed back a board where the avatar moved
        # down one row; the agent must record action->delta for that action.
        a = PlannerArcAgi3Agent()
        first = a.act(_obs(_board()))
        moved = _board()
        moved[2, 2] = 3
        moved[3, 2] = 12  # avatar shifted down one row
        a.act(_obs(moved))
        assert a.delta  # learned at least one action->vector

    def test_resets_phase_on_new_level(self):
        a = PlannerArcAgi3Agent()
        a.phase = "door"
        a.act(_obs(_board(), levels=1))  # level changed 0->1
        assert a.phase == "key"
```

- [ ] **Step 2: Run to verify they pass against the shipped code**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_planner.py -v`
Expected: PASS (these characterize existing behaviour; if any fails, the assertion is wrong about current behaviour — fix the test, not the agent).

- [ ] **Step 3: Commit**

```bash
git add tests/test_arc_agi3_planner.py
git commit -m "test(arc-agi3): characterization tests locking shipped planner behaviour"
```

---

### Task 2: Extract pure geometry into `arc_agi3_grid.py`

Move the stateless geometry + `_Planner` BFS out of the planner, add the `Semantics` contract. No behaviour change.

**Files:**
- Create: `src/tgaer/agents/arc_agi3_grid.py`
- Modify: `src/tgaer/agents/arc_agi3_planner.py` (import from the new module instead of defining locally)
- Test: `tests/test_arc_agi3_grid.py` (create)

**Interfaces:**
- Produces:
  - `Semantics(avatar:int, keys:tuple[int,...], door:int, walls:tuple[int,...], verb:str)` — frozen dataclass.
  - `LS20_DEFAULT: Semantics = Semantics(12, (0,1), 9, (4,11), "navigate")`.
  - `find_role(arr: np.ndarray, values: tuple[int,...], box: Box) -> list[np.ndarray]` — component centroids inside the field box.
  - `field_box(arr) -> Box`, `cells(arr, v)`, `components(arr, values)`, `Planner` (renamed from `_Planner`, public), `GREEN=3`, `NBRS`.
- Consumes: nothing new.

- [ ] **Step 1: Create the geometry module**

Move verbatim from `arc_agi3_planner.py`: `_cells`→`cells`, `_components`→`components`, `_field_box`→`field_box`, `_in_field`→`in_field`, `_NBRS`→`NBRS`, `_Planner`→`Planner`, and `GREEN`, `Box`. Add the new contract and role finder:

```python
# src/tgaer/agents/arc_agi3_grid.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

GREEN = 3
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
Box = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Semantics:
    avatar: int
    keys: tuple[int, ...]
    door: int
    walls: tuple[int, ...]
    verb: str  # "navigate" | "press"


LS20_DEFAULT = Semantics(avatar=12, keys=(0, 1), door=9, walls=(4, 11), verb="navigate")


def cells(arr: np.ndarray, v: int) -> np.ndarray:
    return np.argwhere(arr == v)


# ... components, field_box, in_field, Planner moved verbatim (rename _-prefix away) ...


def find_role(arr: np.ndarray, values: tuple[int, ...], box: Box) -> list[np.ndarray]:
    """Component centroids of `values` that fall inside the play-field `box`."""
    return [c.mean(0) for c in components(arr, values) if in_field(c.mean(0), box)]
```

- [ ] **Step 2: Re-point the planner at the moved symbols**

In `arc_agi3_planner.py`, delete the moved definitions and import them. Re-express `_keys`/`_door` via `find_role`:

```python
from tgaer.agents.arc_agi3_grid import (
    GREEN, NBRS, Box, LS20_DEFAULT, Planner, Semantics,
    cells, components, field_box, find_role, in_field,
)

def _keys(arr):
    return find_role(arr, KEY_VS, field_box(arr))

def _door(arr):
    ds = find_role(arr, (DOOR_V,), field_box(arr))
    return ds[0] if ds else None
```

- [ ] **Step 3: Add a focused geometry unit test**

```python
# tests/test_arc_agi3_grid.py
from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_grid import LS20_DEFAULT, Semantics, field_box, find_role


def test_find_role_filters_to_field_box():
    arr = np.full((10, 10), 3, dtype=int)
    arr[5, 5] = 9
    arr[0, 0] = 9  # outside the green field's tight box -> excluded by pad
    box = field_box(arr)
    found = find_role(arr, (9,), box)
    assert any(abs(c[0] - 5) < 1 and abs(c[1] - 5) < 1 for c in found)


def test_ls20_default_is_navigate():
    assert LS20_DEFAULT.verb == "navigate"
    assert isinstance(LS20_DEFAULT, Semantics)
```

- [ ] **Step 4: Run the full arc-agi3 test set**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_planner.py tests/test_arc_agi3_grid.py -v`
Expected: PASS — Task 1's characterization tests stay green (extraction preserved behaviour).

- [ ] **Step 5: Commit**

```bash
git add src/tgaer/agents/arc_agi3_grid.py src/tgaer/agents/arc_agi3_planner.py tests/test_arc_agi3_grid.py
git commit -m "refactor(arc-agi3): extract grid geometry + Semantics contract"
```

---

### Task 3: `KeyDoorController` — extract per-step navigation; planner becomes a thin wrapper

Pull the online delta/wall learning + two-phase key→door logic out of the agent into a `Semantics`-parameterised controller. The shipped agent delegates to it with `LS20_DEFAULT`.

**Files:**
- Modify: `src/tgaer/agents/arc_agi3_grid.py` (add `KeyDoorController`)
- Modify: `src/tgaer/agents/arc_agi3_planner.py` (reduce to a wrapper)
- Test: `tests/test_arc_agi3_grid.py` (add controller tests)

**Interfaces:**
- Produces:
  - `KeyDoorController()` with:
    - `on_new_level() -> None` — reset phase/goal/blocked, keep learned `delta`.
    - `learn(arr, sem) -> None` — update `delta` + `blocked` from the last move.
    - `step(arr, sem, avail) -> int` — return the chosen action id (navigate verb only in this task; press added in Task 6).
    - `made_progress() -> bool` — True if the last `step` advanced toward the goal (for the stall counter).
  - The controller owns `delta`, `blocked`, `phase`, `last_key_goal`, `probed`, `_prev_action`, `_prev_tl`.
- Consumes: `Semantics`, `Planner`, `find_role`, `field_box`, `cells` from this module.

- [ ] **Step 1: Write controller tests**

```python
# add to tests/test_arc_agi3_grid.py
import numpy as np
from tgaer.agents.arc_agi3_grid import KeyDoorController, LS20_DEFAULT

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
        aid = c.step(_ld_board(), LS20_DEFAULT, [1, 2, 3, 4])
        assert aid in (1, 2, 3, 4)

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
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_grid.py::TestKeyDoorController -v`
Expected: FAIL — `KeyDoorController` undefined.

- [ ] **Step 3: Implement `KeyDoorController`**

Move the agent's `_learn`, `_remember`, `_plan`, `_fallback`, the bootstrap-probe + two-phase key→door block out of `act`, and the `probed/phase/last_key_goal/_prev_*` state into the controller. Read role values from `sem` (`sem.avatar`, `sem.keys`, `sem.door`, `sem.walls`) instead of module constants. `step` returns an **action id** (not an `ArcAction` — the agent wraps it). Set a `self._progressed` flag in `step` when the chosen plan shortens the goal cover, for `made_progress`.

```python
class KeyDoorController:
    def __init__(self) -> None:
        self.delta: dict[int, np.ndarray] = {}
        self.blocked: set[tuple[int, int]] = set()
        self.probed: set[int] = set()
        self.phase = "key"
        self.last_key_goal: np.ndarray | None = None
        self._prev_action: int | None = None
        self._prev_tl: np.ndarray | None = None
        self._progressed = False

    def on_new_level(self) -> None:
        self.phase = "key"
        self.last_key_goal = None
        self.blocked = set()

    def made_progress(self) -> bool:
        return self._progressed

    # learn(), step() with the MOVES = (1,2,3,4) bootstrap, sticky key->door,
    # green-then-yellow _plan fallback, keyboard-preferring _fallback —
    # all transcribed from arc_agi3_planner.py, reading sem.* for role values.
```

- [ ] **Step 4: Reduce `PlannerArcAgi3Agent` to a wrapper**

```python
# arc_agi3_planner.py
from tgaer.agents.arc_agi3_grid import KeyDoorController, LS20_DEFAULT
from tgaer.envs.arc_agi3.arc_agi3_api import COMPLEX_ACTION_ID, GRID_SIZE, ArcAction


def _to_action(action_id: int) -> ArcAction:
    if action_id == COMPLEX_ACTION_ID:
        return ArcAction(id=action_id, x=GRID_SIZE // 2, y=GRID_SIZE // 2)
    return ArcAction(id=action_id)


class PlannerArcAgi3Agent(Agent):
    def __init__(self, seed: int = 0, **_: Any) -> None:
        self._ctl = KeyDoorController()
        self._levels = 0
        self.last_reply: str | None = None

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        frame = obs.get("frame") or []
        if not frame:
            return _to_action((obs.get("available_actions") or [1])[0])
        arr = np.asarray(frame[-1])
        levels = obs.get("levels_completed", self._levels)
        if levels != self._levels:
            self._levels = levels
            self._ctl.on_new_level()
        self._ctl.learn(arr, LS20_DEFAULT)
        aid = self._ctl.step(arr, LS20_DEFAULT, obs.get("available_actions") or [1])
        self.last_reply = f"[planner] act={aid}"
        return _to_action(aid)
```

Keep `delta`/`phase` reachable for Task 1's tests — either via `@property` proxies to `self._ctl` or by having Task 1 assert through `a._ctl.delta`/`a._ctl.phase`. Update Task 1's three assertions to read `a._ctl.delta` and `a._ctl.phase` in this commit.

- [ ] **Step 5: Run the full set green**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_planner.py tests/test_arc_agi3_grid.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tgaer/agents/arc_agi3_grid.py src/tgaer/agents/arc_agi3_planner.py tests/test_arc_agi3_planner.py tests/test_arc_agi3_grid.py
git commit -m "refactor(arc-agi3): KeyDoorController; planner is now a thin wrapper"
```

---

### Task 4: `Scientist.infer` — VL inference, parse + validate

**Files:**
- Create: `src/tgaer/agents/arc_agi3_scientist.py`
- Test: `tests/test_arc_agi3_scientist.py` (create)

**Interfaces:**
- Produces: `Scientist(model:str, api_base:str|None)` with `infer(frame) -> Semantics | None`, and an injectable `_complete(prompt, image_url) -> str` (mirrors the LLM agent so tests stub it).
- Consumes: `Semantics`, `LS20_DEFAULT` from `arc_agi3_grid`; `grid_to_png_data_url` from `tgaer.envs.arc_agi3.rendering`.

- [ ] **Step 1: Write failing tests with a fake VL client**

```python
# tests/test_arc_agi3_scientist.py
from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_grid import Semantics
from tgaer.agents.arc_agi3_scientist import Scientist

# board uses indices 3 (floor), 12 (avatar), 0/1 (key), 9 (door), 4 (wall)
def _frame():
    g = np.full((8, 8), 3, dtype=int)
    g[0, :] = 4
    g[2, 2] = 12
    g[4, 4] = 0
    g[6, 6] = 9
    return [g.tolist()]

def _sci(reply: str) -> Scientist:
    s = Scientist(model="openai/fake", api_base="http://localhost:8000/v1")
    s._complete = lambda _p, _img=None: reply  # type: ignore[method-assign]
    return s

VALID = '{"avatar": 12, "keys": [0, 1], "door": 9, "walls": [4], "verb": "navigate"}'

class TestInfer:
    def test_valid_reply_parses_to_semantics(self):
        sem = _sci(VALID).infer(_frame())
        assert sem == Semantics(12, (0, 1), 9, (4,), "navigate")

    def test_reply_in_prose_and_fences_is_parsed(self):
        reply = f"The avatar is darkred.\n```json\n{VALID}\n```"
        assert _sci(reply).infer(_frame()) is not None

    def test_index_absent_from_grid_rejected(self):
        # avatar=7 never appears on the board -> hallucination -> None
        bad = '{"avatar": 7, "keys": [0], "door": 9, "walls": [4], "verb": "navigate"}'
        assert _sci(bad).infer(_frame()) is None

    def test_bad_verb_rejected(self):
        bad = '{"avatar": 12, "keys": [0], "door": 9, "walls": [4], "verb": "teleport"}'
        assert _sci(bad).infer(_frame()) is None

    def test_unparseable_reply_returns_none(self):
        assert _sci("no json at all").infer(_frame()) is None

    def test_press_verb_accepted(self):
        ok = '{"avatar": 12, "keys": [0], "door": 9, "walls": [4], "verb": "press"}'
        assert _sci(ok).infer(_frame()).verb == "press"

    def test_complete_exception_returns_none(self):
        s = Scientist(model="openai/fake", api_base="x")
        def _boom(_p, _img=None):
            raise RuntimeError("server down")
        s._complete = _boom  # type: ignore[method-assign]
        assert s.infer(_frame()) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_scientist.py::TestInfer -v`
Expected: FAIL — module undefined.

- [ ] **Step 3: Implement `Scientist`**

```python
# src/tgaer/agents/arc_agi3_scientist.py
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from tgaer.agents.arc_agi3_grid import Semantics
from tgaer.envs.arc_agi3.rendering import grid_to_png_data_url

_JSON_RE = re.compile(r"\{[^{}]*\}")
_VERBS = ("navigate", "press")
_SYSTEM = (
    "You are the perception module for an ARC-AGI-3 grid game. You are shown the "
    "board as an image and the list of palette indices present. Identify the roles: "
    "which index is the AVATAR (the piece the player moves), which are KEY markers "
    "(collected on contact), which is the DOOR/goal, and which are WALLS (impassable). "
    "Also decide the interaction VERB: 'navigate' if the avatar wins by moving onto "
    "the key then the door, or 'press' if it must trigger an action on a target. "
    "Reply with ONE JSON object on the final line: "
    '{"avatar": <int>, "keys": [<int>...], "door": <int>, "walls": [<int>...], '
    '"verb": "navigate"|"press"}.'
)


class Scientist:
    def __init__(
        self,
        model: str = "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        api_base: str | None = "http://localhost:8000/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = "local" if api_base else None
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.last_reply: str = ""

    def infer(self, frame) -> Semantics | None:
        try:
            arr = np.asarray((frame or [None])[-1])
            present = set(int(v) for v in np.unique(arr))
            prompt = f"Palette indices present on the board: {sorted(present)}."
            self.last_reply = self._complete(prompt, grid_to_png_data_url(frame))
            return self._parse(self.last_reply, present)
        except Exception:
            return None

    def _parse(self, raw: str, present: set[int]) -> Semantics | None:
        matches = _JSON_RE.findall(raw)
        if not matches:
            return None
        d = json.loads(matches[-1])
        verb = d.get("verb")
        if verb not in _VERBS:
            return None
        avatar, door = int(d["avatar"]), int(d["door"])
        keys = tuple(int(k) for k in d.get("keys", []))
        walls = tuple(int(w) for w in d.get("walls", []))
        used = {avatar, door, *keys, *walls}
        if not used <= present or not keys or not walls:
            return None  # reject hallucinated indices / empty roles
        return Semantics(avatar, keys, door, walls, verb)

    def _complete(self, prompt: str, image_url: str | None) -> str:
        from litellm import completion

        extra = (
            {"api_base": self._api_base, "api_key": self._api_key}
            if self._api_base
            else {}
        )
        content: Any = prompt
        if image_url:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        resp = completion(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **extra,
        )
        return resp.choices[0].message.content or ""
```

- [ ] **Step 4: Run green**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_scientist.py::TestInfer -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tgaer/agents/arc_agi3_scientist.py tests/test_arc_agi3_scientist.py
git commit -m "feat(arc-agi3): Scientist — VL role/verb inference with validation"
```

---

### Task 5: `ScientistPlannerAgent` — per-level query, cache, degradation, navigate verb

**Files:**
- Modify: `src/tgaer/agents/arc_agi3_scientist.py` (add the agent)
- Test: `tests/test_arc_agi3_scientist.py` (add agent tests)

**Interfaces:**
- Produces: `ScientistPlannerAgent(seed:int=0, model:str=..., api_base:str|None=..., stall_limit:int=8, scientist:Scientist|None=None, **_)` implementing `Agent.act(observation) -> ArcAction`. Accepts an injected `scientist` for tests.
- Consumes: `KeyDoorController`, `LS20_DEFAULT` from `arc_agi3_grid`; `_to_action` (lift the helper into `arc_agi3_grid` so both agents share it — move it there in this task and re-import in the planner).

- [ ] **Step 1: Move `_to_action` into `arc_agi3_grid.py`**

Cut `_to_action` from `arc_agi3_planner.py`, paste into `arc_agi3_grid.py` as `to_action`, and import it in the planner (`from ...arc_agi3_grid import to_action`). Re-run Task 1/3 tests to confirm green.

- [ ] **Step 2: Write failing agent tests**

```python
# add to tests/test_arc_agi3_scientist.py
from tgaer.agents.arc_agi3_grid import LS20_DEFAULT, Semantics
from tgaer.agents.arc_agi3_scientist import ScientistPlannerAgent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction

class _FakeSci:
    def __init__(self, sem):
        self.sem = sem
        self.calls = 0
    def infer(self, frame):
        self.calls += 1
        return self.sem

def _obs(levels=0, actions=(1, 2, 3, 4)):
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    g[2, 2] = 12
    g[5, 5] = 0
    g[7, 7] = 9
    return {
        "frame": [g.tolist()],
        "available_actions": list(actions),
        "levels_completed": levels,
        "state": "NOT_FINISHED",
    }

class TestScientistAgent:
    def test_emits_legal_action_from_scientist_semantics(self):
        sci = _FakeSci(LS20_DEFAULT)
        act = ScientistPlannerAgent(scientist=sci).act(_obs())
        assert isinstance(act, ArcAction) and act.id in (1, 2, 3, 4)

    def test_scientist_queried_once_per_level(self):
        sci = _FakeSci(LS20_DEFAULT)
        a = ScientistPlannerAgent(scientist=sci)
        for _ in range(5):
            a.act(_obs(levels=0))
        assert sci.calls == 1  # cached after first query of the level

    def test_requeries_on_new_level(self):
        sci = _FakeSci(LS20_DEFAULT)
        a = ScientistPlannerAgent(scientist=sci)
        a.act(_obs(levels=0))
        a.act(_obs(levels=1))
        assert sci.calls == 2

    def test_none_from_scientist_falls_back_to_ls20(self):
        sci = _FakeSci(None)  # infer returns None -> degrade
        act = ScientistPlannerAgent(scientist=sci).act(_obs())
        assert act.id in (1, 2, 3, 4)  # still plays via LS20_DEFAULT
```

- [ ] **Step 3: Run to verify they fail, then implement the agent**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_scientist.py::TestScientistAgent -v` → FAIL.

```python
# add to arc_agi3_scientist.py
import numpy as np
from tgaer.agents.arc_agi3_grid import KeyDoorController, LS20_DEFAULT, to_action
from tgaer.core.agent_base import Agent


class ScientistPlannerAgent(Agent):
    def __init__(
        self,
        seed: int = 0,
        model: str = "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        api_base: str | None = "http://localhost:8000/v1",
        stall_limit: int = 8,
        scientist: Scientist | None = None,
        **_: Any,
    ) -> None:
        self._sci = scientist or Scientist(model=model, api_base=api_base)
        self._ctl = KeyDoorController()
        self._stall_limit = stall_limit
        self._levels = -1
        self._sem = LS20_DEFAULT
        self._stall = 0
        self.last_reply: str | None = None

    def act(self, observation: Any) -> ArcAction:
        obs = observation or {}
        frame = obs.get("frame") or []
        if not frame:
            return to_action((obs.get("available_actions") or [1])[0])
        arr = np.asarray(frame[-1])
        levels = obs.get("levels_completed", self._levels)
        if levels != self._levels:
            self._levels = levels
            self._ctl.on_new_level()
            self._stall = 0
            self._sem = self._sci.infer(frame) or LS20_DEFAULT
        self._ctl.learn(arr, self._sem)
        aid = self._ctl.step(arr, self._sem, obs.get("available_actions") or [1])
        self._stall = 0 if self._ctl.made_progress() else self._stall + 1
        if self._stall >= self._stall_limit:  # re-query once on a stall
            self._sem = self._sci.infer(frame) or self._sem
            self._stall = 0
        self.last_reply = f"[scientist] act={aid} verb={self._sem.verb}"
        return to_action(aid)
```

- [ ] **Step 4: Run green**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_scientist.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tgaer/agents/arc_agi3_scientist.py src/tgaer/agents/arc_agi3_planner.py tests/test_arc_agi3_scientist.py
git commit -m "feat(arc-agi3): ScientistPlannerAgent — per-level VL semantics + degradation"
```

---

### Task 6: Press-verb branch + stall re-query coverage

**Files:**
- Modify: `src/tgaer/agents/arc_agi3_grid.py` (`KeyDoorController.step` verb branch)
- Test: `tests/test_arc_agi3_grid.py`, `tests/test_arc_agi3_scientist.py`

**Interfaces:**
- Consumes: `Semantics.verb`, the available-action list passed into `step`.
- Produces: when `sem.verb == "press"`, `step` navigates the avatar footprint-adjacent to the target then returns the interaction action id — a keyboard action (the first available id in `(5, 7)`), else `COMPLEX_ACTION_ID`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_arc_agi3_grid.py
def test_press_verb_emits_interaction_when_adjacent():
    # avatar already next to the door; press verb -> interaction action, not a step
    g = np.full((6, 6), 3, dtype=int)
    g[2, 2] = 12
    g[2, 3] = 9  # door directly to the right (adjacent)
    sem = Semantics(avatar=12, keys=(), door=9, walls=(4,), verb="press")
    c = KeyDoorController()
    c.delta = {1: np.array([1, 0]), 2: np.array([-1, 0]),
               3: np.array([0, 1]), 4: np.array([0, -1])}
    aid = c.step(g, sem, [1, 2, 3, 4, 5])
    assert aid == 5  # keyboard interaction preferred
```

```python
# tests/test_arc_agi3_scientist.py
def test_stall_requery_fires_after_limit():
    class _CountSci:
        def __init__(self):
            self.calls = 0
        def infer(self, frame):
            self.calls += 1
            return LS20_DEFAULT
    sci = _CountSci()
    # walls everywhere -> controller cannot progress -> stalls
    g = np.full((10, 10), 4, dtype=int)
    g[2, 2] = 12
    obs = {"frame": [g.tolist()], "available_actions": [1, 2, 3, 4],
           "levels_completed": 0, "state": "NOT_FINISHED"}
    a = ScientistPlannerAgent(scientist=sci, stall_limit=3)
    for _ in range(6):
        a.act(obs)
    assert sci.calls >= 2  # initial + at least one stall re-query
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_grid.py -k press tests/test_arc_agi3_scientist.py -k stall -v`
Expected: FAIL.

- [ ] **Step 3: Implement the verb branch in `step`**

At the top of `KeyDoorController.step`, after locating the avatar and the target (key while any remain via `find_role(arr, sem.keys, box)`, else door via `find_role(arr, (sem.door,), box)`): if `sem.verb == "press"` and the footprint is adjacent (goal cover `<= 1` on the current lattice), return the interaction id:

```python
def _interaction(self, avail: list[int]) -> int:
    for a in (5, 7):
        if a in avail:
            return a
    return COMPLEX_ACTION_ID  # to_action attaches centre coords

# inside step(), press branch:
if sem.verb == "press" and target is not None and self._adjacent(av, target):
    self._progressed = True
    return self._interaction(avail)
```

`_adjacent` reuses the `Planner._cover` ≤ 1 footprint test. Navigation toward the target (when not yet adjacent) reuses the existing navigate path.

- [ ] **Step 4: Run green + full suite**

Run: `PYTHONPATH=src uv run pytest tests/test_arc_agi3_grid.py tests/test_arc_agi3_scientist.py tests/test_arc_agi3_planner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tgaer/agents/arc_agi3_grid.py tests/test_arc_agi3_grid.py tests/test_arc_agi3_scientist.py
git commit -m "feat(arc-agi3): press-verb interaction branch + stall re-query"
```

---

### Task 7: Dispatch wiring, config, end-to-end integration test

**Files:**
- Modify: `src/tgaer/evaluation/dispatch.py` (register `"scientist"`)
- Create: `configs/experiments/arc_agi3_scientist.yaml`
- Test: `tests/test_eval_dispatch.py` (add a scientist-kind case)

**Interfaces:**
- Consumes: `ScientistPlannerAgent`.
- Produces: agent kind `"scientist"` resolvable by `_build_arc_agi3_agent`; a config that round-trips through `run_eval` with an injected scripted transport + injected fake scientist.

- [ ] **Step 1: Write the failing integration test**

```python
# add to tests/test_eval_dispatch.py
def test_scientist_kind_is_registered():
    from tgaer.evaluation.dispatch import _ARC_AGI3_AGENTS
    assert "scientist" in _ARC_AGI3_AGENTS

def test_scientist_config_runs_with_injected_transport(monkeypatch):
    # The scientist agent must construct without a live VL server (it only
    # queries on the first frame); the scripted transport wins the level.
    cfg = yaml.safe_load(Path("configs/experiments/arc_agi3_scientist.yaml").read_text())
    # force the agent's scientist to a no-network stub via the constructor default:
    import tgaer.agents.arc_agi3_scientist as sci_mod
    monkeypatch.setattr(
        sci_mod.Scientist, "infer", lambda self, frame: None  # degrade -> LS20
    )
    result = dispatch.run_eval(cfg, transport=_ScriptedTransport())
    assert result.score == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_eval_dispatch.py -k scientist -v`
Expected: FAIL — kind not registered / config missing.

- [ ] **Step 3: Register the kind**

```python
# dispatch.py
from tgaer.agents.arc_agi3_scientist import ScientistPlannerAgent

_ARC_AGI3_AGENTS = {
    "random": RandomArcAgi3Agent,
    "llm": ArcAgi3LLMAgent,
    "planner": PlannerArcAgi3Agent,
    "scientist": ScientistPlannerAgent,
}
```

- [ ] **Step 4: Write the config**

```yaml
# configs/experiments/arc_agi3_scientist.yaml
experiment_name: "arc_agi3_scientist"
seed: 0

env:
  kind: "arc_agi3"
  game_id: "ls20-9607627b"
  max_actions: 80

agent:
  kind: "scientist"  # key->door controller + per-level Qwen3-VL semantics scientist
  model: "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
  api_base: "http://localhost:8000/v1"
  stall_limit: 8

guards:
  futile_action: false
  repeated_plan: false

evaluation:
  max_steps: 80
  output_dir: "results/arc_agi3/arc_agi3_scientist/"
```

- [ ] **Step 5: Run green + full repo suite**

Run: `PYTHONPATH=src uv run pytest -q`
Expected: PASS (all existing + new).

- [ ] **Step 6: Lint + commit**

```bash
uvx ruff@<repo-pin> check . && ruff format --check .
git add src/tgaer/evaluation/dispatch.py configs/experiments/arc_agi3_scientist.yaml tests/test_eval_dispatch.py
git commit -m "feat(arc-agi3): wire scientist agent kind + experiment config"
```

---

## Self-Review

**Spec coverage:**
- Roles + verb output → Task 4 (`Scientist._parse`). ✓
- Once-per-level + stall re-query → Task 5 (cache) + Task 6 (stall). ✓
- Degradation to LS20 → Task 4 (`None`) + Task 5 (`or LS20_DEFAULT`). ✓
- 30B-AWQ default, model-as-config → Task 4/5 constructor + Task 7 config. ✓
- Extract geometry + `KeyDoorController`, planner thin wrapper proven by its tests → Tasks 1-3. ✓
- Navigate vs press branch → Task 6. ✓
- Dispatch + config + tests → Task 7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows code; the geometry "move verbatim" steps name exact symbols. The only `<repo-pin>` is the ruff version, resolved by reading `pyproject.toml`/CI at execution.

**Type consistency:** `Semantics(avatar, keys, door, walls, verb)` used identically across Tasks 2/4/5/6. `step(arr, sem, avail) -> int` and `to_action(int) -> ArcAction` consistent Tasks 3/5/6. `infer(frame) -> Semantics | None` consistent Tasks 4/5.
