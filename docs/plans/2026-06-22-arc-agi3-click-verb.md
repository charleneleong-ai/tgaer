# ARC-AGI-3 `click` verb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `KeyDoorController` a `click` verb that points ACTION6 at a chosen target cell, so the agent can act in the 18/25 roster games that offer coordinate-click.

**Architecture:** Two localized changes to `arc_agi3_grid.py` — (1) `step()` returns an `ArcAction` instead of a bare `int` so it can carry `x/y`; (2) a top-level `click` branch emits `ArcAction(id=6, x=col, y=row)` at the two-phase key→door target via the existing role-finders. The two agents return `step()` directly. Mechanic and coordinate convention were pinned by a live probe (see spec).

**Tech Stack:** Python 3.11+, numpy, pytest (`pythonpath=["src"]`), ruff.

## Global Constraints

- Package is src-layout, NOT pip-installed: run tests as `./.venv/bin/python -m pytest` or `PYTHONPATH=src uv run python -m pytest`. Never `import tgaer` without the src path.
- **Coordinate convention (probe-pinned, verbatim):** `ACTION6 x = column, y = row`. A target at array cell `[r][c]` ⇒ `ArcAction(id=6, x=c, y=r)`.
- **Mechanic (probe-pinned):** direct-click — emit one click at the target centroid; no navigation.
- `COMPLEX_ACTION_ID == 6`, `GRID_SIZE == 64` (from `arc_agi3_api.py`).
- Lint gate before any push (no CI lint workflow / no `[tool.ruff]` config exists; `ruff format` is the repo convention): `uv run ruff check .` AND `uv run ruff format --check .`; fold fixes into the same commit.
- No `Co-Authored-By` trailers. Conventional commits.
- Verb is **config/Semantics-supplied**, never auto-detected (auto-routing is out of scope).

---

### Task 1: `KeyDoorController.step` returns `ArcAction`

Pure refactor — navigate/press behaviour is unchanged; only the return type widens from `int` to `ArcAction`. This unblocks Task 2 carrying `x/y` out of `step`.

**Files:**
- Modify: `src/tgaer/agents/arc_agi3_grid.py` (`KeyDoorController.step`, ~line 213-265)
- Modify: `src/tgaer/agents/arc_agi3_planner.py:54-56`
- Modify: `src/tgaer/agents/arc_agi3_empirical.py` (the `act` tail that wraps `step`)
- Test: `tests/test_arc_agi3_grid.py` (`TestKeyDoorController`)

**Interfaces:**
- Produces: `KeyDoorController.step(arr, sem, avail) -> ArcAction` (was `-> int`). Navigate/press return `ArcAction(id=aid)` with `x=y=None`.
- `ArcAction` is already imported in `arc_agi3_grid.py` (line 18).

- [ ] **Step 1: Migrate the existing grid tests to the new return type**

In `tests/test_arc_agi3_grid.py`, change every `TestKeyDoorController` assertion that binds `step()` to an int. Replace:

```python
    def test_step_returns_directional_action(self):
        c = KeyDoorController()
        aid = c.step(_ld_board(), LS20_DEFAULT, [1, 2, 3, 4])
        assert aid in (1, 2, 3, 4)
```
with:
```python
    def test_step_returns_directional_action(self):
        c = KeyDoorController()
        act = c.step(_ld_board(), LS20_DEFAULT, [1, 2, 3, 4])
        assert isinstance(act, ArcAction) and act.id in (1, 2, 3, 4)
```

And in the three press tests, replace `aid = c.step(...)` / `assert aid == 5` with `act = c.step(...)` / `assert act.id == 5`, and `assert aid in (1, 2, 3, 4)` with `assert act.id in (1, 2, 3, 4)`. Add `ArcAction` to the imports:
```python
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction
```

- [ ] **Step 2: Run the grid tests to verify they fail**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest tests/test_arc_agi3_grid.py -q`
Expected: FAIL — `step` returns `int`, `act.id` raises `AttributeError` / `isinstance` is False.

- [ ] **Step 3: Wrap every `step` return in `ArcAction`**

In `arc_agi3_grid.py`, change the signature to `-> ArcAction` and wrap each return. The bootstrap, no-avatar fallback, key-path, door-path and final-fallback returns become `return ArcAction(id=action)`; the press interaction becomes `return ArcAction(id=self._interaction(avail))`. Full method:

```python
    def step(self, arr: np.ndarray, sem: Semantics, avail: list[int]) -> ArcAction:
        """Choose and return the next action for navigate / press / click verbs."""
        self._progressed = False
        move_avail = [a for a in avail if a in _MOVES]

        # Bootstrap: probe each directional action once to learn its move vector.
        unprobed = [a for a in move_avail if a not in self.probed]
        if unprobed and len(self.delta) < len(move_avail):
            action = unprobed[0]
            self._remember(arr, action, sem)
            return ArcAction(id=action)

        av = cells(arr, sem.avatar)
        if not len(av) or not self.delta:
            action = self._fallback(avail, move_avail)
            self._remember(arr, action, sem)
            return ArcAction(id=action)

        ks = self._keys(arr, sem)
        d = self._door(arr, sem)
        if ks:
            centre = av.mean(0)
            self.last_key_goal = min(ks, key=lambda c: abs(c - centre).sum())

        fp = (av - av.min(0)).astype(int)
        tl = av.min(0)

        if sem.verb == "press":
            target = self.last_key_goal if ks and self.last_key_goal is not None else d
            if target is not None and self._adjacent(av, target):
                self._progressed = True
                return ArcAction(id=self._interaction(avail))

        if self.phase == "key" and self.last_key_goal is not None:
            path = self._plan(arr, fp, tl, self.last_key_goal, sem)
            if path:
                self._progressed = True
                action = path[0]
                self._remember(arr, action, sem)
                return ArcAction(id=action)
            self.phase = "door"  # reached the key — commit to the door this step

        if d is not None:
            path = self._plan(arr, fp, tl, d, sem)
            if path:
                self._progressed = True
                action = path[0]
                self._remember(arr, action, sem)
                return ArcAction(id=action)

        action = self._fallback(avail, move_avail)
        self._remember(arr, action, sem)
        return ArcAction(id=action)
```

- [ ] **Step 4: Update both agents to return `step()` directly**

In `arc_agi3_planner.py`, the `act` tail becomes (drop the `to_action` wrap; `step` is already an `ArcAction`):
```python
        action = self._ctl.step(arr, LS20_DEFAULT, obs.get("available_actions") or [1])
        self.last_reply = f"[planner] act={action.id}"
        return action
```
Apply the equivalent change in `arc_agi3_empirical.py`'s `act` (replace `return to_action(self._ctl.step(...))` with binding `action = self._ctl.step(...)`, set `last_reply` from `action.id`, `return action`). Keep `to_action` imported — it still serves the no-frame early-return path in both agents.

- [ ] **Step 5: Run the full controller + agent suite to verify green**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest tests/test_arc_agi3_grid.py tests/test_arc_agi3_planner.py tests/test_arc_agi3_empirical.py -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
cd /workspace/tgaer
git add src/tgaer/agents/arc_agi3_grid.py src/tgaer/agents/arc_agi3_planner.py src/tgaer/agents/arc_agi3_empirical.py tests/test_arc_agi3_grid.py
git commit -m "refactor(arc-agi3): KeyDoorController.step returns ArcAction (carries click coords)"
```

---

### Task 2: `click` verb + `CLICK_DEFAULT`

**Files:**
- Modify: `src/tgaer/agents/arc_agi3_grid.py` (add `CLICK_DEFAULT`, `_click_target`, click branch in `step`)
- Test: `tests/test_arc_agi3_grid.py` (new `TestClickVerb` class)

**Interfaces:**
- Consumes: `KeyDoorController.step -> ArcAction` (Task 1); `_keys`, `_door` (existing); `COMPLEX_ACTION_ID` (imported).
- Produces: `CLICK_DEFAULT: Semantics` with `verb="click"`; `KeyDoorController._click_target(arr, sem, avail) -> ArcAction | None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_arc_agi3_grid.py`:

```python
from tgaer.agents.arc_agi3_grid import CLICK_DEFAULT


class TestClickVerb:
    def _sem(self):
        return Semantics(avatar=12, keys=(0,), door=9, walls=(4,), verb="click")

    def test_click_default_is_click_verb(self):
        assert CLICK_DEFAULT.verb == "click"
        assert isinstance(CLICK_DEFAULT, Semantics)

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
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest tests/test_arc_agi3_grid.py::TestClickVerb -q`
Expected: FAIL — `CLICK_DEFAULT` undefined / click branch absent.

- [ ] **Step 3: Add `CLICK_DEFAULT` and the click logic**

In `arc_agi3_grid.py`, add the constant next to `LS20_DEFAULT` (line 34):
```python
CLICK_DEFAULT = Semantics(avatar=12, keys=(0, 1), door=9, walls=(4, 11), verb="click")
```

Add the `_click_target` helper to `KeyDoorController` (next to `_door`):
```python
    def _click_target(
        self, arr: np.ndarray, sem: Semantics, avail: list[int]
    ) -> ArcAction | None:
        """Direct-click at the two-phase target: a key while any remain, else the
        door. Emits ACTION6 at the target centroid (x=col, y=row). None when
        ACTION6 is unavailable or no target is on the board."""
        if COMPLEX_ACTION_ID not in avail:
            return None
        ks = self._keys(arr, sem)
        target = ks[0] if ks else self._door(arr, sem)
        if target is None:
            return None
        r, c = int(round(target[0])), int(round(target[1]))
        return ArcAction(id=COMPLEX_ACTION_ID, x=c, y=r)
```

Add the click branch as the FIRST decision in `step`, right after `move_avail` is computed (before bootstrap — click games have no move lattice to probe):
```python
        if sem.verb == "click":
            if (clk := self._click_target(arr, sem, avail)) is not None:
                self._progressed = True
                return clk
            return ArcAction(id=self._fallback(avail, move_avail))
```

- [ ] **Step 4: Run to verify green**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest tests/test_arc_agi3_grid.py -q`
Expected: PASS (TestClickVerb + the migrated TestKeyDoorController).

- [ ] **Step 5: Lint**

Run: `cd /workspace/tgaer && uv run ruff check src/tgaer/agents/arc_agi3_grid.py tests/test_arc_agi3_grid.py && uv run ruff format --check src/tgaer/agents/arc_agi3_grid.py tests/test_arc_agi3_grid.py`
(Apply `ruff format` + `ruff check --fix` and re-stage if anything changes.)
Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /workspace/tgaer
git add src/tgaer/agents/arc_agi3_grid.py tests/test_arc_agi3_grid.py
git commit -m "feat(arc-agi3): click verb — direct-click ACTION6 at the key→door target"
```

---

### Task 3: Review, full suite, PR

**Files:** none (process task).

- [ ] **Step 1: Self-review the diff for simplification**

Run `/simplify` (or the `code-review` skill) on the working-tree diff. Fold any findings into the Task 1/2 commits via `git commit --amend` / interactive squash BEFORE pushing — order is change → simplify → fold → commit → push.

- [ ] **Step 2: Run the full test suite**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest -q`
Expected: PASS (no regressions across the repo).

- [ ] **Step 3: Push the branch and open the stacked PR**

Base the PR on `feat/arc-agi3-empirical-semantics` (this branch is stacked on it). Use a single-quoted heredoc body with Summary / Test plan / Commits sections, link every symbol to source on the branch, and render-check with `gh pr view <N> --json body --jq '.body' | head -40`. Include the probe findings table from the spec.

```bash
cd /workspace/tgaer
git push -u origin feat/arc-agi3-click-verb
gh pr create --base feat/arc-agi3-empirical-semantics --head feat/arc-agi3-click-verb --title "feat(arc-agi3): click verb — coordinate-click capability for the controller" --body "$(cat <<'EOF'
...
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- `Semantics.verb` gains `click` + `CLICK_DEFAULT` → Task 2 Step 3. ✓
- `step` returns `ArcAction` → Task 1. ✓
- direct-click at target centroid, x=col/y=row → Task 2 (`_click_target`, tests pin convention). ✓
- two agents return `step()` directly → Task 1 Step 4. ✓
- fallbacks (no ACTION6 / no target) → Task 2 tests + branch. ✓
- "verb is config/Semantics-supplied, not auto-detected" → no auto-routing task exists (correctly out of scope). ✓
- probe harness already written/committed-as-untracked → no task needed. ✓

**Placeholder scan:** the PR heredoc body in Task 3 Step 3 is intentionally `...` (author fills per the live results at push time); every code step has complete code. No other placeholders.

**Type consistency:** `step -> ArcAction` used consistently in Tasks 1-2; `_click_target -> ArcAction | None` matches its call site (`clk := ...; if ... is not None`); `CLICK_DEFAULT` named identically in constant, import, and test. ✓
