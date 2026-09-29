# Replay Effect Model (stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A measurement-only bench tool that fits a coarse typed effect model from recorded ARC-AGI-3 transitions and reports, per game, whether it predicts effects better than the mechanism already shipped.

**Architecture:** One script, `sia-oss/bench/replay_effects.py`. Pure logic (effect signature, model, scorer) lives at module level so tests can import it by path without loading games; only `main()` drives the harness through `play(..., on_step=hook)`. Four measurement arms are the *same* `EffectModel` with different key lists — the marginal baseline is the empty key — so there is one code path to get right.

**Tech Stack:** Python 3.12, numpy, typer, loguru, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-27-replay-effect-model-design.md`

## Global Constraints

- **Nothing under `src/` may change.** Stage 1 is measurement only; any `src/` edit means the plan was misread.
- No LLM, no network, no new dependency.
- `sia-oss/bench` is a script directory, not a package: tests load modules via `importlib.util.spec_from_file_location`, following `tests/test_arc_agi3_bench_ab.py`.
- Importing `replay_effects` must not call `require_starter()` or load any game. Harness calls belong inside `main()` only.
- Type hints on every signature, explicit generic parameters (`dict[str, int]`, never bare `dict`).
- Terse comments; rationale goes in the PR body, not the source.
- `bucket(n) = 0 if n == 0 else 1 + floor(log2(n))`, used for both changed-cell counts and component sizes.

## Review Focus

- **Click games never pin an avatar** (lp85, m0r0, s5i5, g50t): `avatar_delta` is `None` for every transition and must not crash or silently bias accuracy. Test in Task 1.
- **Death/respawn frames are not transitions**: the agent excludes them via `obs["terminal"]`; scoring them would teach the model a reset is an ordinary effect. Test in Task 4.
- **A click target with no resolvable component** leaves colour/size unknown; the key must degrade to `(action,)` rather than raise. Test in Task 2.
- **A game with exactly one distinct effect** makes marginal accuracy 1.0 and class-keyed no better; the report must not divide by zero and must not read as a win. Test in Task 3.
- **The very first transition of a game** has an empty model: `predict` returns `None`, which must count as a miss rather than crash. Test in Task 3.

---

### Task 1: Effect signature

**Files:**
- Create: `sia-oss/bench/replay_effects.py`
- Test: `tests/test_arc_agi3_bench_replay_effects.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Effect = tuple[bool, tuple[int, int] | None, frozenset[int], frozenset[int], int]`; `bucket(n: int) -> int`; `effect_signature(prev: np.ndarray, cur: np.ndarray, level_advanced: bool, avatar: int | None) -> Effect`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the replay effect model.

Loaded by path because `sia-oss/bench` is a script directory, not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "sia-oss/bench"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


re_ = _load("replay_effects")


def _grid(avatar_rc: tuple[int, int] | None = None, extra: dict[tuple[int, int], int] | None = None) -> np.ndarray:
    g = np.full((10, 10), 3, dtype=np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    if avatar_rc is not None:
        g[avatar_rc] = 12
    for rc, v in (extra or {}).items():
        g[rc] = v
    return g


class TestBucket:
    @pytest.mark.parametrize(("n", "want"), [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (7, 3), (8, 4)])
    def test_log2_buckets_with_zero_reserved(self, n: int, want: int) -> None:
        assert re_.bucket(n) == want


class TestEffectSignature:
    def test_identical_frames_have_zero_churn(self) -> None:
        g = _grid((2, 2))
        eff = re_.effect_signature(g, g.copy(), False, 12)
        assert eff[4] == 0

    def test_avatar_translation_is_recorded(self) -> None:
        eff = re_.effect_signature(_grid((2, 2)), _grid((3, 2)), False, 12)
        assert eff[1] == (1, 0)

    def test_avatar_delta_is_none_on_a_click_game(self) -> None:
        # lp85/m0r0/s5i5/g50t never pin an avatar; this must not crash.
        eff = re_.effect_signature(_grid(None, {(2, 2): 5}), _grid(None), False, None)
        assert eff[1] is None
        assert eff[3] == frozenset({5})  # the 5 vanished

    def test_appeared_and_vanished_are_symmetric(self) -> None:
        eff = re_.effect_signature(_grid((2, 2), {(5, 5): 7}), _grid((2, 2), {(5, 5): 9}), False, 12)
        assert eff[2] == frozenset({9})
        assert eff[3] == frozenset({7})

    def test_level_advance_is_carried_through(self) -> None:
        g = _grid((2, 2))
        assert re_.effect_signature(g, g.copy(), True, 12)[0] is True

    def test_the_signature_is_hashable(self) -> None:
        g = _grid((2, 2))
        assert len({re_.effect_signature(g, g.copy(), False, 12)}) == 1

    def test_chrome_cells_do_not_change_the_signature(self) -> None:
        # The test most worth having: chrome has broken two mechanisms already. The
        # caller passes _settled frames, so a cell that only self-animates is already
        # masked to a constant and must not register as churn.
        settled_a = _grid((2, 2))
        settled_b = _grid((2, 2))
        assert re_.effect_signature(settled_a, settled_b, False, 12)[4] == 0
        # A real in-field change still registers.
        assert re_.effect_signature(settled_a, _grid((2, 3)), False, 12)[4] > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_arc_agi3_bench_replay_effects.py -q`
Expected: FAIL — collection error, `replay_effects.py` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
#!/usr/bin/env python3
"""Does a coarse effect model predict what an action does, better than what ships?

Measurement only: it fits nothing into the agent and changes nothing under `src/`.
Each recorded transition is predicted *before* it is learned from (prequential), so
every number is out-of-sample without needing a split.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

# (level_advanced, avatar_delta, appeared, vanished, churn_bucket)
Effect = tuple[bool, tuple[int, int] | None, frozenset[int], frozenset[int], int]
# (signature, action, colour, size_bucket)
Ctx = tuple[Any, int, int | None, int | None]
Key = tuple[Any, ...]

CLICK_ACTION = 6  # the API's click is action id 6; not a colour, so not roster-fitted


def bucket(n: int) -> int:
    """0 is reserved for "none"; everything else is a log2 band."""
    return 0 if n <= 0 else 1 + int(math.log2(n))


def centroid(arr: np.ndarray, value: int) -> tuple[float, float] | None:
    cells = np.argwhere(arr == value)
    if not len(cells):
        return None
    return float(cells[:, 0].mean()), float(cells[:, 1].mean())


def effect_signature(
    prev: np.ndarray, cur: np.ndarray, level_advanced: bool, avatar: int | None
) -> Effect:
    """A coarse, hashable description of what one action did."""
    before = {int(v) for v in np.unique(prev)}
    after = {int(v) for v in np.unique(cur)}
    delta: tuple[int, int] | None = None
    if avatar is not None:
        p, c = centroid(prev, avatar), centroid(cur, avatar)
        if p is not None and c is not None:
            delta = (int(round(c[0] - p[0])), int(round(c[1] - p[1])))
    changed = int(np.count_nonzero(prev != cur)) if prev.shape == cur.shape else -1
    return (
        bool(level_advanced),
        delta,
        frozenset(after - before),
        frozenset(before - after),
        bucket(changed),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_arc_agi3_bench_replay_effects.py -q`
Expected: PASS, 12 tests.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
uv run ruff check sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
git add sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
git commit -m "feat(bench): coarse typed effect signature for replay measurement"
```

---

### Task 2: The model with hierarchical back-off

**Files:**
- Modify: `sia-oss/bench/replay_effects.py`
- Test: `tests/test_arc_agi3_bench_replay_effects.py`

**Interfaces:**
- Consumes: `Effect`, `Ctx`, `Key` from Task 1.
- Produces: `EffectModel` with `predict(ctx: Ctx) -> tuple[Effect | None, int]` returning `(modal effect, index of the key level used)` and `-1` when nothing matches; `observe(ctx: Ctx, effect: Effect) -> None`; `seen(ctx: Ctx) -> bool`. Plus the key functions `class_keys`, `state_keys`, `marginal_keys`, each `Callable[[Ctx], list[Key]]`.

- [ ] **Step 1: Write the failing tests**

```python
class TestEffectModel:
    @staticmethod
    def _eff(churn: int) -> Any:
        return (False, None, frozenset(), frozenset(), churn)

    def test_an_empty_model_predicts_nothing(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        assert m.predict(("sig", 6, 5, 2)) == (None, -1)

    def test_it_predicts_the_mode_at_the_specific_key(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        for _ in range(3):
            m.observe(("s1", 6, 5, 2), self._eff(4))
        m.observe(("s1", 6, 5, 2), self._eff(9))
        got, level = m.predict(("s2", 6, 5, 2))
        assert got == self._eff(4)
        assert level == 0  # most specific key

    def test_a_rule_learned_at_one_state_fires_at_another(self) -> None:
        # This is the whole point: cross-state generalisation.
        m = re_.EffectModel(re_.class_keys)
        m.observe(("s1", 6, 5, 2), self._eff(4))
        assert m.predict(("s2", 6, 5, 2))[0] == self._eff(4)

    def test_it_backs_off_to_colour_then_action(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        m.observe(("s1", 6, 5, 2), self._eff(4))
        # Novel size, same colour -> backs off one level.
        assert m.predict(("s1", 6, 5, 9))[1] == 1
        # Novel colour -> backs off to the action alone.
        assert m.predict(("s1", 6, 77, 9))[1] == 2

    def test_an_unresolvable_click_target_degrades_to_the_action(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        m.observe(("s1", 6, 5, 2), self._eff(4))
        got, level = m.predict(("s1", 6, None, None))
        assert got == self._eff(4)
        assert level == 2

    def test_the_state_keyed_arm_does_not_generalise_across_states(self) -> None:
        m = re_.EffectModel(re_.state_keys)
        m.observe(("s1", 6, 5, 2), self._eff(4))
        assert m.predict(("s2", 6, 5, 2)) == (None, -1)

    def test_the_marginal_arm_ignores_the_key_entirely(self) -> None:
        m = re_.EffectModel(re_.marginal_keys)
        m.observe(("s1", 1, None, None), self._eff(4))
        assert m.predict(("s9", 6, 77, 3))[0] == self._eff(4)

    def test_seen_reports_whether_the_specific_key_is_known(self) -> None:
        m = re_.EffectModel(re_.class_keys)
        assert not m.seen(("s1", 6, 5, 2))
        m.observe(("s1", 6, 5, 2), self._eff(4))
        assert m.seen(("s1", 6, 5, 2))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_arc_agi3_bench_replay_effects.py -q -k EffectModel`
Expected: FAIL with `AttributeError: module has no attribute 'EffectModel'`.

- [ ] **Step 3: Write the minimal implementation**

```python
def class_keys(ctx: Ctx) -> list[Key]:
    """Most specific first. Back-off is what lets a rule fire somewhere new."""
    _, action, colour, size = ctx
    keys: list[Key] = []
    if colour is not None and size is not None:
        keys.append((action, colour, size))
    if colour is not None:
        keys.append((action, colour))
    keys.append((action,))
    return keys


def state_keys(ctx: Ctx) -> list[Key]:
    """Per state, no generalisation — the control arm."""
    sig, action, colour, size = ctx
    return [(sig, action, colour, size)]


def marginal_keys(ctx: Ctx) -> list[Key]:
    """One bucket for the whole game: predicting the commonest effect."""
    return [()]


class EffectModel:
    """Counts effects per key and predicts the mode, backing off when unseen."""

    def __init__(self, keys: Callable[[Ctx], list[Key]]) -> None:
        self._keys = keys
        self._counts: dict[Key, Counter[Effect]] = {}

    def predict(self, ctx: Ctx) -> tuple[Effect | None, int]:
        for level, key in enumerate(self._keys(ctx)):
            counter = self._counts.get(key)
            if counter:
                return counter.most_common(1)[0][0], level
        return None, -1

    def observe(self, ctx: Ctx, effect: Effect) -> None:
        for key in self._keys(ctx):
            self._counts.setdefault(key, Counter())[effect] += 1

    def seen(self, ctx: Ctx) -> bool:
        keys = self._keys(ctx)
        return bool(keys) and keys[0] in self._counts
```

Note `class_keys` omits the specific key when colour is `None`, so an unresolvable
click target degrades rather than raising — and `predict` reports the level it used,
which is what the report needs to say how often the specific key was available.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_arc_agi3_bench_replay_effects.py -q`
Expected: PASS, 20 tests.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
uv run ruff check sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
git add sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
git commit -m "feat(bench): effect model with hierarchical back-off"
```

---

### Task 3: Prequential scorer and the four arms

**Files:**
- Modify: `sia-oss/bench/replay_effects.py`
- Test: `tests/test_arc_agi3_bench_replay_effects.py`

**Interfaces:**
- Consumes: `EffectModel`, the three key functions, `Effect`, `Ctx`.
- Produces: `inert_view(effect: Effect) -> tuple[bool]`; `Prequential` with `step(ctx: Ctx, effect: Effect) -> None`, `report() -> dict[str, float]`, and attribute `n: int`. Report keys exactly: `"class"`, `"state"`, `"inert"`, `"marginal"`, `"first_sighting"`, `"backoff_0"`, `"backoff_1"`, `"backoff_2"`.

- [ ] **Step 1: Write the failing tests**

```python
class TestPrequential:
    @staticmethod
    def _eff(churn: int) -> Any:
        return (False, None, frozenset(), frozenset(), churn)

    def test_the_first_transition_is_a_miss_not_a_crash(self) -> None:
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))
        assert p.n == 1
        assert p.report()["class"] == 0.0

    def test_a_deterministic_rule_is_learned_after_first_sighting(self) -> None:
        p = re_.Prequential()
        for _ in range(11):
            p.step(("s1", 6, 5, 2), self._eff(4))
        # 1 miss (empty model) then 10 hits.
        assert p.report()["class"] == pytest.approx(10 / 11)

    def test_an_inconsistent_rule_does_not_beat_its_own_purity(self) -> None:
        p = re_.Prequential()
        for i in range(100):
            p.step(("s1", 6, 5, 2), self._eff(4 if i % 4 else 9))
        assert p.report()["class"] <= 0.76

    def test_class_beats_state_when_the_rule_generalises(self) -> None:
        p = re_.Prequential()
        for i in range(20):
            p.step((f"s{i}", 6, 5, 2), self._eff(4))  # every state is new
        r = p.report()
        assert r["class"] > r["state"]
        assert r["state"] == 0.0  # per-state learning never sees a state twice

    def test_a_single_effect_game_gives_marginal_parity_and_no_zero_division(self) -> None:
        p = re_.Prequential()
        for i in range(10):
            p.step((f"s{i}", 1, None, None), self._eff(0))
        r = p.report()
        assert r["marginal"] == pytest.approx(r["class"])

    def test_the_inert_arm_only_judges_whether_anything_changed(self) -> None:
        # Same "changed vs not", different churn magnitude: inert is right, class is wrong.
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))
        p.step(("s1", 6, 5, 2), self._eff(9))
        r = p.report()
        assert r["inert"] > r["class"]

    def test_first_sighting_accuracy_is_reported_separately(self) -> None:
        p = re_.Prequential()
        p.step(("s1", 6, 5, 2), self._eff(4))   # first sighting of this key
        p.step(("s1", 6, 5, 2), self._eff(4))   # not a first sighting
        assert 0.0 <= p.report()["first_sighting"] <= 1.0

    def test_backoff_levels_are_reported_as_shares(self) -> None:
        p = re_.Prequential()
        for _ in range(4):
            p.step(("s1", 6, 5, 2), self._eff(4))
        r = p.report()
        assert sum(r[k] for k in ("backoff_0", "backoff_1", "backoff_2")) <= 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_arc_agi3_bench_replay_effects.py -q -k Prequential`
Expected: FAIL with `AttributeError: module has no attribute 'Prequential'`.

- [ ] **Step 3: Write the minimal implementation**

```python
def inert_view(effect: Effect) -> tuple[bool]:
    """What `_inert` knows: whether anything changed at all."""
    return (effect[4] == 0,)


class Prequential:
    """Predict with the model as it stands, score, then learn. No split, no leakage."""

    def __init__(self) -> None:
        self._arms: dict[str, EffectModel] = {
            "class": EffectModel(class_keys),
            "state": EffectModel(state_keys),
            "marginal": EffectModel(marginal_keys),
        }
        self._inert = EffectModel(class_keys)
        self._hits: Counter[str] = Counter()
        self._levels: Counter[int] = Counter()
        self._first_hits = 0
        self._first_n = 0
        self.n = 0

    def step(self, ctx: Ctx, effect: Effect) -> None:
        fresh = not self._arms["class"].seen(ctx)
        for name, model in self._arms.items():
            got, level = model.predict(ctx)
            if got == effect:
                self._hits[name] += 1
                if name == "class" and fresh:
                    self._first_hits += 1
            if name == "class":
                self._levels[level] += 1
        got_inert, _ = self._inert.predict(ctx)
        if got_inert is not None and inert_view(got_inert) == inert_view(effect):
            self._hits["inert"] += 1
        if fresh:
            self._first_n += 1

        for model in self._arms.values():
            model.observe(ctx, effect)
        self._inert.observe(ctx, effect)
        self.n += 1

    def report(self) -> dict[str, float]:
        n = max(self.n, 1)
        out = {k: self._hits[k] / n for k in ("class", "state", "inert", "marginal")}
        out["first_sighting"] = self._first_hits / max(self._first_n, 1)
        for level in (0, 1, 2):
            out[f"backoff_{level}"] = self._levels[level] / n
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_arc_agi3_bench_replay_effects.py -q`
Expected: PASS, 28 tests.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
uv run ruff check sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
git add sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
git commit -m "feat(bench): prequential scorer with inert, state and marginal arms"
```

---

### Task 4: Harness wiring and the report

**Files:**
- Modify: `sia-oss/bench/replay_effects.py`
- Test: `tests/test_arc_agi3_bench_replay_effects.py`

**Interfaces:**
- Consumes: `Prequential`, `effect_signature`.
- Produces: `context_of(prim: tuple[str, Any], arr: np.ndarray, sig: Any) -> Ctx`; `is_transition(obs: dict[str, Any]) -> bool`; a typer `main()` taking `--games`, `--max-steps`, `--seed`.

- [ ] **Step 1: Write the failing tests**

```python
class TestHarnessGlue:
    def test_a_respawn_frame_is_not_a_transition(self) -> None:
        # The agent excludes these; scoring them teaches that a reset is an ordinary
        # effect. This is the one that silently poisons the model if missed.
        assert not re_.is_transition({"terminal": True})
        assert re_.is_transition({})
        assert re_.is_transition({"terminal": False})

    def test_a_click_context_carries_colour_and_size_bucket(self) -> None:
        arr = _grid(None, {(5, 5): 7, (5, 6): 7})
        ctx = re_.context_of(("click", (5, 5)), arr, "sig")
        assert ctx[0] == "sig"
        assert ctx[2] == 7
        assert ctx[3] == re_.bucket(2)

    def test_a_simple_action_context_has_no_colour_or_size(self) -> None:
        ctx = re_.context_of(("act", 3), _grid((2, 2)), "sig")
        assert ctx[1] == 3
        assert ctx[2] is None and ctx[3] is None

    def test_an_out_of_bounds_click_yields_no_colour(self) -> None:
        ctx = re_.context_of(("click", (99, 99)), _grid((2, 2)), "sig")
        assert ctx[2] is None and ctx[3] is None

    def test_importing_the_module_loads_no_games(self) -> None:
        # main() owns every harness call; import must stay side-effect free.
        src = (BENCH / "replay_effects.py").read_text()
        head = src.split("def main(")[0]
        assert "require_starter()" not in head
        assert "Arcade(" not in head
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_arc_agi3_bench_replay_effects.py -q -k HarnessGlue`
Expected: FAIL with `AttributeError: module has no attribute 'is_transition'`.

- [ ] **Step 3: Write the minimal implementation**

```python
def is_transition(obs: dict[str, Any]) -> bool:
    """A respawn frame is a fresh level start, not a successor of the last action."""
    return not obs.get("terminal")


def context_of(prim: tuple[str, Any], arr: np.ndarray, sig: Any) -> Ctx:
    """Key parts for one primitive: colour and size only when a click resolves."""
    kind, payload = prim
    if kind != "click":
        return (sig, int(payload), None, None)
    row, col = int(payload[0]), int(payload[1])
    if not (0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]):
        return (sig, CLICK_ACTION, None, None)
    colour = int(arr[row, col])
    size = 0
    for comp in components(arr, (colour,)):
        if any(int(r) == row and int(c) == col for r, c in comp):
            size = len(comp)
            break
    return (sig, CLICK_ACTION, colour, bucket(size))
```

Then the typer command, which is the only place the harness is touched:

```python
import typer  # noqa: E402
from loguru import logger  # noqa: E402

import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
)

from tgaer.agents.arc_agi3_explorer import frame_signature  # noqa: E402
from tgaer.agents.arc_agi3_grid import components  # noqa: E402
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    load_agent_class,
    play,
    require_starter,
)

app = typer.Typer(add_completion=False)
SUITE = "tu93,s5i5,ar25,sp80,ls20,m0r0,lp85,g50t"


@app.command()
def main(
    games: str = typer.Option(SUITE, help="Comma-separated game ids."),
    max_steps: int = typer.Option(600),
    seed: int = typer.Option(0),
) -> None:
    """Score the effect model prequentially along the explorer's own trajectory."""
    require_starter()
    logger.info(
        "{:6} {:>6} {:>7} {:>7} {:>7} {:>9} {:>6}",
        "game", "n", "class", "state", "inert", "marginal", "first",
    )
    for game_id in games.split(","):
        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(REPO / "environment_files"),
        )
        pre = Prequential()
        prev: dict[str, Any] = {}

        def hook(step: int, obs: Any, env: Any, actor: Any, _p: dict[str, Any] = prev) -> None:
            obs = obs or {}
            frame = obs.get("frame") or []
            if not frame:
                return
            arr = np.asarray(frame[-1], dtype=np.int16)
            inner = getattr(actor, "_explorer", actor)
            levels = int(obs.get("levels_completed", 0))
            settled = inner._settled(arr)
            # Effects are measured on settled frames, per the spec: chrome must not
            # register as churn. Contexts key on the raw board, because the colour
            # under a click is what the agent actually clicked.
            if _p and is_transition(obs):
                eff = effect_signature(
                    _p["settled"], settled, levels > _p["levels"], inner._det.avatar
                )
                pre.step(_p["ctx"], eff)
            prim = (getattr(actor, "trace", {}) or {}).get("prim")
            if prim is None:
                _p.clear()
                return
            sig = frame_signature(settled, inner._field(arr))
            _p.update(
                settled=settled,
                levels=levels,
                ctx=context_of(tuple(prim), arr, sig),
            )

        try:
            play(load_agent_class(None, "explorer"), game_id, arc, None, max_steps, seed=seed, on_step=hook)
        except Exception as exc:  # a broken game must not abort the sweep
            logger.error("{}: {}: {}", game_id, type(exc).__name__, exc)
            continue
        r = pre.report()
        logger.info(
            "{:6} {:>6} {:>6.1%} {:>7.1%} {:>7.1%} {:>9.1%} {:>6.1%}",
            game_id, pre.n, r["class"], r["state"], r["inert"], r["marginal"], r["first_sighting"],
        )
        logger.info(
            "        back-off: specific {:.0%}  colour {:.0%}  action {:.0%}",
            r["backoff_0"], r["backoff_1"], r["backoff_2"],
        )


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_arc_agi3_bench_replay_effects.py -q`
Expected: PASS, 33 tests.

- [ ] **Step 5: Run the whole suite, then the tool for real**

Run: `uv run pytest -q` — expected: all pass, no `src/` change.
Run: `uv run python sia-oss/bench/replay_effects.py --max-steps 600`
Expected: one row per game with the six percentages populated.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
uv run ruff check sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
git status --short src/    # must print nothing
git add sia-oss/bench/replay_effects.py tests/test_arc_agi3_bench_replay_effects.py
git commit -m "feat(bench): wire the replay effect model to the explorer trajectory"
```

---

## Reading the result

The spec's gate for stage 2: `class` must beat **both** `state` and `inert`, and
`first_sighting` must beat `marginal`. If `first_sighting` is at or below `marginal`,
the model memorises rather than generalises and stage 2 is not built.

Expect `inert` to be high — effect purity is 98–100% and `inert` only judges
changed-vs-not. `class` losing to `inert` is the likely outcome and is a real answer,
not a bug: it would say the shipped mechanism already captures the predictable part.
