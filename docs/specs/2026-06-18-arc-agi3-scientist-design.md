# ARC-AGI-3 VL Scientist — per-level semantics for the key→door controller

**Date:** 2026-06-18
**Status:** Design approved, pre-implementation
**Branch:** `feat/arc-agi3-vl-scientist`

## Summary

The shipped `PlannerArcAgi3Agent` (PR #12) is a reset-free key→door navigator that
solves LS20 (1/7 levels) where every frozen-LLM baseline scores 0. It works because
it *knows* LS20's semantics — which palette indices are the avatar, key, door, walls —
hardcoded as module constants. On the other 24 roster games it scores 0: the navigation
controller is fine, but the **identification** is wrong (different games recolour the
roles). This is the controller half of a controller+scientist hybrid still missing its
scientist.

This spec adds the scientist: a **Qwen3-VL** model that infers each game's semantics
**once per level** and feeds them to the controller. The VL model is repurposed from the
failed per-step policy role (80 calls/episode, scored 0/7) to a per-level identifier
(~1 call/level) — the regime where a strong model's perception pays off and call count
is negligible.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scientist output | Roles map **+ interaction verb** | Roles alone only cover pure-navigation games; the verb (`navigate` vs `press`) lets press-on-target games work too. |
| VL cadence | **Once per level + stall re-query** | Bounded VL calls (~levels + a few). Controller does all per-step work. Per-step VL is the 0/7 baseline. |
| Degradation | **Fall back to LS20 heuristic** | Scientist is a pure enrichment layer; server-down/invalid-output preserves the shipped 1/7 with no live model. |
| Default model | **`QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`** | One-shot role labelling rewards stronger perception; ~1 call/level makes the cost negligible. 4B stays available for CI/cheap runs. |
| Integration | **Extract shared geometry; new `scientist` agent kind** | Shipped planner can't regress; scientist unit-testable against a fake VL client; geometry lives in one place. |

## Architecture

```
observation ──> ScientistPlannerAgent.act
                  │
                  ├─ new level? ─> Scientist.infer(frame) ─> Semantics | None
                  │                   (VL: board PNG + palette legend → strict JSON)
                  │                   None ─> LS20_DEFAULT
                  │                   cache per level
                  │
                  └─ KeyDoorController.step(arr, semantics) ─> ArcAction
                       (online delta/wall learning, two-phase key→door,
                        verb branch: navigate vs press, keyboard-pref fallback)
```

VL is touched ~once per level (+ rare stall re-query); the controller runs every step.

## Components

### 1. `src/tgaer/agents/arc_agi3_grid.py` (new — pure geometry, no LLM)

Extracted verbatim from the shipped planner (behaviour-preserving):
`_cells`, `_components` (4-connected BFS), `_field_box`, `_in_field`, `_NBRS`, and the
`_Planner` footprint-BFS class. Generalises `_keys`/`_door` into one role finder:

```python
def find_role(arr: np.ndarray, values: tuple[int, ...], box: Box) -> list[np.ndarray]:
    """Component centroids of `values` that fall inside the green play-field box."""
```

Plus the semantics contract and the LS20 default:

```python
@dataclass(frozen=True)
class Semantics:
    avatar: int
    keys: tuple[int, ...]
    door: int
    walls: tuple[int, ...]
    verb: str  # "navigate" | "press"

LS20_DEFAULT = Semantics(avatar=12, keys=(0, 1), door=9, walls=(4, 11), verb="navigate")
```

### 2. `KeyDoorController` (in `arc_agi3_grid.py`)

The reusable navigation core, parameterised by a `Semantics`. Owns the online
action→delta map, the wall set, the two-phase sticky key→door goal, and the
keyboard-preferring fallback — i.e. all of the shipped planner's per-step logic, minus
the hardcoded constants (now read from `semantics`).

```python
class KeyDoorController:
    def __init__(self) -> None: ...
    def on_new_level(self) -> None: ...               # reset phase/goal/blocked, keep delta map
    def step(self, arr: np.ndarray, sem: Semantics, avail: list[int]) -> ArcAction: ...
    def made_progress(self) -> bool: ...              # for the scientist's stall counter
```

`verb` branch inside `step`:
- `navigate` — footprint-cover arrival onto key then door (today's logic).
- `press` — navigate footprint-**adjacent** to the target, then emit the interaction
  action: prefer a keyboard action (5/7) if available, else `ACTION6` at the target cell.

### 3. `PlannerArcAgi3Agent` (refactor — becomes a thin wrapper)

The shipped agent is reduced to: a `KeyDoorController` driven with `LS20_DEFAULT`. Its
**existing test suite staying green is the proof** the extraction preserved the 1/7. No
behaviour change; the only touch to the shipped path.

### 4. `src/tgaer/agents/arc_agi3_scientist.py` (new)

```python
class Scientist:
    def __init__(self, model: str, api_base: str | None, ...) -> None: ...
    def infer(self, frame) -> Semantics | None:
        """Vision prompt (board PNG via grid_to_png_data_url + palette legend + the
        indices actually present) → strict JSON {avatar, keys, door, walls, verb}.
        Validates every index appears in the grid and verb ∈ {navigate, press}.
        Returns None on any failure (network / parse / invalid)."""

class ScientistPlannerAgent(Agent):
    def __init__(
        self,
        seed: int = 0,
        model: str = "openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
        api_base: str | None = "http://localhost:8000/v1",
        stall_limit: int = 8,
        **_: Any,
    ) -> None: ...
```

`act`: on `levels_completed` change → `Scientist.infer` (→ `Semantics` | `LS20_DEFAULT`),
cache; drive `KeyDoorController.step`. A stall counter increments when
`controller.made_progress()` is False; at `stall_limit` it re-queries the scientist once
(with stall context) and resets. The scientist is injectable so tests use a fake client.

### 5. Wiring, config, tests

- `dispatch.py`: register `"scientist": ScientistPlannerAgent` in `_ARC_AGI3_AGENTS`.
- `configs/experiments/arc_agi3_scientist.yaml`: kind=`scientist`,
  model=`openai/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`, `api_base: http://localhost:8000/v1`,
  game_id=`ls20-9607627b`, `max_actions: 80`, guards off.
- `tests/test_arc_agi3_scientist.py` against a **fake VL client** (no live model):
  - `Scientist.infer` — canned JSON → `Semantics`; garbage / index-absent-from-grid /
    bad verb → `None`.
  - Degradation — fake client raises → `LS20_DEFAULT`, and the scripted L0 still clears
    (reuses the planner's existing scripted-board fixture).
  - Verb branch — `press` emits the interaction action adjacent to the target;
    `navigate` covers the target.
  - Stall re-query fires after `stall_limit` no-progress steps; per-level cache means one
    `infer` per level, not per step.
  - Geometry extraction — the refactored `PlannerArcAgi3Agent`'s existing tests stay green.

## Out of scope

- Serving the VL model (assumed: vLLM on `:8000`, per the existing qwen3vl configs).
- L1+ navigation on the 5-cell lattice (separate known issue; tracked in PR #12).
- RL / fine-tuning the scientist (the `docs/specs/2026-06-14-arc-agi3-rl-design.md` track).
