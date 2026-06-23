# ARC-AGI-3 `click` verb — coordinate-click capability for the controller

**Date:** 2026-06-22
**Status:** approved design, pre-implementation
**Scope:** capability + harness (no live roster-win claim)

## Summary

The `KeyDoorController` can only emit directional keys (action ids 1–4) and the
press interaction keys (5/7). 18 of 25 roster games offer **ACTION6** (the
coordinate-click), and the directional-only controller scores 0 on all of them.
This adds a **`click` verb**: the controller points ACTION6 at a chosen target
cell instead of navigating onto it. The ACTION6 plumbing already exists
(`ArcAction.x/y`, `ArcTransport.act`, `COMPLEX_ACTION_ID`); the gap is *choosing
a purposeful click target* and *carrying coordinates out of `step`*.

This is a **capability**, not a roster solver. We validate the mechanic on a real
game but do not claim wins across the click roster (most click games have richer
or game-specific mechanics — see Probe findings).

## Probe findings (live, 2026-06-22, no VL server)

`experiments/probe_click_mechanic.py` sent scripted ACTION6 clicks at detected
object centroids and diffed the returned frame. Two unknowns pinned as fact:

| Unknown | Result | Evidence |
|---|---|---|
| **Coordinate convention** | `ACTION6 x = column, y = row` | `tn36`: click `(x=col=26, y=row=45)` changed array cell `[45][26]` `1→5`; swapped ordering touched an unrelated single cell. |
| **Mechanic** | **direct-click** — clicking an object's cell affects that object | `tn36` `1→5` (toggle/select), `r11l` target `1→15`. |

Caveats recorded honestly: `r11l` reacts with a large cascade (not a simple
toggle); `ft09`/`lp85` had inert targets (background pockets / top-row legend
tiles) — clicking did nothing. So **direct-click-at-target is a real, validated
capability, but not every click game is a one-click toggle**. The convention is
the standard image convention, consistent with `to_action`'s symmetric centre
click.

## Architecture

Three localized changes; no new module.

### 1. `Semantics.verb` gains `"click"` (`arc_agi3_grid.py`)

The role values (avatar/keys/door) are unchanged. `click` means: emit ACTION6 at
the current target's centroid using the two-phase key→door target selection
already used by `navigate`. A `CLICK_DEFAULT` Semantics constant (sibling of
`LS20_DEFAULT`, `verb="click"`) is added so configs and tests can select the verb.

**Who sets `verb="click"`:** in this scope the verb is **supplied by Semantics /
config**, not auto-detected. The empirical/planner detectors still produce
navigate semantics, so a click game is exercised by passing `CLICK_DEFAULT`
(via a config or test), not by the agent inferring "this is a click game".
Auto-routing navigate-vs-click from `available_actions` is detection → it belongs
to the deferred "generalize empirical to clicks" scope and is **out of scope here**.

### 2. `KeyDoorController.step` returns `ArcAction` (was `int`)

The controller owns the full decision including coordinates. `step` returns an
`ArcAction` directly:
- navigate/press paths return `ArcAction(id=aid)` (no coords) — behaviour
  unchanged, just wrapped.
- click path returns `ArcAction(id=6, x=col, y=row)` at the target centroid.

`to_action(int)` stays for the agents' no-frame early-return path.

### 3. `click` logic in `step`

Mirrors the `press` branch but emits a click instead of navigating:
```
if sem.verb == "click":
    target = nearest key while keys remain, else door   # reuse last_key_goal / _door
    if target is not None and 6 in avail:
        r, c = round(target)
        return ArcAction(id=6, x=c, y=r)                 # x=col, y=row
    return to_action(self._fallback(avail, move_avail))  # no target / no ACTION6
```
No navigation, no BFS — direct click at the target. Reuses `_keys`, `_door`,
`find_role`, `last_key_goal`; the "click-target detector" is the existing
role-finder, no new machinery.

### Agents

`PlannerArcAgi3Agent.act` and `EmpiricalPlannerAgent.act` change their tail from
`return to_action(self._ctl.step(...))` to `return self._ctl.step(...)` (step now
returns an `ArcAction`). The `last_reply` log line is preserved.

## Data flow

```
frame → arr → controller.step(arr, sem, avail)
   navigate/press → ArcAction(id=aid)
   click          → ArcAction(id=6, x=col, y=row) at target centroid
→ env.act(id, x, y) → ACTION6 POST with {x, y}
```

## Testing (TDD, synthetic frames — deterministic, CI-safe)

One area file (extend the existing controller test module), grouped by sub-feature:
- **`click` verb emits the right coordinate** — synthetic frame with a key at
  `(r,c)`, `verb="click"` ⇒ `ArcAction(id=6, x=c, y=r)`; two-phase: clicks the
  door once keys are gone.
- **fallbacks** — ACTION6 absent from `avail`, or no target found ⇒ falls back to
  a keyboard/move action, never crashes.
- **return-type contract** — navigate + press still return a well-formed
  `ArcAction` (no regression); a parametrized check over the three verbs.

Probe report is empirical grounding, not a CI test (needs the live API).

## Out of scope

- No roster-win claim; no sweep.
- No empirical *click-target* auto-detection from the action→observation stream
  (that was the deferred "generalize empirical to clicks" scope).
- No click-to-move or select-then-act mechanics — direct-click only, as the probe
  validated. The other two conventions stay unimplemented (YAGNI) until a game
  demonstrably needs them.

## Files

- `src/tgaer/agents/arc_agi3_grid.py` — `click` verb, `step → ArcAction`
- `src/tgaer/agents/arc_agi3_planner.py` — return `step()` directly
- `src/tgaer/agents/arc_agi3_empirical.py` — return `step()` directly
- `tests/...` — click-verb + return-type tests
- `experiments/probe_click_mechanic.py` — live probe (untracked, already written)
