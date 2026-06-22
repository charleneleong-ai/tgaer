# ARC-AGI-3 Empirical Semantics — detect roles from the action stream, not a VL label

**Date:** 2026-06-19
**Status:** Design approved, pre-implementation
**Branch:** `feat/arc-agi3-empirical-semantics`

## Summary

The controller+scientist hybrid identifies each game's roles (avatar / key / door /
walls) so the reset-free `KeyDoorController` can navigate. Two strategies exist so far:
hardcoded `LS20_DEFAULT` (PR #12, clears LS20 1/7, scores 0 on the other 24 games because
the palette is LS20-specific) and the **VL scientist** (PR #13, Qwen3-VL labels roles
once per level). The live VL run exposed the scientist's core weakness: a confidently-wrong
but internally-consistent labelling (avatar tagged as an 84-cell wall blob) passes the
hallucination guard and *overrides* the trusted prior — it took a geometry trust-guard
(`avatar_is_sprite`) to recover the 1/7 floor.

That episode shows the real lesson: **role identity is observable from the environment
stream — we should not be guessing it from a static image.** The avatar is the thing that
moves when the agent acts; the key is the thing that disappears under the avatar; the door
is the cell whose contact ends the level. This spec adds `EmpiricalSemantics`: a detector
that *derives* `Semantics` from the action→observation stream the agent already produces,
demoting the VL scientist to a **cold-start hint** for the first few steps of a brand-new
game (before any action has been observed). Evidence always overrides the cold-start.

The win is the 24 zero-games: wherever the controller's navigation is sound, empirical
detection supplies correct semantics with no per-game tuning and no VL mislabel risk.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| System of record | **Empirical evidence; VL is cold-start only** (Q1·A) | The thing-that-moves is ground truth. VL gets the agent off the starting line on step 0, then evidence pins-and-overrides. Kills the mislabel failure mode at the root. |
| Avatar signal | **Action-conditioned translation** (Q2·A) | The avatar's motion is *correlated with my action* (action 1 → Δ₁, wall → Δ=0); an enemy/clock/animation moves regardless. Controllability is the definition of a player avatar and the one signal that generalises past LS20. Yields the controller's `delta` map in the same pass. |
| Key signal | **Disappearance-on-contact** (Q3·A) | "Collected on contact" — the component that vanishes the frame after the avatar overlaps it. Unambiguous: nothing else disappears under the avatar. |
| Door signal | **Level-increment-on-contact** (Q3·A) | The cell the avatar occupies when `levels_completed` ticks up. Terminal ground truth from the env; arrives only *after* the first clear, so level 0 leans on the cold-start. |
| Cold-start gap | **VL (or LS20 geometry heuristic) proposes candidate key/door until confirmed** (Q3·A) | Before a disappearance/level-tick has fired, navigate toward the cold-start's guess; each confirmation event promotes-and-pins a value to a known role. |
| Commit thresholds | **Asymmetric, per signal** (Q4·A): avatar = 2 consistent action→Δ; key = 1 disappearance; door = 1 level-tick | Match the threshold to each signal's noise profile: corroborate the continuous/inferred signal (avatar); trust the discrete/unambiguous env events (key, door) on first sight. |
| Lifecycle | **Semantics persist across levels of a game; reset only on a new episode/game** | All 7 LS20 levels share one palette, so a role pinned on L0 is reused on L1+. Only the controller's goal/phase state resets per level (`on_new_level`, already exists). |
| Integration | **New `empirical` agent kind; VL client injected, optional** | Keeps `scientist` and `planner` untouched and A/B-able. Detector is unit-testable against a synthetic frame stream with no model. |

## Architecture

```
observation ──> EmpiricalPlannerAgent.act
                  │
                  ├─ new episode? ─> reset detector; VL cold-start ─> provisional Semantics
                  │
                  ├─ EmpiricalSemantics.observe(prev_frame, action, frame, levels_completed)
                  │     ├─ avatar: did a component translate by a consistent action→Δ?  (≥2 ⇒ pin)
                  │     ├─ key:    did a component vanish under the avatar footprint?     (1  ⇒ pin)
                  │     └─ door:   did levels_completed tick while avatar on a cell?      (1  ⇒ pin)
                  │
                  ├─ Semantics = pinned roles, with cold-start filling any unpinned slot
                  │
                  └─ KeyDoorController.learn / .step  (unchanged — drives navigation)
```

`EmpiricalSemantics` is a pure, stateful detector with one entry point (`observe`) and one
accessor (`semantics()`); it holds no game I/O. The agent is the only thing that touches
the env and the VL client. The controller is reused verbatim.

## Components

### `EmpiricalSemantics` (new module `arc_agi3_semantics.py`)

Lives in its own module rather than `arc_agi3_grid.py`, which is already the large
geometry+controller+planner file; the detector is stateful and self-contained, so it
earns a sibling. It imports the geometry helpers (`components`, `field_box`, `in_field`)
from `arc_agi3_grid`.

State: per-role candidate evidence + pinned values; a per-action Δ tally; the previous
frame's component index (value → set of cells) for cross-frame diffing.

- `observe(prev, action, cur, levels) -> None` — one step of evidence:
  - **avatar:** for the value whose single in-field component translated, record `action→Δ`;
    when an action has ≥2 consistent non-zero Δ, pin avatar = that value. (The detector does
    *not* write the controller's `delta` map — the `KeyDoorController` owns its move-lattice
    via its own bootstrap probe; keeping a single source of truth avoids a coupled double-write.)
  - **key:** diff component sets prev→cur; a value present under/adjacent to the avatar in
    `prev` and *absent* in `cur` pins key.
  - **door:** if `levels` increased this step, the value of the cell the avatar occupied/was
    adjacent to last step pins door.
  - **walls** stay the controller's job — it already treats refused moves as the wall-oracle,
    so walls need no separate empirical pin (the `blocked` set + `sem.walls` fallback cover it).
- `semantics(cold_start: Semantics) -> Semantics` — pinned roles, with any unpinned slot
  filled from `cold_start`; verb defaults to `navigate` (press-verb detection is out of scope,
  see below).
- `confidence` introspection (which roles are pinned vs provisional) for logging/tests.

### `EmpiricalPlannerAgent` (new, in `arc_agi3_empirical.py`)

Mirrors `ScientistPlannerAgent`: owns a `KeyDoorController`, calls `EmpiricalSemantics.observe`
each step, and feeds `semantics(cold_start)` into `controller.learn/step`. Cold-start source:
an injected `Scientist` queried once on episode start (frame 0); if the VL client is absent or
returns `None`, the cold-start is the existing LS20 connected-component heuristic / `LS20_DEFAULT`.

### Wiring

New `empirical` kind in `evaluation/dispatch.py` + `configs/experiments/arc_agi3_empirical.yaml`
(mirrors the scientist config; VL `api_base` optional).

## Data flow & the cold-start handoff

1. **Episode start:** VL cold-start (if available) proposes provisional avatar/key/door so the
   controller can move on step 1. No evidence yet.
2. **Probe phase:** the controller's existing bootstrap probes each directional action once.
   Those are exactly the observations `EmpiricalSemantics` needs — by the end of probing the
   avatar is pinned from action-conditioned Δ (overriding the cold-start avatar if they differ).
3. **Navigate:** first key-collect pins key by disappearance; first level-clear pins door by
   level-tick. From level 1 on, all three roles are evidence-pinned and the cold-start is unused.
4. **Override rule:** a pinned value always supersedes the cold-start for that role. The
   `avatar_is_sprite` geometry guard becomes unnecessary for the avatar (we no longer guess it),
   but stays useful as a sanity filter on the *cold-start* VL label before any movement is seen.

## Error handling & edge cases

- **No cold-start (VL down, fresh game):** start from `LS20_DEFAULT`; the probe phase still pins
  the avatar within ~2 steps, so a wrong default self-corrects fast.
- **Avatar never moves (all-walls / wrong default avatar):** no action→Δ ever corroborates;
  detector stays provisional and the agent behaves exactly like today's degraded planner. No regression.
- **Ambiguous disappearance** (two components vanish same frame): require the vanished cell to be
  *under the avatar footprint*, not anywhere; if still ambiguous, leave key provisional.
- **Multi-index roles** (key = `{0,1}`): pin the *specific* value observed disappearing; accumulate
  a set across collects rather than overwriting.

## Testing

One area file `tests/test_arc_agi3_empirical.py`, split by sub-feature:

- `TestAvatarDetection` — synthetic 2-3 frame streams: a component that translates by a consistent
  action-Δ pins after 2 obs; a distractor that moves *independent of action* never pins; a 1-frame
  flicker does not pin.
- `TestKeyDetection` — a component vanishing under the avatar pins key; one vanishing elsewhere does not.
- `TestDoorDetection` — `levels_completed` increment pins the avatar-adjacent cell as door; no increment, no pin.
- `TestColdStartHandoff` — provisional VL semantics drive step 1; an evidence pin overrides a *disagreeing*
  cold-start; absent VL falls back to `LS20_DEFAULT`.
- `TestAgentIntegration` — `EmpiricalPlannerAgent` emits legal actions; semantics persist across a level
  bump; the injected fake-VL is queried once per episode, not per level.

Module-level helpers (`_frame`, `_sub`, picklable predicates) hoisted per the test conventions; parametrize
the noise/edge cases rather than copy-pasting.

## Out-of-scope follow-ups

- **Press-verb detection.** Empirical verb inference (navigate vs press) needs a separate signal — e.g.
  progress stalls under pure navigation but an interaction action changes state. Deferred; `navigate` is
  assumed, matching LS20.
- **L1+ movement-model ceiling.** The coarse move-lattice misalignment from PR #12 is a *navigation* limit,
  unaffected by better semantics. Separate follow-up.
- **Enemy/hazard semantics.** Detecting "things that end the episode on contact" (death) is a fourth role
  this spec doesn't cover; the avatar/key/door triad is the LS20-family scope.
