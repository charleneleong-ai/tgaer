# Explorer Phase 5 — ls20 induction-telemetry replay (the bootstrap engages, then oscillates)

**Date:** 2026-06-26 · **Branch:** `feat/arc-agi3-explorer` · Design: [`docs/specs/2026-06-25-ls20-induction-telemetry-replay-design.md`](../specs/2026-06-25-ls20-induction-telemetry-replay-design.md) · Prior: [`2026-06-25-explorer-phase4-live-roster.md`](2026-06-25-explorer-phase4-live-roster.md)

## Hypothesis

Phase 4 gave the directed bootstrap **zero** live lift and was pixel-identical to Phase 3b, so its verdict inferred the Phase 4 path *never engaged* — that [`_nav_affordance`](../../tree/feat/arc-agi3-explorer/src/tgaer/agents/arc_agi3_explorer.py#L368-L391) returned `None` because the real 64×64 grid never surfaces a controllable avatar the detector can pin within budget ("dead on arrival"). The sweep measures outcome, not internals, so it could not say *which* stage dies: avatar never pins, lattice never completes, or affordance steers wrong. This diagnostic instruments one real ls20 trajectory to find out — capture the live `(obs, action)` stream once, replay it free through a fresh deterministic explorer logging per-step `self.trace`.

## Setup

One live `ls20-9607627b` game captured at the agent boundary ([`capture_ls20.py`](../../tree/feat/arc-agi3-explorer/experiments/capture_ls20.py), `MAX_ACTIONS=1000`, `reset_on_game_over=True`, `guards=[]`), then re-fed through a fresh `ExplorerArcAgi3Agent` ([`replay_telemetry.py`](../../tree/feat/arc-agi3-explorer/experiments/replay_telemetry.py)). The replay asserts each replayed action equals the captured one, so the trajectory is a faithful reproduction (no non-determinism). Captures are gitignored; the run scored 0 levels over the full 1000-action budget — the same outcome Phase 4 saw on ls20.

```bash
PYTHONPATH=src python experiments/capture_ls20.py   # one live game (ARC API quota, not $)
PYTHONPATH=src python experiments/replay_telemetry.py
```

## Result

```
[replay] {'avatar_pins_at': 6, 'max_lattice': 4,
          'branches': {'probe': 4, 'choose': 1, 'affordance': 995}}
```

| Induction stage | Phase 4 inference | Phase 5 measurement |
|---|---|---|
| Avatar pins | never (assumed) | **yes — step 6**, value `12` |
| Move lattice completes | never (assumed) | **yes — size 4** (all directions) |
| `_nav_affordance` engages | never (assumed) | **yes — 995 / 1000 steps** |
| Levels completed | 0 | **0** (all 1000 steps at `levels=0`) |

Every stage Phase 4 assumed was dead is actually **healthy**. The avatar is pinned, the lattice is complete, and the directed bootstrap fires on 99.5% of steps. The failure is one layer up: of the 995 affordance steps, **985 are the vertical pair** (action 1 `down` ×549, action 2 `up` ×436) and only **10 are horizontal** (actions 3/4). **86% of affordance-to-affordance transitions are immediate `down↔up` reversals** — a two-step limit cycle. The avatar pins, then ping-pongs in place on one axis for ~994 steps, never traversing horizontally to reach the key/door.

## Verdict

**Phase 4's "never engaged / dead on arrival" verdict is refuted.** Induction works on real ls20 — avatar, lattice, and `_nav_affordance` all fire. The bug is in affordance **steering**, not induction: [`_nav_affordance`](../../tree/feat/arc-agi3-explorer/src/tgaer/agents/arc_agi3_explorer.py#L368-L391) re-selects the nearest salient target from scratch every step and [`_route`](../../tree/feat/arc-agi3-explorer/src/tgaer/agents/arc_agi3_explorer.py#L334-L350) returns only the first move of the path. With no target commitment and no anti-reversal guard, the chosen target flips between two cells straddling the avatar and the avatar oscillates between them — making exactly as much progress as blind exploration (none). That outcome-equality is why Phase 4's aggregate was pixel-identical to Phase 3b and misread as "the path didn't run": **identical outcome did not imply identical path.** The offline replay is what separated them.

## Next move

**Fix affordance steering, not induction.** The lever Phase 4 named ("make induction fire") is already satisfied; the real lever is breaking the `_nav_affordance` limit cycle. Candidates, cheapest first:

- **Anti-reversal guard** — forbid a primitive that undoes the immediately-prior move (kills the 2-cycle directly; smallest change).
- **Target commitment / hysteresis** — latch a chosen affordance target until reached or proven unreachable, instead of re-picking the nearest each step (removes the target-flip that drives the cycle).
- **Cycle → frontier fallthrough** — when affordance routing revisits a recent cell, yield to `_nav_move`/`_choose` frontier exploration so the avatar leaves the column.

Validate the fix the same free way: re-capture is **not** needed — re-run `replay_telemetry.py` against the existing capture after each change and watch the branch histogram lose its `down↔up` dominance and the avatar's horizontal-move share rise. Only re-sweep live once the replay shows the avatar traversing toward the goal.
