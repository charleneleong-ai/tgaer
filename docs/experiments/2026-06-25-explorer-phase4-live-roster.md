# Explorer Phase 4 — live roster re-sweep (directed bootstrap, zero live lift)

**Date:** 2026-06-25 · **Branch:** `feat/arc-agi3-explorer` · Design: [`docs/specs/2026-06-23-arc-agi3-explorer-design.md`](../specs/2026-06-23-arc-agi3-explorer-design.md) · Prior: [`2026-06-24-explorer-phase3b-live-roster.md`](2026-06-24-explorer-phase3b-live-roster.md)

## Hypothesis

Phase 3b proved the induce→exploit loop is correct in-sim but unreachable live: win-induction only fires *after* a first win, and blind exploration never manufactures that first win on a real 64×64 lock grid (the [cold-start gap](2026-06-24-explorer-phase3b-live-roster.md)). Phase 4 ([`ed735d7`](../../commit/ed735d7)) closes the gap on paper: once the avatar + move lattice are induced from *controllability alone* (no win needed), [`_nav_affordance`](../../src/tgaer/agents/arc_agi3_explorer.py) steers toward the nearest salient object — a candidate key/door — so the first win is *sought*, not stumbled into. In-sim it beats blind 22→80 steps (8×8) up to 142→4280 (48×48, ~30×). **Claim under test:** directed bootstrap lifts the keyboard cluster (`ls20`, `lf52`) that scored **0** in Phase 3b.

## Setup

Identical harness to Phase 3b — game-agnostic [`ExplorerArcAgi3Agent`](../../src/tgaer/agents/arc_agi3_explorer.py) over the full 25-game roster, one scorecard, `MAX_ACTIONS=1000` — only the agent code changed (Phase 3b → Phase 4). A like-for-like A/B.

```bash
AGENT=explorer MAX_ACTIONS=1000 python experiments/sweep_roster.py
```

Scorecard `e28f763c-7cc9-4477-9c6d-cc2a73ca4e90` · raw rows: [`roster_results_explorer.jsonl`](../../experiments/roster_results_explorer.jsonl) · Phase 3b baseline preserved at [`roster_results_explorer_phase3b.jsonl`](../../experiments/roster_results_explorer_phase3b.jsonl).

## Result

| Metric | Phase 3b | Phase 4 | Δ |
|---|---|---|---|
| **Total levels completed** | **1** | **1** | **0** |
| Environments completed | 0 / 25 | 0 / 25 | 0 |
| GAME_OVER / NOT_FINISHED | 7 / 18 | 7 / 18 | 0 |
| Errors | 0 | 0 | 0 |

The aggregate is **identical**. Only scorer both runs: **`lp85`** (click-only, `available_actions=[6]`) — solves level 0 in ~5 clicks (`level_scores[0]=115`, beating the 17-action baseline) then forfeits level 1. The keyboard targets did not move:

| Game | Verb | win_levels | Phase 3b | Phase 4 |
|---|---|---|---|---|
| `ls20-9607627b` | keyboard (1–4) | 7 | 0 / NOT_FINISHED | **0 / NOT_FINISHED** |
| `lf52-271a04aa` | click+keyboard (1–4,6,7) | 10 | 0 / NOT_FINISHED | **0 / NOT_FINISHED** |

Both ran the full 1000-action budget with `levels_completed=0` and zero level-0 progress — same as Phase 3b.

## Verdict

**Directed bootstrap gave zero live lift — the in-sim 30× win did not transfer to a single real game.** The result is not noisy regression; it's pixel-identical to Phase 3b on every aggregate, which means the Phase 4 path almost certainly *never engaged* on the keyboard games. `_nav_affordance` is gated on the avatar + lattice being induced first; if that induction doesn't fire, it returns `None` and the agent falls straight back to the same blind `_nav_move`/`_choose` Phase 3b ran. The in-sim proof presented a clean single controllable sprite on an ≤48×48 grid — the real 64×64 games evidently don't surface a controllable avatar the current detector can pin within budget, so the bootstrap that depends on it is dead on arrival.

What this run **cannot** say is *which* stage fails — avatar never pins, lattice never completes, or affordance steers to the wrong object. The sweep measures outcome, not internals (`guard_hints_fired=0` is the only internal signal and it's uninformative here). Another paid blind sweep won't resolve that.

## Next move

**Instrument the induction pipeline, don't re-sweep.** Add per-step telemetry to a single offline `ls20` replay — log (a) whether the avatar pins and at which step, (b) the move-lattice size over time, (c) whether `_nav_affordance` fires vs falls back — to isolate the failing stage. That's a local, free diagnostic. The fix that follows depends on what it shows: a noisy-controllability avatar detector needs a longer confirmation window or a motion-consistency relaxation; a wrong-target affordance heuristic needs a better salience prior. Either way the lever is *making induction fire on real grids*, which Phase 4 assumed and this run shows it does not.
