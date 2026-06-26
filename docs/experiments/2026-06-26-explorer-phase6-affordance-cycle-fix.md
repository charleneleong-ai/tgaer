# Explorer Phase 6 — strict planner routing breaks the affordance limit cycle

**Date:** 2026-06-26 · **Branch:** `feat/arc-agi3-explorer` · Prior: [`2026-06-26-explorer-phase5-ls20-induction-telemetry.md`](2026-06-26-explorer-phase5-ls20-induction-telemetry.md)

## Hypothesis

Phase 5 localised the ls20 failure to `_nav_affordance` steering, not induction: the avatar pins, the lattice completes, but affordance fires 995/1000 steps in a vertical limit cycle. Tracing the cycle to its source: [`Planner.path`](../../tree/feat/arc-agi3-explorer/src/tgaer/agents/arc_agi3_grid.py#L322) ran a BFS whose *arrival* test is "footprint covers the goal cell", but when no node covered the goal it returned a **best-effort approach** path to the closest reachable cell instead of `None`. The induced move lattice has **stride 5**; the salient click-targets sit 2 cells off it (rows 13/18 vs the avatar's reachable rows 15/20), so they are *approachable but never coverable*. With two such phantoms straddling the avatar, the nearest flips every step and `_route` ping-pongs forever. **Claim under test:** making `Planner.path` return `None` for an uncoverable goal — honouring its own docstring — makes affordance skip the phantoms and yield to frontier exploration, breaking the cycle.

## Setup

A closed-loop sim is required: the Phase 5 replay is fixed-trajectory and `replay_traces` asserts replayed == captured, so a behaviour-changing fix diverges from the old capture by design — it can localise the bug but cannot validate the fix. [`_StridePhantomSim`](../../tree/feat/arc-agi3-explorer/tests/test_arc_agi3_explorer.py) reproduces the mechanism minimally: a stride-5 avatar, two off-stride decoys (value 7) straddling it, and a real key(5)→door(9) reachable on-stride to the side. [`TestAffordancePhantomEscape`](../../tree/feat/arc-agi3-explorer/tests/test_arc_agi3_explorer.py) drives a fresh explorer over it.

## Result

| | levels solved | steps | affordance branch |
|---|---|---|---|
| Baseline (best-effort path) | **0** | 120 (budget) | dominates — phantom cycle |
| Fix (strict path) | **2** | ~14 | yields once decoys exhausted |

The one-change fix ([`e36de1b`](../../commit/e36de1b)) drops the best-effort fallback. Re-running the **real ls20 capture** through the fixed agent (no assertion — the fix changes its choices) collapses the affordance share **995 → 74** and hands control to frontier exploration (`choose` **1 → 922**): on real frames the agent stops chasing phantoms. Full suite **191 passing**, including the empirical planner — the other `Planner.path` caller, which already treated a `None` route as "no path", so strict mode is safe there.

## Verdict

**The limit cycle is fixed at its root** — a planner that no longer pretends an unreachable goal is reachable. Affordance now steers only toward targets the lattice can land on, and falls through to frontier exploration when none qualify. The fix is a net deletion (the best-effort tracking is gone), aligning `Planner.path` with its stated arrival contract.

## Next move

**Live re-sweep is the real-world confirmation** (closed-loop sim + fixed-trajectory replay both fall short of a true live run). Re-capture is also worthwhile: the existing `ls20_capture.jsonl` predates the fix, so it no longer replays faithfully — a fresh capture would let the telemetry diagnostic re-localise whatever stage now binds (frontier coverage on a 64×64 grid is the likely next wall). Open question surfaced by the trace: the stride-5 lattice may itself be coarse for a 64×64 board — if real progress needs sub-stride alignment, the move-induction granularity is the Phase 7 lever.
