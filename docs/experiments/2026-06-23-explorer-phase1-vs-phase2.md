# Explorer Phase 1 vs Phase 2 — roster sweep

**Date:** 2026-06-23 · **Branch:** `feat/arc-agi3-explorer` · Design: [`docs/specs/2026-06-23-arc-agi3-explorer-design.md`](../specs/2026-06-23-arc-agi3-explorer-design.md)

## Setup

Game-agnostic [`ExplorerArcAgi3Agent`](../../src/tgaer/agents/arc_agi3_explorer.py) over the full 25-game ARC-AGI-3
roster, `MAX_ACTIONS=500`, one scorecard. Phase 2 adds reset-on-death + fatal-edge avoidance
([`arc_agi3_env.py`](../../src/tgaer/envs/arc_agi3/arc_agi3_env.py) `reset_on_game_over`, agent `_fatal` set).

```bash
AGENT=explorer MAX_ACTIONS=500 python experiments/sweep_roster.py
```

Baselines: planner scores 1 level (`ls20` only); random ≈ 6 levels (field data).

## Result

| Metric | Phase 1 | Phase 2 | Δ |
|---|---|---|---|
| GAME_OVER (died) | 23/25 | **6/25** | −17 |
| NOT_FINISHED (full budget) | 2/25 | **19/25** | +17 |
| **Total levels** | **1** | **1** | **0** |

Only scorer both phases: `lp85` (1 level). Under Phase 2 it went `GAME_OVER@179` → `NOT_FINISHED@500` —
survived the full budget but still only reached level 1. `ls20` likewise: 0 levels, `GAME_OVER@186` → `NOT_FINISHED@500`.

Raw rows: [`roster_results_explorer_phase1.jsonl`](../../experiments/roster_results_explorer_phase1.jsonl) ·
[`roster_results_explorer.jsonl`](../../experiments/roster_results_explorer.jsonl) (Phase 2).

## Verdict

Phase 2 solved the death problem (death rate **92% → 24%**, 17 games stopped forfeiting) but produced **zero
level gain**. Staying alive ≠ winning: the explorer spends the retained budget wandering, not seeking. The
bottleneck has moved from **survival → goal-seeking**.

## Next move

**Phase 3 — win-relation induction.** Record the transition immediately preceding each `levels_completed++`,
induce a goal predicate ("reach value V" / "click value V"), exploit it via the BFS `Planner`. Phase 2 is the
prerequisite (can't induce a win-relation if you die before finding one; can't exploit a goal if death keeps
resetting the map). Generalization proof: re-solve `ls20` **without** the hardcoded `LS20_DEFAULT`.
