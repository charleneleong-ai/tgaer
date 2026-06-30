# Explorer Phase 7 — position memory breaks the ls20 limit cycle (first level cleared)

**Date:** 2026-06-30 · **Branch:** `feat/arc-agi3-position-memory` · Prior: [`2026-06-26-explorer-phase6-affordance-cycle-fix.md`](2026-06-26-explorer-phase6-affordance-cycle-fix.md)

## Hypothesis

Phase 6 landed a strict-path planner fix in a closed-loop sim but was never confirmed live. The Phase 7 live re-sweep settled it: on a fresh 1000-step `ls20-9607627b` capture through the merged agent, the telemetry was **bit-identical to pre-fix** — `{affordance: 995, choose: 1}`, 87% immediate down↔up reversals, 0 levels. The strict-path guard never fires on the real board because the salient decoys are *reachable*, so [`Planner.path`](../../tree/feat/arc-agi3-position-memory/src/tgaer/agents/arc_agi3_grid.py) returns a valid route every step. The cycle is one layer deeper, and signature-space reasoning can't reach it.

## Setup

Two diagnostics localised the real mechanism:

- **Branch-distribution replay** of the live capture: [`_nav_affordance`](../../tree/feat/arc-agi3-position-memory/src/tgaer/agents/arc_agi3_explorer.py#L375) is a *memoryless greedy* controller — it re-selects the nearest salient target every frame. Two salient cells straddling the avatar flip which is nearest every step (stepping onto one occludes it), so it re-issues the same move forever, starving [`_choose`](../../tree/feat/arc-agi3-position-memory/src/tgaer/agents/arc_agi3_explorer.py#L413), the cycle-free frontier explorer (995 vs 1 step).
- **Signature-vs-position counting** on the same capture: the avatar occupies ~30 distinct cells but the board produces **741 distinct frame signatures** (193 signatures for 14 cells in a 200-step window). Incidental per-frame churn fragments one position into many signatures, so *every* signature-keyed memory — `_choose` and a first attempt that gated affordance on the `StateGraph`'s per-signature untested set — is blind to the loop. The cycle lives in avatar-**position** space; the agent's memory lived in fragmented-signature space.

A bench across the sim battery confirmed the dead end: anti-reversal, untested-at-signature, and target-commitment each only enlarge the trap (and target-commitment *regresses* the Phase 6 phantom sim, 2→1). The fix has to be position-keyed.

## Result

[`_nav_affordance`](../../tree/feat/arc-agi3-position-memory/src/tgaer/agents/arc_agi3_explorer.py#L375) now refuses a move that lands the avatar on a recently-occupied cell (`self._recent`, the last `_RECENT_CELLS=8` positions, updated every step, cleared on a new level); [`_nav_move`](../../tree/feat/arc-agi3-position-memory/src/tgaer/agents/arc_agi3_explorer.py#L368) (the exploit) is exempt. Live re-sweep on `ls20-9607627b`:

| metric | pre-fix | Phase 7 |
|---|---|---|
| immediate reversals | 87% | **28%** |
| frontier `_choose` / affordance | 1 / 995 | **709 / 287** |
| distinct avatar cells | 30 | **45** |
| **levels cleared** | **0** | **1** |

The first nonzero ls20 score. Breaking the position cycle unstarved the proven frontier explorer (1→709 steps), which is what actually traversed the board and cleared the level — the fix walks the agent off the cold-start cliff rather than replacing the explorer. Closed-loop [`_StraddleDecoySim`](../../tree/feat/arc-agi3-position-memory/tests/test_arc_agi3_explorer.py) adds a per-frame blinker so signatures never repeat (faithful to the real churn); position memory solves it (RED 0 → GREEN 2) where every signature-space fix cannot. 192 tests green; ls20 / lock / phantom unregressed.

## Verdict

**The limit cycle is broken in the right space.** Three prior fixes (strict-path, anti-reversal, untested-at-signature) all reasoned in signature or reachability space and only enlarged the trap; the position-keyed veto is the first to escape it, validated live, not just in sim. The journey corrected two sim-vs-real gaps that bit Phases 6–7: the live re-sweep is now the gate before any "cycle fixed" claim, and the regression sim carries per-frame churn so it can't be satisfied by a signature trick.

## Next move

**Phase 8 — position-keyed exploration.** Post-bootstrap the agent now spends most steps in `_choose`, which is *still* signature-keyed and churn-blind (TODO flagged at [`frame_signature`](../../tree/feat/arc-agi3-position-memory/src/tgaer/agents/arc_agi3_explorer.py#L50)). The same 741-signature fragmentation can re-form a cycle there on a longer board; position memory only protects the cold-start window. Re-keying the `StateGraph` on avatar position (or a denoised signature) so the frontier explorer survives churn is the real fix — and the lever that should take ls20 past one level.
