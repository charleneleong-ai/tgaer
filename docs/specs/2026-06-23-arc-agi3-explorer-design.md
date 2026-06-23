# ARC-AGI-3 Explorer — game-agnostic mechanic induction

**Date:** 2026-06-23 · **Branch:** `feat/arc-agi3-explorer` · **Status:** Phase 0+1 in progress

## Problem

The current controller ([`KeyDoorController`](../../src/tgaer/agents/arc_agi3_grid.py)) hardcodes one
win-model — *collect keys → walk to door* — with LS20 colour semantics. Across the 25-game roster it
scores on exactly **1 game** (`ls20`, 1/7 levels); the other 24 are flat zero. The VL scientist
([#17](../../src/tgaer/agents/arc_agi3_scientist.py)) regressed `ls20` to 0 — a rendered-PNG round-trip
loses the exact cell coordinates the BFS planner needs.

## What the field does (ARC-AGI-3 2025 preview)

Verified across the ARC Prize technical report, the Graph-Based Exploration paper (arXiv 2512.24156),
and the 1st/3rd-place public solutions:

- **Algorithmic exploration ≫ frontier LLM ≫ random.** Training-free graph exploration solved ~19
  levels under a 4k-interaction budget; the frontier LLM+DSL baseline solved 5 — *below* random's 6.
- **No LLM-finetune traction.** 1st place (StochasticGoose) is a *small CNN trained from scratch at
  test time, per game, reset per level*, predicting which actions change the frame — an exploration
  prior, **not** a finetuned foundation model. The benchmark is anti-memorization by design.
- **Decision:** build a general explore→induce→exploit loop. Do **not** GRPO-finetune the 35B LLM.
  The LLM is demoted to an optional cold-start hint, off the critical path.

## Architecture — `explore → induce → exploit`

A single `ExplorerAgent` that knows nothing about a game's win condition on entry:

1. **Explore** — build a directed **state graph** of frame signatures; head toward the nearest state
   with untested actions (frontier-directed), instead of flailing.
2. **Induce** *(Phase 3)* — record the transition that immediately precedes every `levels_completed++`
   → a reusable goal predicate ("reach cell of value V" / "click value V").
3. **Exploit** *(Phase 4)* — navigate to the induced goal with the existing BFS
   [`Planner`](../../src/tgaer/agents/arc_agi3_grid.py).

### Reused as-is
`components`, `field_box`, `in_field`, `cells`, `avatar_is_sprite`, `Planner`, `to_action`,
`COMPLEX_ACTION_ID` from `arc_agi3_grid.py`; the controllability/consume/vanish detectors in
[`EmpiricalSemantics`](../../src/tgaer/agents/arc_agi3_empirical.py) (promoted to emit a goal predicate
in Phase 3).

### New module `arc_agi3_explorer.py`
| Component | Responsibility |
|---|---|
| `frame_signature(arr)` | stable hash of the in-field region → graph node identity |
| `proposals(arr, available)` | ordered action primitives at a node; `ACTION6` → salience-ranked click targets (component centroids) |
| `StateGraph` | nodes = signatures w/ untested primitives; edges = (sig, primitive) → sig; shortest path to nearest frontier |
| `ExplorerAgent(Agent)` | explore→induce→exploit orchestration; per-level graph reset |

## Phasing
0. **Budget knob** *(this PR)* — `max_actions`/`max_steps` configurable in `sweep_roster.py` via
   `MAX_ACTIONS` env var. The real benchmark budget is far above tgaer's default 80 (winners used 4k);
   without this the explorer looks worse than it is.
1. **State-graph frontier explorer** *(this PR)* — systematic exploration, no win-model. *Metric:
   beat random (>6 levels); get the six `[6]`-click games off zero.*
2. Action-effect classification — generalise move-learning to all verbs incl. `ACTION6` coords.
3. Win-relation induction — re-solve `ls20` **without** hardcoded `LS20_DEFAULT` (the generalization proof) + ≥1 new game.
4. Exploit via `Planner` — multi-level games complete >1 level.
5. *(optional)* StochasticGoose-style per-game CNN exploration prior — only if 1–4 plateau.

## Validation
`ExplorerAgent` over the 25-game roster → `roster_results_explorer.jsonl`, head-to-head vs the current
1/25 and random's 6. Phase 3 must reproduce `ls20` without hardcoded semantics.

## Explicitly out of scope
GRPO/PPO finetuning of the 35B LLM; VL/pixel perception; any approach that memorizes the public games.
