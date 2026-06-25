# Explorer Phase 3b — live roster sweep (the cold-start wall)

**Date:** 2026-06-24 · **Branch:** `feat/arc-agi3-explorer` · Design: [`docs/specs/2026-06-23-arc-agi3-explorer-design.md`](../specs/2026-06-23-arc-agi3-explorer-design.md) · Prior: [`2026-06-23-explorer-phase1-vs-phase2.md`](2026-06-23-explorer-phase1-vs-phase2.md)

## Hypothesis

Phase 3b wired win-induction end-to-end: induce the avatar (controllability), its move lattice, and the navigate goal (value that vanishes under the avatar on a level-up), then exploit via the BFS [`Planner`](../../src/tgaer/agents/arc_agi3_grid.py). It re-solves a synthetic `ls20` sim with **no** `LS20_DEFAULT` prior ([`TestLs20WithoutHardcodedSemantics`](../../tests/test_arc_agi3_explorer.py)). **Claim under test:** that induce→exploit loop lifts levels on the *live* 25-game roster — especially the keyboard/navigate games the LS20-tuned planner can't generalise to.

## Setup

Game-agnostic [`ExplorerArcAgi3Agent`](../../src/tgaer/agents/arc_agi3_explorer.py) over the full 25-game roster, one scorecard, `MAX_ACTIONS=1000` (2× Phase 2, to give induction room to fire on real 64×64 grids).

```bash
AGENT=explorer MAX_ACTIONS=1000 python experiments/sweep_roster.py
```

Scorecard `0da01377-b629-4506-bb24-7b402b029d01` · raw rows: [`roster_results_explorer.jsonl`](../../experiments/roster_results_explorer.jsonl). Baselines: planner scores 1 level (`ls20` only, hardcoded); random ≈ 6 levels (field data).

## Result

| Metric | Value |
|---|---|
| Games | 25 |
| **Total levels completed** | **1** / 183 available |
| Environments completed | 0 / 25 |
| GAME_OVER / NOT_FINISHED | 7 / 18 |
| Errors | 0 |

Only scorer: **`lp85`** — a **click** game (`score=16.0`, 1/8 levels, `NOT_FINISHED`). The scorecard splits cleanly by verb cluster:

| Cluster | Envs | Actions | Levels completed |
|---|---|---|---|
| click | 7 | 1337 | **1** |
| keyboard | 4 | 644 | **0** |

The ls20-family games — exactly the induce→exploit targets — both scored **0** at the full 1000-action budget:

| Game | Verb | win_levels | levels | state |
|---|---|---|---|---|
| `ls20-9607627b` | keyboard (1–4) | 7 | 0 | NOT_FINISHED |
| `lf52-271a04aa` | click+keyboard (1–4,6,7) | 10 | 0 | NOT_FINISHED |

## Verdict

**The induce→exploit loop never engaged on a live game — it's gated behind a first win that blind exploration can't manufacture.** Win-induction only fires *after* a `levels_completed++` (`_observe_door` runs on a level-up; the click goal is learned post-advance). So level 1 must be won blind, and only then does exploitation light up. On a real 64×64 grid that first win is the wall:

- The **one** bootstrap (`lp85`) was a **click** game — blind salience-clicking has a real chance of landing a winning cell. Every other click env stalled, but the cluster scored 1.
- **No keyboard/navigate game bootstrapped a single level** (`ls20`, `lf52`, and the 2 other keyboard envs all 0). Blind directional wandering on a key→door grid won't stumble through the key→lock→door sequence in budget — so `_nav_move` stayed dormant the whole game.

This is the **cold-start bootstrap gap**: the Phase 3b machinery is correct (proven in-sim) but unreachable live, because it *amplifies* a first win rather than *producing* one. The synthetic proof passed only because the 8×8 no-key sim made the blind first win trivial (~70 steps); the live key/lock games at 64×64 don't.

## Next move

**Phase 4 — directed first-win bootstrap (already landed: [`ed735d7`](../../commit/ed735d7)).** Once the avatar + lattice are induced (controllability only, no win needed), [`_nav_affordance`](../../src/tgaer/agents/arc_agi3_explorer.py) steers toward the nearest salient object — a candidate key/door — so the first win is *sought*, not stumbled into. Cost becomes path-length (linear) not area (quadratic): on a 2-level locked sim, directed beats blind 22→80 steps (8×8) rising to 142→4280 (48×48, 30×). **Re-sweep with Phase 4** to test whether directed bootstrap lifts the keyboard cluster (`ls20`, `lf52`) that scored 0 here — the live validation this run could not provide.
