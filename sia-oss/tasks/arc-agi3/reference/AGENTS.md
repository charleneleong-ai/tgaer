# Persistent memory — read this before proposing anything

This file is your cross-generation memory. It is regenerated after every
generation by an external supervisor script (`sia-oss/supervisor.py`, run
outside your sandbox) and copied fresh into your working directory. **Every
entry below reflects a hypothesis that was already tried and measured** —
either in this OSS-sia harness or in an earlier tool (SIA Foundry) against
the identical suite. Re-proposing a closed item wastes a generation; the
supervisor will flag it if it happens again.

Baseline: **RHAE = 0.4263%** across the 8-case suite — the official
ARC-AGI-3 metric (level-index-weighted, capped by levels completed; see
`task.md` Objective). `lp85` and `ls20` pass their gate; the other six do
not. A genuine improvement raises RHAE without dropping `lp85` or `ls20`'s
pass status. The old unweighted `total_level_score` (131.08) is no longer
the metric — it was almost entirely lp85's capped level 1, which the real
metric values at 2.8%. **14 generations in, only 1 has ever attempted a
real change to `explorer.py` (it regressed) — most default to
reading/scaffolding without ever committing to an edit. Don't do that: pick
the directive below and implement it.**

## Closed — do not retry any of these (condensed; ask if you need detail)

| Family | Result |
|---|---|
| Exploration-gate throttling (4 variants) | All net-negative |
| Position-keyed `StateGraph` identity (5 variants) | Fixes `sp80`, breaks `ls20` every time |
| Per-cell frame-signature denoising (all variants, incl. safety-guarded) | Best case -0.11; safety-guarded = exactly baseline. Whole avenue closed, not just configs |
| Raise `click_targets()` cap 12→25 | No change on `bp35` |
| Click-based routing fallback for off-lattice targets (run_4) | Regressed to 130.53 (broke `sc25`); too broad a trigger — see note below if revisiting |
| Scaffold-only / no explorer.py change (runs 1, 2×2, 9, 12) | No-op every time — doesn't move the score |
| Clearing `_blocked` on level change | Exactly baseline. 7 of 8 games never finish level 1, so `_on_new_level` fires once in the whole suite — every cross-level state fix is near-untestable here |
| Repeat count folded into `_live`'s sort key | 0.3516 (-0.075). Cost `ls20`/`sp80`/`sc25` their first level: re-ordering between visits to one signature desynchronises the per-signature untested set, exactly as `_live`'s docstring warns |
| Per-level click-repeat filter (`_unspent`, swept 16→256) | Knife-edge, not an improvement. Exactly baseline at every threshold except 64/68, where `lp85` clears level 2 (0.5089). A width-8 window in a range of 250 is a lucky perturbation of one deterministic rollout — do not ship a tuned constant |
| Learned chrome mask for `_learn_inert` (per-cell volatility, excluded cells changing >50% of steps) | Exactly baseline. The mask *works* — 104 chrome cells correctly identified on `lp85`, active 562 of 591 steps — but it buys nothing, for the reason in the note below. Do not retry masking, cropping, or denoising to fix `lp85` |

**`--seed` does not perturb this agent.** Seeds 0-3 give byte-identical
scorecards, so a seed sweep cannot separate a real effect from a lucky one.
Robustness has to be argued from mechanism or from a parameter sweep instead.

## Do this — the directive for this generation

**Target `lp85` level 2. It is 81.5% of the whole metric and it is reachable.**
Under RHAE, `lp85` alone contributes 2.778 of the 3.41 total env-score, and it
is **cap**-limited at 1/8 levels — its efficiency is already pinned at the 1.15
ceiling, so no efficiency work on it can ever pay. Only a second level can, and
that one level is worth +0.69pp RHAE: the benchmark goes 0.4263% → ~1.12%, more
than every other game combined. Every other game is worth ≤0.09pp.

A trace of the budget says why it fails, and it is not a tuning problem:
`lp85` clears level 1 in 10 actions, then spends 591 on level 2 without
finishing. On that board `_det.avatar` is never induced — so there is no
lattice, no goal value, no navigation, and every one of the 591 actions is a
click. 542 of them land on the *same cell* (`grid[18,20]`), because
`frame_signature` mints a fresh state every step, so `untested_at(sig)` always
returns a full list and `_choose` returns the same top-salience target forever.
`_stalls` rotation never runs, because nothing is ever exhausted.

**Why that cell keeps looking worth clicking — measured, not inferred.**
Rendering the frames (`measure.py --frames --focus lp85`) shows a rectangular
ring of cells that recolours every frame, *inside* the play field, which is why
the field-box crop cannot exclude it. But masking that chrome out and
re-testing showed the clicked cell is **not inert**: with chrome excluded, the
click still changes the board every time, and its `_inert` count stays 0.

So the action is *effective but unproductive* — a cycle, not a no-op. That is
the whole lesson: `_inert` asks "did anything change?", and on this board the
answer is honestly yes. **Novelty is not progress, and no amount of change
detection will separate them.** Only a notion of progress can — distance to a
goal, or a state abstraction under which the cycle is visibly a cycle.

So the fix is a real goal signal on a click-only, avatar-less board.

**Implement goal-directed pathfinding instead of frontier-BFS.** `act()`
currently only routes over edges the frontier search already opened; it
never plans a route toward a *specific* target once `click_targets()`
identifies one. An independent A*-over-world-model agent using exactly this
mechanism cleared all 25 public ARC-AGI-3 games in 45% fewer actions than
the human median — [pbshgthm/arc-skill](https://github.com/pbshgthm/arc-skill).
This is the highest-confidence untried direction; implement it, don't spend
turns deciding whether to. Read `act()`/`_choose()`/`click_targets()` once,
then write and test the change.

## Also untried, lower priority (only if turns remain)

- **Explicit explore/solve phase split** — bounded pure-exploration budget
  first, then switch to goal-directed search over the discovered graph,
  instead of interleaving both in one loop. Reaches RHAE=0.30 on the full
  55-game set in a published non-LLM technique
  ([arXiv:2605.25931](https://arxiv.org/pdf/2605.25931)).
- **Action-effect prediction as a staleness gate** — predict the frame delta
  before committing to an action from prior observed transitions; if it
  diverges, treat the local graph as stale. Cheap model-free analog of
  arc-skill's falsifiability constraint.
- **Cross-level structure caching within a game** — `StateGraph` resets per
  level; a level's lattice/object semantics likely carry over to the next
  level of the same game.
- **Cold-start bootstrap for `sk48`/`cn04`** (0 levels ever cleared) — force
  interaction with every distinct salient object before falling back to
  frontier search.
- If revisiting click-based routing for `bp35`: prove the target is
  unreachable by the current lattice (attempt the route, fail, *then* click)
  rather than pattern-matching the lattice's shape — the naive version broke
  `sc25` because its lattice looked one-axis while its real routing still
  worked.

## How to update this file

If you land on a new closed result (measured, reverted), the supervisor
will append it to the table above after your generation ends — you do not
need to edit this file yourself. Focus turns on the directive above.
