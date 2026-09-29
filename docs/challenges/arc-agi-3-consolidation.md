# Learning the dynamics, and what survives a level

What the agent should keep between levels and games, derived from the metric rather
than from intuition. Written 2026-09-28 after nine local levers closed.

## The metric decides what knowledge is worth

A level scores `min(cap, (human / ours)^2)`, and a game averages its levels **weighted
by level index over all its levels**. Two consequences that set the whole design:

- **Later levels are worth more.** In an 8-level game, level 1 is `1/36` of the score
  and levels 2-8 are `35/36`.
- **Actions on a level you never clear are free** — the numerator is zero either way.

So exploration early is cheap and exploitation late is valuable. The right shape is to
**spend actions on the first levels building a model of the game, then run near
optimally on the later ones**. The agent currently does the opposite: it is uniformly
greedy, and it deletes most of its model at every level boundary.

## Three tiers, split by what a reset actually falsifies

| tier | contents | reset when | why |
| --- | --- | --- | --- |
| **A. board state** | `StateGraph`, `_plan`, `_churn`, `_walk_novelty`, `_prev_*` | every level | a signature **names a board**; on a new one it means nothing |
| **B. game model** | what an action *does* — `_inert` by (action, context), inverse pairs, goal/role induction | every game | the controls do not change between levels |
| **C. cross-game prior** | action arity, movement actions coming in inverse pairs, click purity by (colour, size) | never | this is what can transfer to the 110 unseen games |

The bug this exposes: **`_inert` is tier B sitting in tier A.** `_on_new_level` wiped it
with the graph, so the agent's most load-bearing mechanism (`-0.2177pp` when removed,
and the only one generalising per primitive *across* states) was rebuilt from nothing
every level — even though `replay_effects` measured that key predicting
changed-or-not at **60-100%** and generalising across states on **25 of 25 games**.

Tier C is the one that earns a submission. `publicScore` is reproducible and a change
that is a **local no-op can still be an out-of-distribution gain** — colour-agnostic
role inference was bit-identical on all 25 local games and moved the score `+0.01`.

## The consolidation step

At a level boundary, before discarding tier A:

1. **Distil, then drop.** Fold the level's graph into tier B statistics — which
   primitives changed nothing, which pairs of simple actions undid each other, which
   colours behaved as goals — then throw the graph away.
2. **Carry tier B forward unchanged.** It describes the game, not the board.
3. **Age rather than trust.** Demotion is soft (a sort key, not a filter), so evidence
   that has gone stale costs ordering and is outvoted as the new board reports its own.
   Do not promote tier B to a hard filter.

## What to measure, not what to assume

RHAE is a poor test of transfer: it mixes in which levels happened to clear. The direct
measurement is **actions-per-level against the human baseline, regressed on level
index**. If knowledge compounds, later levels should get relatively cheaper.

Today it does not. `tu93` runs 383 -> 194 -> 886 actions and `lp85` 10 -> 364 -> 100 ->
777 — no downward trend, exactly as wiping tier B every level predicts. **That slope is
the number a consolidation mechanism has to move**, and it is a better target than RHAE
because it isolates transfer from luck.

## Which game to aim at

Not `tu93`. Three routing mechanisms each took it from 5/5 seeds to 0/5, and the last
of them moved **no other game at all** — it clears on a knife-edge trajectory, so its
26% of the ceiling is not capturable by changing how it moves.

`lp85` is the target: 37.8% of the ceiling at 18.7% capture, and **flooded rather than
stranded** — it never fails to find a route, it drowns in branching (1647 states, 15950
untested pairs). Which candidate is tried first is the whole game, so tier B carry-over
and ordering are the levers that can reach it.
