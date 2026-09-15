# World model + offline planning — scope

Eighth attempt at the LLM line, and the first one that changes the **object the
model produces**. Read `project-arcagi3-llm-as-programmer` and the closure note
on `feat/arc-agi3-llm-codegen` first: seven attempts, two models, six validator
designs, zero promotions. This is not a retry of that. It is scoped with a
zero-LLM kill switch precisely so it cannot drift into an eighth.

## Why now

Three measurements this session, all on the 25-game set at merged `main`:

| finding | consequence |
| --- | --- |
| Budget is flat above ~600 actions (0.1879% at 600, 0.1886% at 2400 **and** 6000) | more search does not pay |
| tu93's whole leverage is +0.019pp even at 4x | the worst offenders are worth nothing |
| lp85 L3's 100 actions are near-irreducible blind-search cost | the **best** lever is also closed to better search |

lp85 L3 is the specific case that motivates this. It costs 100 real actions:
one 80-action budget exhaustion, then a **19-action win**. The winning tail is
89% novel states, so the agent was not sitting on knowledge it failed to use —
it genuinely had to search. And `_choose` already dives depth-first, so there is
no scheduling fix. Blind search in the real environment costs what it costs.

**But the metric only charges *real* actions.** `S_l = min(1.15, (h_l/a_l)^2)`
counts actions spent in the environment. Search inside a simulator is free. That
is the entire thesis: move the 81 wasted actions out of the scored budget and
into offline compute.

If lp85 L3 took 19 actions instead of 100, its level score goes 9.61 -> **115**
(the cap binds at <=28 actions against a baseline of 31). lp85's E goes 4.06% ->
12.84%, and **RHAE 0.1886% -> 0.54%** from that one level. Doing the same to L2
and L4 takes it to roughly 0.66%. Our standing Kaggle best is 0.17.

## What changes

The closed line asked for `policy(frame) -> action`. That is still
LLM-as-policy, just compiled — a reactive observation-to-action mapping. Ask
instead for the pair the winners build:

```python
def simulate(state: State, action: Action) -> State: ...
def is_goal(state: State) -> bool: ...
```

Then **we** plan: BFS over `simulate` from the live state until `is_goal`, and
emit the action sequence. The model never picks a move and never sees a reward.

This also dissolves the failure mode that closed the old line. Its validator had
to judge whether a policy's *choice* was good — subjective, needing a
"productivity" heuristic that went through six designs and still reported
`0 judged` on all eight games. A world model's validator is exact and dense:

> replay every recorded transition `(s, a, s')` and check `simulate(s, a) == s'`

No heuristic, no bar to tune, one number per candidate, and it runs on
transitions we already have.

## Milestones, in order, each falsifiable

**M0 — hand-written simulator, no LLM. This was the kill switch. RUN, and it
passes on the thesis while moving the constraint.**

`copy.deepcopy(env._game)` is a perfect world model and forks in 4.7 ms, so M0
did not need a hand-written simulator at all — it measures the *ceiling* of the
approach directly. Results against the explorer:

| level | explorer | planned | offline expansions | outcome |
| --- | --- | --- | --- | --- |
| L1 | 9 | **5** | 9 | already at the cap either way |
| L2 | 364 | **8** | 48, in 3s | score 1.09 -> **115** (capped) |
| L3 | 100 | no plan | 80k, plateaued | greedy stuck in a local minimum |
| L4 | 428 | not reached | — | — |

On L2 alone: lp85's E goes 4.06% -> 10.40% and **RHAE 0.1886% -> 0.4422%**
(+0.2536pp), more than doubling, from a search that spent **zero** real actions
and three seconds of CPU.

Three results, in order of how much they change the plan:

1. **The thesis holds.** Planning through a world model beat blind search by
   **45x** on lp85 L2. That is not a tuning delta; it is a different regime.
2. **The world model is not the constraint — the heuristic is.** Uninformed BFS
   over the *same perfect simulator* failed completely: 40k expansions and 26k
   distinct states still only reached depth 6, because branching is ~6 with
   almost no state merging. Adding a goal-distance heuristic took L2 from
   unsolvable to 48 expansions. This is the finding that redirects M1.
3. **Heuristic shape matters more than heuristic presence.** Counting misplaced
   blocks (the win condition as written, `khartslnwa`) takes ~3 values on a
   two-block level and left greedy search flailing at depth 19. Summing each
   block's distance to its nearest goal — same zero set, real gradient — solved
   L2 in 3 seconds. L1 also dropped from 21 expansions to 9.
4. **Greedy is not enough for every level.** L3 drives h from 66 to 26 and then
   plateaus across 80k expansions: a local minimum where blocks must temporarily
   move *away* from their goals. Needs A\* with backtracking, or a better
   heuristic, not more compute.

**What M0 does not show.** It used the game's internals — `deepcopy` of the real
game as the simulator, and the real sprite tags (`bghvgbtwcb`, `goal`) for the
heuristic. Neither is available in the kernel. M0 is an upper bound by
construction; that was its purpose. What it licenses is M1, not a submission.

It also settles the feasibility question it was meant to: lp85's sprite scale is
**3** (`crxpafuiwp = 3`), so the 64x64 frame does not downsample to a clean
logical grid — at 4x4 blocks only 55% are single-coloured. State extraction has
to be object-level, not pixel-level. Usefully, the *action* space turned out
trivial to recover from observation alone: probing a stride-3 lattice finds 2-6
distinct button effects per level, and they are static within a level.

**M1 — the model writes the simulator *and the heuristic*.** Revised by M0: ask
for three functions, not one.

```python
def simulate(state, action) -> state   # validated: exact next-state prediction
def is_goal(state) -> bool             # validated: fires exactly on recorded wins
def distance(state) -> float           # validated: does greedy on it reach a goal?
```

`distance` is now the interesting one — M0 showed it is the constraint, and that
the obvious formulation (count the unsatisfied conditions) is the bad one while a
graded version of the *same* predicate is transformative. That is exactly the
kind of restatement a model is good at and a fitted constant is not.

Hold out a fraction of the recorded transitions and score `simulate` on exact
next-state prediction. The open question M1 answers is whether partial accuracy
is worth anything: a simulator that is 95% right may produce plans that are 100%
wrong, in which case the usable threshold is near 1.0 and that is a finding.

State extraction is a prerequisite and is not free — M0 sidestepped it with
`deepcopy`. **That prerequisite is now tested and passes**
(`sia-oss/bench/m1_markov.py`): keying state as each component's
`(colour, top-left, pixel count)` gave **119 distinct `(state, action)` pairs
over 1500 transitions with zero ambiguity**, identical to keying on the whole
frame. The hidden `StepCounter` does not leak into the dynamics, so a simulator
over object-level state is well-posed rather than merely plausible.

Two honest limits on that result: it covers lp85 L1 only (2 buttons, a small
reachable space), and it resets on GAME_OVER, so it never probes the
budget-exhaustion boundary — exactly where the hidden counter *would* bite.
Re-run it per level before relying on it.

**What is left in M1 is the semantic step, and it is the LLM's actual job.** The
heuristic M0 proved decisive was "distance from each block to its nearest goal",
computed from privileged sprite tags (`bghvgbtwcb`, `goal`). From pixels alone
lp85 L2 is ~40 same-sized 2x2 tiles in four colours, and deciding which are
cargo and which are targets is an inference about the game, not a measurement
of it. That is the first point in this whole line where no amount of careful
engineering substitutes for a model — which is a much better place to spend an
LLM than the reactive policy of the closed line.

**M2 — plan through it live, then gate.** Only here do real actions get spent.
Promotion needs the usual discipline: `gate.py` (RHAE up, no game regresses)
**and** `sweep.py` on any constant introduced. Per-game non-regression is
non-negotiable — see the three headline "wins" that evaporated under sweeps.

## Scope boundaries

- **lp85 only** through M2. It is 86% of our current score, it is click-only
  (`available_actions=[6]`), its per-level budgets are small (13/60/80/150), and
  its dynamics are a deterministic sprite permutation per button. Generalising
  to 25 games is a later question and explicitly not this scope.
- **The explorer stays the fallback** for every step a plan does not cover, so a
  failed generation costs an LLM call and nothing on the scoreboard. This
  invariant is inherited from the closed line and was never the problem.
- **Backends must disable thinking** (`chat_template_kwargs`) — a reasoning
  model burns the whole token budget and returns empty content with
  `finish_reason="length"`. Measured: 6229 tokens/229s vs 173 tokens/6.8s for
  the same usable output.
- **Stop rule.** If M0 fails, close the line and say so in
  `sia-oss/tasks/arc-agi3/reference/AGENTS.md`. If M1's accuracy plateaus below
  whatever M0 shows is the usable threshold, same. This scope exists to make
  those two exits cheap.

## Known risks

1. **lp85's button semantics are a lookup.** `step()` resolves a button tag to
   `chmfaflqhy(level_name, button_id, R/L, ...)`, which moves a *set* of sprites
   — effectively a per-button permutation table. Learnable from transitions, but
   only with coverage: L3 gives ~97 transitions against an unknown number of
   buttons. Coverage may be the binding constraint rather than the model.
2. **Exactness.** Planning through an approximate simulator can be worse than
   not planning. M1 measures this rather than assuming it.
3. **This is attempt 8.** Both variables anyone reaches for next — the model and
   the validator — were already varied in attempts 1-7 and neither was the
   constraint. The claim here is that the *output type* was, which M0 tests
   without a model at all.
