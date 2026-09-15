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

**M0 — hand-written simulator, no LLM. This is the kill switch.**
Write `simulate`/`is_goal` for lp85 by hand from the recorded transitions, run
BFS, and measure real actions to clear L3. If a *human-written*, known-correct
simulator plus offline planning cannot beat 100 actions, the approach is dead
and no model was needed to prove it. Cost: hours, zero GPU, zero LLM.
**Do not proceed to M1 until M0 clears L3 in under ~40 actions.**

M0 also settles the open feasibility question: lp85's sprite scale is **3**
(`crxpafuiwp = 3`), so the 64x64 frame does not downsample to a clean logical
grid — at 4x4 blocks only 55% are single-coloured. State extraction has to be
object-level (components and positions, which `click_targets` already computes),
not pixel-level. If object-level state turns out not to be Markov enough to
simulate, that surfaces in M0 too.

**M1 — the model writes the simulator.** Hold out a fraction of the recorded
transitions; score candidates on exact next-state prediction. Report accuracy,
not a pass/fail. The open question M1 answers is whether partial accuracy is
worth anything: a simulator that is 95% right may produce plans that are 100%
wrong, in which case the usable threshold is near 1.0 and that is a finding.

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
