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
Promotion needs the usual discipline: `ab.py` (mean RHAE up by more than
2 sd across seeds, no game loses seeds)
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

## The M1 backend

`sia-oss/bench/launch_vllm.sh`, copied to the pod and run there; then
`ssh -N -L 8011:127.0.0.1:8011 pi-a100-80gb` locally and point
`HTTPChatBackend(base_url="http://127.0.0.1:8011/v1", model="qwen-27b")` at it.
Verified end to end: the repo's own backend seam returns clean code.

Three launch failures preceded a working one, none of them guessable:

1. **Do not use the pod's base `python3`.** `~/.local` is shared with other
   tenants and pins `transformers` 4.40.1, which vllm 0.26.0 cannot import
   (`ALLOWED_LAYER_TYPES`). Upgrading it would risk someone else's job. Use
   `~/vllm_venv` (transformers 5.16.1 + vllm 0.26.0, a matching pair).
2. **`ninja` must be on `PATH`**, not merely installed. vLLM shells out to it to
   JIT the MoE kernels, and invoking the interpreter by absolute path does not
   put the venv's `bin` on `PATH`.
3. **Model choice is decided by memory, not preference.** `Qwen3.6-35B-A3B` is
   cached but needs ~70GB of weights and leaves no room for KV cache at a
   utilisation this shared card can spare. The FP8 27B that would best match the
   Kaggle kernel is a **12K stub** in the cache — never actually downloaded. So
   `Qwen3.8-27B` dense at `--gpu-memory-utilization 0.88`, which lands at 70.7GB
   and leaves the other tenant alone.

Shared-pod discipline: port 8011 rather than 8000, utilisation capped, and the
other tenant's process (566MB, ~20% util) untouched throughout.

## M1 result — the heuristic transfers, the simulator does not

Run on lp85 L1 with Qwen3.8-27B (`sia-oss/bench/m1_codegen.py`, two requests:
`is_goal`+`distance`, then `simulate`).

| function | result |
| --- | --- |
| `is_goal` | **9/9 wins, 8/340 false positives** |
| `distance` | **8 distinct values**, and it solves L1 in **5 real actions** |
| `simulate` | **0/340 exact** |

**The headline: the generated `distance`, plugged into M0's search in place of the
privileged sprite-tag heuristic, clears lp85 L1 in the same 5 real actions**
(`sia-oss/bench/m1_endtoend.py`). It takes 133 offline expansions against M0's 9,
which costs nothing — the metric charges real actions only. On the measure that
matters, a model-written heuristic matched a hand-written one with access to the
game's internals.

**Two of the three failures along the way were mine, not the model's**, and both
are worth stating because each looked like a model limitation:

1. **The prompt never showed a winning board.** It said "9 transitions reached
   the solved board" and included only the start board in full. The model wrote
   `# The solved board is the initial board` and hardcoded the start state as the
   target — exactly inverted. Showing one win took `is_goal` from 0/9 to 9/9.
2. **The state representation was non-stationary.** `objects()` stripped the
   *per-frame* modal colour. Colour 4 is lp85's floor while unsolved, but a
   solved board is mostly one 41x41 colour-4 region, which shifts the mode and
   makes colour 4 appear as an object. The model inferred "41x41 colour-4 object
   means solved" — correct from what it could see — then wrote a `distance` over
   colour-4 objects that are absent from every unsolved state, returning infinity
   everywhere. Pinning the background took `distance` from 1 distinct value to 8.

**The real limitation is circularity, not capability.** Both functions hardcode
the solved board they were shown: `is_goal` checks a 59-object subset, `distance`
measures Manhattan distance to per-colour target positions read off that same
board. The *shape* is exactly M0's winning heuristic and it demonstrably works —
but computing distance-to-goal from a known goal is not yet search, because the
goal configuration is what an unseen level withholds. The open problem is now
narrow and stateable: **infer the target configuration from an unsolved board.**
lp85 shows its goal markers on the board, so this is plausible; it was simply
never asked for, because the prompt handed over a win instead.

**`simulate` at 0% did not block any of this**, because expansion ran against a
`deepcopy` of the real game. Whether that is available in the Kaggle kernel is
now the highest-value open question: if the kernel's arcade is local, planning
costs zero real actions there too and a learned simulator is unnecessary; if it
is remote, `simulate` becomes load-bearing and 0% is fatal. **Check before
building anything further on this.**

## Does `deepcopy` work in the kernel? No — but a shadow might

**Answered, and it was already in the codebase.** The `ForwardModel` docstring in
`arc_agi3_kaggle.py` says it outright, and exists because of it:

> there is no way to get one from the environment where it counts. The scored run
> sets OPERATION_MODE=online against a gateway sidecar with no local
> environments, so there is no game object to fork the way the local sk48 solver
> forked one with deepcopy.

Confirmed in `arc_agi3_build_notebook.py`. Two paths, and only one is scored:

| path | mode | environments |
| --- | --- | --- |
| **Scored run** (Cell 5) | `OPERATION_MODE=online` -> `http://gateway:8001/` | `ENVIRONMENTS_DIR=` **empty** |
| Mock / preflight | `OperationMode.OFFLINE` | `{COMP}/environment_files` |

So M0's ceiling is not reachable by the route M0 used, and **`simulate` scoring
0% in M1 is fatal rather than incidental**. This should have been established
before M0 was scoped; a previous generation hit the same wall and wrote it down.

**The shadow alternative.** The bundled `environment_files` *are* present in the
kernel — the mock reads them. So an agent could instantiate its own copy of the
game, replay its own actions into it to hold it in sync, and fork *that*. Two
prerequisites, both now measured:

- **Seed-independence**: 8/8 scoring games produce identical opening frames
  across seeds 0/1/3/7, so a shadow starts where the gateway starts.
- **Replay determinism** (`sia-oss/bench/shadow_sync.py`): 8/8 games replay a
  300-action trajectory byte-identical into a fresh instance, lp85 including a
  level transition.

**Still unverified, and all three are load-bearing:**

1. The gateway must serve the same build as the version-pinned bundle
   (`environment_files/<game>/<hash>`). Untestable from outside the kernel.
2. The scored games must be among the bundled 25. Local-25 RHAE tracking the
   public score is evidence, not proof.
3. Determinism was measured under random play, which clears almost nothing — only
   one level transition is actually covered.

**And a judgement call that is not the model's to make.** A shadow built from the
shipped game source *reads* the dynamics rather than inferring them, which is a
different thing from the world model this document set out to learn. The files
ship with the competition and the top of the leaderboard is hard to explain
without something in this family, but whether to use it is the submitter's call.
Recorded here as measured and available, not adopted.

## How badly does the generated `simulate` fail?

"0/340 exact" understates it. Exact-match on a ~38-object board would score a
one-object slip the same as garbage, so the gap was measured directly:

| measure | generated `simulate` | echo-the-input baseline |
| --- | --- | --- |
| exact next state | 0/340 | 0/340 |
| objects correct | 2996/13123 = **22.8%** | **46.0%** |
| **objects the action moved** | 8/7091 = **0.1%** | 0% by construction |

It is **worse than doing nothing**: returning the input unchanged would get 46%
of objects right, because most of the board is static scenery. The generated code
scrambles that scenery *and* gets 0.1% of the ~21 objects-per-action that
actually move. This is not a simulator that needs refining; it has not learned
the dynamics at all.

Together with `learnability.py` — no action in any of five games has a fixed
effect, and new effects were still appearing after 400 sightings — **both routes
to a forward model are currently failing**: a learned table does not converge,
and a model-written rule is below a do-nothing baseline. Anything built on
planning in the scored kernel is blocked behind this.

## The local simulator as an offline oracle (2026-09-18)

Reframed: the forked game is a **research instrument we read locally**, not a
shadow to ship. The live agent will not have the source; the point is to learn
from the oracle what good play looks like and fold that back into the agent.

**Uninformed offline planning reaches a level in 11 of 25 games** — no heuristic
at all, just "did the level counter go up" (`sia-oss/bench/m0_suite.py`). Five of
those the explorer has never cleared:

| game | oracle levels | oracle actions | explorer | now E% | oracle E% |
| --- | --- | --- | --- | --- | --- |
| m0r0 | 2 | 38 | 1293 for 1 | 0.0026 | 16.43 |
| tu93 | 3 | 47 | 1463 for 3 | 0.0455 | 14.86 |
| cd82 | 1 | **5** | never cleared | 0.0000 | 5.48 |
| ft09 | 1 | **4** | never cleared | 0.0000 | 5.48 |
| sp80 | 1 | **4** | 192 | 0.1965 | 5.48 |
| vc33 | 1 | **3** | never cleared | 0.0000 | 4.11 |
| ls20 | 1 | 13 | 68 | 0.3738 | 4.11 |
| sk48 | 1 | 14 | never cleared | 0.0000 | 3.19 |
| s5i5 | 1 | 13 | 1708 | 0.0004 | 3.19 |
| ar25 | 1 | 15 | 570 | 0.0098 | 3.19 |
| lp85 | 1 | 5 | — | 4.0591 | 3.19 |

Projected RHAE **2.75% against 0.1886%**. Read it as a *ceiling on what the
explorer leaves behind*, not a target: it needs a forked game, and it is not
uniformly better (lp85 scores worse because the oracle took 1 level where the
explorer takes 4, and it failed outright on sc25).

**Risks 2 and 3, both now closed.** The scored kernel plays the same 25 games —
measured in commit mode, and `2/25` was the recorded ceiling; the "~110
concurrent" in the notebook is thread slots, not distinct games. And a fresh
instance replaying an identical action list stays **byte-identical through level
transitions**: tu93 over 3 levels / 47 actions, m0r0 over 2 / 38, plus sp80,
ls20, vc33.

### What the oracle says about the architecture

Winning plans are **short, narrow, and repetitive**: 3-18 actions, using 1-4 of
the 2-8 available actions, and 60-80% repeats. ar25's win is essentially
`act2 x10` then `act3 x5`.

Mean consecutive-run length, oracle plan against what the explorer plays:

| game | oracle | explorer |
| --- | --- | --- |
| ar25 | **7.50** | 1.42 |
| s5i5 | **6.50** | 1.09 |
| lp85 | **5.00** | 1.37 |
| ls20 | **3.25** | 1.67 |
| sp80 | 2.00 | 1.68 |
| m0r0 | 1.88 | 1.79 |
| sk48 | 1.75 | 1.38 |
| tu93 | 1.29 | 1.27 |

The explorer sits at 1.1-1.8 everywhere: it almost never repeats an action. That
is structural, not incidental — `_choose` takes an *untested* primitive at the
current state, and every repetition changes the state, which makes the other
primitives untested again. A novelty-ordered frontier search cannot emit
`act2 x10` except by accident.

**Honest limits on that claim.** It is a strong gap on 4 of 8 games and absent on
the other 4 (tu93 and m0r0 nearly match). And the comparison is not like for
like: the oracle plan is an optimal exploitation path, the explorer's trace is
mostly exploration. What it establishes is a *hypothesis worth gating* — a
run/momentum prior that keeps pressing what just worked — not a proven win.
Two such changes were already rejected by the gate this session.
## Why five games never clear

The oracle clears cd82, ft09, sk48, vc33 and a second m0r0 level that the
explorer never reaches. Comparing its winning plans against the explorer's
candidate set splits them into two causes:

**ft09 and vc33 — the winning button is not in the action set.** No click the
explorer proposes reproduces the oracle's winning effect, and raising `k` from
12 to 96 or dropping the field filter does not surface it. vc33's button is at
col 60, outside the field box (1,0)-(63,51), so `in_field` excludes it. ft09's is
in-field, 36px, non-background — so the centroid test (`arr[cr,cc] != v`, which
drops hollow shapes) rejects it. Budget and search were never going to help.

**cd82, sk48 and m0r0 — candidates are fine.** The oracle's first action
(`act3`/`act1`/`act1`) is proposed and is played; they fail on the continuation.

A caveat on method: the first version of this test asked whether the oracle's
exact *cell* was proposed. That is too strict — the explorer clicks component
centroids, so a different cell on the same button is equivalent. The numbers
above come from an effect-equivalence test, which happens to agree.

## The oracle's labels, and what they say about the agent (2026-09-21)

`m0_suite.py` found winning plans and logged only their length. `oracle_labels.py`
keeps them: for each of the 11 games the oracle solves, the board as it stood
before every winning action — **172 labelled decisions**, committed under
`sia-oss/bench/oracle/`. `oracle_recall.py` replays those boards through the
agent's own `proposals` and reports where the winning move lands.

Everything here is inferred from play — `available_actions`, the rendered grid,
the level counter — so nothing learned from it is privileged the way `m0_plan`'s
sprite-tag heuristic was.

**Coverage is not the bottleneck. Ranking is.**

| cutoff | recall |
| --- | --- |
| proposable at all | 167/172 (**97%**) |
| @12 | 167/172 (97%) |
| @4 | 150/172 (87%) |
| @1 | 31/172 (**18%**) |

The winning move is almost always on the list; the agent takes it first 18% of
the time. That kills the candidate-generator program these labels were gathered
to support, and redirects the question to selection.

**The plans are simple actions, not clicks** — 147 of 172. Only ft09, vc33, s5i5
and lp85 win by clicking. Read with care: `actions_for` lists simple ids before
click cells and BFS returns the first shortest path it finds, so ties break
toward simple actions. What survives the caveat is that a short simple-action
solution *exists* — cd82 in 5, sp80 in 4 — for games the explorer never clears
in 6000.

Two corrections to "Why five games never clear" above:

- **ft09's level-clearing click is proposable after all.** Of its four
  decisions, steps 0 and 3 are covered and step 3 is the one that takes the
  level; the two uncovered are intermediate setup clicks. The game still cannot
  be completed, but not for the reason recorded there.
- **vc33 is confirmed** at 0/3 — the col-60 button outside the field box
  (1,0)-(63,51).

The effect-equivalence caveat recorded above was re-derived here the hard way:
exact-cell recall scores lp85 0/5 and s5i5 0/13, against 5/5 and 13/13 by effect.
`oracle_recall.py` therefore defaults to `--effect`.

## Click effect is predictable, but constant in most games (2026-09-21)

`effect_purity.py` fork-probes a stride-3 lattice at six on-policy frames per
game across all 25, and buckets each cell by `(colour, component-size)`.
Excluding the background colour and singleton buckets, **1096/1112 buckets
(99%) are unanimous** on whether clicking does anything. Mid-episode does not
degrade it; su15 (80%) and ft09 (89%) are the only games below unanimous.

So effect is learnable from frame features. It is also **uninformative in 14 of
25 games**, because it is constant there:

| regime | games |
| --- | --- |
| no click ever works | ar25, ls20, re86, sk48, tr87, tu93, wa30 |
| every cell works | bp35, lf52, r11l, s5i5, sp80, tn36, vc33 |
| informative | cd82, cn04, dc22, ft09, g50t, ka59, lp85, m0r0, sb26, sc25, su15 |

vc33 sits in the all-effective set, so effect-ranking could never have surfaced
its button — the earlier diagnosis of it as a ranking problem was wrong on
mechanism as well as on cause.

A lever that looked promising and is not: the explorer spends clicks in only 2
of the 7 click-inert games (sk48 280, ar25 147; the other five emit none). Cutting
them moves ar25's E% from 0.0098 to ~0.017, about **0.0003pp** — two orders of
magnitude under the 0.030pp noise floor.

## A learned static ranker does not transfer (2026-09-21)

`oracle_rank.py` fits a ranker on the labelled decisions and scores it
leave-one-game-out, because the scored kernel plays games the fit never saw.
In-sample numbers are not reported: they answer the wrong question.

**Leave-one-game-out recall@1: 31/167 (19%) -> 36/167 (22%).** All five gained
decisions are lp85; every other game is identical to baseline. Split by action
kind, the result is unambiguous:

| decisions | baseline@1 | model@1 |
| --- | --- | --- |
| clicks (20) | 0% | 25% |
| simple (147) | 21% | 21% |

The ranker transfers a little on clicks and not at all on simple actions, which
are 88% of the decisions. **This falsifies the static-prior programme**, and the
reason is structural rather than a feature-engineering shortfall: the map from
an action id to its meaning is game-specific, so no frame-derived feature can
tell which id is right without having watched that id act in *this* game.

The agent's own branch statistics agree, measured by teacher-forcing it along
each oracle plan:

| branch | fired | agrees with the oracle | what it uses |
| --- | --- | --- | --- |
| `probe` | 28 | **36%** | effects observed in this episode |
| `affordance` | 27 | 30% | learned avatar and move lattice |
| `_choose` | 117 | **15%** | static proposal order |

`_choose` makes 68% of the decisions on the worst signal available.

**But the branch gap does not survive a control, and `probe` is not a lever.**
`_probe_moves` is a bootstrap: it takes each of the four directional ids once to
build the move lattice, guarded by a `_probed` set that never resets, so 28
firings is 4 moves x 7 games with moves. It is already saturated — **145 of the
147 winning simple actions use ids 1-4**, which it covers; only 3 use action 5.

The 36% vs 15% comparison was confounded. `_choose` only fires early in the
click-only games (ft09, lp85, s5i5, vc33), where click recall@1 is 0% anyway, so
the branches never competed at the same positions or on the same games.
Restricted to games where both occur:

| branch | n | agrees |
| --- | --- | --- |
| `probe` | 28 | 36% |
| `affordance` | 27 | 30% |
| `_choose` | 92 | 20% |
| — at idx 4-9 | 26 | **31%** |
| — at idx 10+ | 66 | 15% |

`_choose` at comparable early positions scores 31% against probe's 36% — inside
noise at n=26. What varies is decision depth, not branch, and even that rests on
two games dominating the deep bucket. **Do not retry "make probe fire more".**

## Every scoring game is efficiency-limited, not level-limited (2026-09-21)

`evaluate.py` scores `env = min(cap, weighted)` where `cap = k(k+1)/2 /
total_weight` for `k` levels cleared, and `weighted` sums `l * min(1.15,
(baseline_l / our_actions_l)^2)`. Only actions inside a *completed* level are
scored, so the budget burned after the last clear costs nothing — and the cap
binds only once play matches the human baseline.

On the shipping budget not one scoring game is anywhere near its cap:

| game | levels | env% | cap% | capturing |
| --- | --- | --- | --- | --- |
| lp85 | 4/8 | 4.061% | 27.778% | 15% |
| tu93 | 3/9 | 0.046% | 13.333% | 0.3% |
| sp80 | 1/6 | 0.196% | 4.762% | 4% |
| ls20 | 1/7 | 0.374% | 3.571% | 10% |
| ar25 | 1/8 | 0.009% | 2.778% | 0.3% |
| m0r0 | 1/6 | 0.001% | 4.762% | 0.02% |
| s5i5 | 1/8 | 0.000% | 2.778% | ~0% |

**Perfect efficiency at the levels already cleared is worth +2.203pp** — RHAE
0.1875% to ~2.39%, without clearing a single new level, and against +0.73pp for
winning all four games that never clear. The headroom is in playing what we
already win faster.

Per level, against the human baseline the metric actually scores:

| game | lvl | ours | baseline | ratio |
| --- | --- | --- | --- | --- |
| **lp85** | **1** | **10** | 17 | **0.6x — beats baseline, hits the 1.15 cap** |
| ls20 | 1 | 68 | 22 | 3.1x |
| lp85 | 3 | 100 | 31 | 3.2x |
| sp80 | 1 | 192 | 39 | 4.9x |
| tu93 | 1 | 383 | 19 | 20.2x |
| lp85 | 4 | 777 | 16 | 48.6x |
| m0r0 | 1 | 2087 | 30 | 69.6x |
| s5i5 | 1 | 1707 | 20 | 85.3x |

lp85 L1 is the important row: the agent already plays *above* the human baseline
when it finds the path quickly, so this is not a capability ceiling.

**The waste has two mechanisms, and they need different fixes**
(`action_budget.py`, per-level revisit rate inside scored levels):

| game | level | actions | revisited | prims | mechanism |
| --- | --- | --- | --- | --- | --- |
| tu93 | 1-3 | 383/194/886 | 81-88% | 4 | cycling |
| sp80 | 1 | 192 | 83% | 30 | cycling |
| lp85 | 2-3 | 364/777 | 3-7% | 12-21 | broad undirected search |
| ls20 | 1 | 68 | 3% | 4 | broad undirected search |

tu93 and sp80 return to boards they have already seen for most of their actions.
lp85 and ls20 do not: lp85's 777 actions on level 3 reached ~730 *distinct*
boards.

**But a high revisit rate is not cycling, and there is no cycle breaker to
build.** Splitting `_choose` into its four sub-paths shows `rotate` — the stall
pathology — at **0% in every one of these games**:

| game | untested | plan | frontier | rotate |
| --- | --- | --- | --- | --- |
| sp80 | 98% | 0% | 2% | **0%** |
| lp85 | 99% | 0% | 1% | **0%** |
| ls20 | 100% | 0% | 0% | **0%** |
| tu93 | 54% | 34% | 13% | **0%** |

A board holding several untested primitives is revisited once per primitive,
which is correct breadth-first play — sp80 revisits 83% of boards *and* takes a
never-tried action 98% of the time. Only tu93 carries real overhead, 47% of its
actions routing back to frontiers, and removing all of it is worth **+0.0047pp**,
well under the 0.0625pp noise floor. **Do not build a cycle breaker.**

What the speed ladder is worth, applied to every game at once:

| speedup | RHAE | delta |
| --- | --- | --- |
| x1.9 | 0.3432% | +0.156pp |
| x4 | 0.8571% | +0.670pp |
| x10 | 1.3280% | +1.141pp |
| perfect | 2.3905% | +2.203pp |

So a *uniform* 2x is measurable at 2.5x the noise floor, but a single game's
routing fix is not. With `rotate` at 0% and `untested` dominating, the only
general saving left is trying candidates in a better order — which is the
recall@1 lever again, at 18%. Efficiency and selection are the same problem.

## In-episode colour demotion: measured, swept, rejected (2026-09-23)

`_inert` already demotes a primitive that leaves the board untouched, but it
keys on the *cell*, so every dead cell is relearned separately. Colour carries
across the board, and the evidence said it should: pooled over six on-policy
frames, 43/62 colours are all-live or all-dead, and in lp85 **9 of 11 colours
never do anything**. lp85 is the biggest scorer (4.061% against a 27.778% cap),
so a working filter there was worth ~+0.39pp at x4.

It does not work at any threshold.

| DEAD_COLOUR_TRIES | RHAE | levels |
| --- | --- | --- |
| baseline | **0.1868%** | **8** |
| 1 | 0.1403% | 6 |
| 2 | 0.1400% | 6 |
| 3 | 0.1397% | 6 |
| 5 | 0.1393% | 6 |
| 8 | 0.1901% | 8 |
| 12 | 0.1898% | 8 |

**When the mechanism fires it costs two levels; when it does not fire it is
baseline.** At 8 and 12 the threshold is high enough that demotion rarely
triggers, and the +0.003pp there is a fifth of the noise floor. There is no
value at which it helps.

The failure is an exploration trap, not a bad threshold. A colour that reaches
its dead-click threshold *before* its first effective click is sorted last,
which makes that effective click less likely, which keeps it demoted.
`_live_colours` offers recovery only after a success the demotion prevents. The
information is real; acting on it greedily is self-confirming.

**And the prize was never there to win.** Measured before building the
non-greedy variant, purely observationally: replay the unchanged agent, note
from consecutive frames whether each action changed the board, and count the
clicks a *perfect* in-episode colour filter would have skipped.

| game | actions | clicks | dead | avoidable | % of actions |
| --- | --- | --- | --- | --- | --- |
| lp85 | 2501 | 2500 | 138 | 125 | 5.0% |
| ar25 | 2501 | 738 | 715 | 201 | 8.0% |
| sp80, ls20, m0r0, s5i5 | 2501 | — | — | **0** | 0.0% |

A perfect filter is worth **+0.0038pp**, sixteen times under the noise floor. No
implementation of this idea can pay, greedy or otherwise.

The reason is that `click_targets` already solves it. Only **1.4%** of lp85's
board cells do anything, but **94.5%** of the clicks the agent actually issues
do (2500 clicks, 138 dead), because salience ranking proposes component
centroids rather than arbitrary cells. The 98.6%-dead board was never the
agent's problem — it never clicks most of the board. This is the
"coverage is not the bottleneck" result from a different direction.

Two instrument bugs fell out of running this, both in tooling written the same
day. `ab.py` reported "no game changed how often it scores" while lp85 went from
4 levels to 1 — the port from `gate.py` kept per-game frequency and dropped
per-game depth. And `sweep.py` counted any delta above 1e-9 as a gain, so it
read the two inert points as "2/6 values gain"; judged against the 0.030pp noise
floor the same sweep is NONE. Both are fixed, and both were found by running the
instrument on a change already known to be bad — worth repeating deliberately.

## Four changes measured, four rejected

Every one came from a correct measurement, and the gate plus sweep refused all of
them. Recorded so they are not re-attempted.

| change | gate | why it failed |
| --- | --- | --- |
| `_inert` compared on the chrome-masked view | 0.1879 -> **0.1558** | lp85 -1 level, and it did nothing on sp80: the mask targets animation (>50% churn), while a step-budget HUD ticks on ~5-10% of steps |
| reject an induced avatar outside the field box | 0.1879 -> **0.1852** | **sp80 itself got worse** (E 0.196 -> 0.130). Affordance chasing chrome was not pure waste |
| repeat a productive primitive (`RUN_LIMIT`) | 0.1879 -> **0.1111** | six games regressed. "Productive" is not "progressing" |
| clicks scan the whole grid | **PASS** 0.1879 -> 0.1896 | sweep **SPIKE**: only 2/5 values of `k` beat baseline |

The last is the instructive one. vc33 clears at k=9/12/16/24 — the *mechanism* is
stable — but at k=16/24 the extra candidates displace in-field ones and sc25
stops scoring entirely (0.1229). k=12 is the lucky spot where vc33, sc25 and
ar25 all survive. Appending the out-of-field candidates as a demoted tail instead
fixes the displacement and loses the reach: nothing regresses, and vc33 is never
tried (0.1879, no change).

**`CLICK_TARGETS_K = 12` is a hard proposal budget, and reach trades against
focus inside it at roughly one game for one game.** Do not retry this as a
candidate-list tweak; it needs candidates scored by measured effect rather than
salience rank, which is the Phase-2 filtering `click_targets` already promises.

Two oracle-derived priors were also falsified: a run/momentum prior copied from
the plans' 60-80% repeat rate (above), and — from the earlier M1 work — the idea
that a table of action effects could be learned online at all. Across
lp85/ls20/sp80/tu93/sc25 **no action has a single fixed effect**; each does 24-75
distinct things and new ones were still appearing after 400 sightings.
