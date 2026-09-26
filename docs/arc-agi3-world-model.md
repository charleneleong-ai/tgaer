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

## State fragmentation: the shipped key is already the best one (2026-09-24)

`frame_signature`'s own TODO warns it keys on every in-field pixel — "live ls20:
741 signatures for 30 avatar cells, blinding the StateGraph frontier to
revisits". Copying the 6.71% Preview agent's coarser object key looked like the
fix. It is not: measured against the agent's *real* signature (the settled
board, animation masked, inside its field box), coarsening is strictly worse.

| game | shipped pixel key | object key | shape+centroid |
| --- | --- | --- | --- |
| lp85 | **431 states, 291 rankable** | 733, 146 | 733, 146 |
| tu93 | **156 states, 147 rankable** | 721, 46 | 721, 46 |
| ls20 | 562, 153 | 562, 153 | 562, 153 |

"Rankable" is states where two or more distinct actions were tried — what a
value model can learn from and what the frontier needs to see a revisit.
Ambiguous transitions are **1 in ~2600** across all three, so every key is
effectively Markov; the shipped one simply merges more. The chrome mask is
doing the work the object key was supposed to do, and object tuples do not
benefit from it because every animated pixel still perturbs a component.

**Fragmentation is not the blocker.** An earlier reading of "728 states, 28
rankable" came from measuring the raw board rather than the settled one.

## The in-episode value model, and why its harness does not answer the question

`value_model.py` back-labels a cleared level by distance-to-win over the graph
the agent walked, trains on earlier levels and scores the held-out one — the
6.71% agent's method, offline.

**The harness has a confound that invalidates its baseline.** `_choose` plays
`untested[0]`, so proposal rank *is* visit order, while the episode moves toward
the win — later visits are closer to it, so rank is anti-correlated with
distance-to-win by construction. lp85 scores 0/215 for *both* arms against a 32%
chance floor because of it. Rank is unusable as feature or baseline here.

What survived: on tu93, adding a 21-feature board descriptor (colour histogram,
foreground centroid and spread) moved the model from 28% to **34%** against a
27% chance floor — 1.8 sd at n=102. Suggestive only. It does locate the missing
ingredient, though: `oracle_rank.features` describes the *action*, so for a
simple action it carries only the id and the model can learn "action 2 is
usually good" but never "in this state, action 2". That is precisely what a
learned grid representation supplies.

Four bugs were found and fixed inside this experiment before the confound
surfaced — a hardcoded rank, duplicated (state, action) rows from frontier
re-traversal, the raw board in place of the settled one, and a recomputed
proposal order that did not match the agent's.

### The redesigned harness, and what it says

Rebuilt to remove the confounds rather than patched around them:

- **rank is no longer a feature or a baseline** anywhere — it is visit order;
- **ties break at random**, not by position. Row order is the order the agent
  first tried each action, which is anti-correlated with distance, so
  `argmin`'s first-index bias scored a constant predictor at *zero* rather than
  at chance;
- **a chance floor is reported beside every arm**, so "beats baseline" is
  judged against something absolute;
- `oracle_rank.py` grew a **within-game** split — train on a game's earlier
  levels, score its last — on labels that come from BFS shortest paths and so
  were never contaminated by the agent's own wandering.

The result is a negative.

| harness | baseline | model | chance |
| --- | --- | --- | --- |
| oracle labels, cross-game | 19% | 22% | — |
| oracle labels, **within-game** | 11% | **11%** | **13%** |
| trajectory labels, tu93 | 28% | **35%** | 27% |
| trajectory labels, lp85 | 0% | **0%** | 32% |

Within-game training on unbiased labels sits *at* the chance floor on 53
decisions across three games — too little data to conclude much, but no support
either. The one positive, tu93 at 35% against a 27% floor, comes from the
trajectory harness where labels are plentiful.

lp85 anti-transfers outright: the mean *within-group* correlation between
predicted and true distance is **-0.433** across 215 states — the model ranks
the closest action last. Training levels carry distances of 1-66 (mean 37.9)
against the held-out level's 1-27 (mean 13.8). **Level-to-level transfer inside
one game can be negative**, which is a real obstacle to the retrain-per-level
design and not an artefact.

The board descriptor that lifted tu93 in the confounded harness changed nothing
cross-game: 19% -> 22% with and without it, the same five lp85 decisions.

### More oracle depth does not unlock it

Re-ran the oracle at 4x the expansion budget and 2x the per-level deadline
(`--levels 8 --budget 60000 --per-game 300`). **281 labels against 172, and every
added label is tu93** — 3 levels and 47 labels becoming 8 levels and 156. The
other ten games moved by exactly zero. Their level 2 is not reachable by
uninformed BFS at any budget worth spending, which is a different wall from the
one tu93 hit.

With tu93 training on 135 decisions across 7 levels and tested on its 8th — the
split this design actually calls for — the model scores **exactly the baseline**:

| within-game | decisions | baseline | model | chance |
| --- | --- | --- | --- | --- |
| tu93 (8 levels) | 21 | 33% | **33%** | 25% |
| m0r0 | 23 | 9% | 9% | 6% |
| ar25 | 11 | 0% | 0% | 6% |
| pooled | 55 | 16% | **16%** | 13% |

Model and baseline agree exactly in all three games, which is the finding rather
than a coincidence: `rank` remains a feature here, so with hand-crafted features
the best the ranker finds is to re-learn the ordering it already had. Data was
not the binding constraint.

Holding out only the last level wasted most of tu93, so the split became
**forward-chaining** — for each level k, train on levels < k and score k, which
is also the online situation. That scores 138 of tu93's decisions instead of 21,
and the answer is unambiguous:

| within-game, forward-chained | scored | baseline | model | chance |
| --- | --- | --- | --- | --- |
| tu93 (8 levels) | 138 | 26% | 27% | **25%** |
| m0r0 | 23 | 9% | 9% | 6% |
| ar25 | 11 | 0% | 0% | 6% |
| pooled | 172 | 22% | 23% | **21%** |

At n=138 the chance floor has an sd of 3.7 points, so 32% is the bar for a 2 sd
result. The baseline sits **+0.27 sd** above chance and the model **+0.54 sd**.
**Neither the shipped proposal order nor a learned ranker is distinguishable
from picking at random on our deepest game.** That is a statement about
`proposals` as much as about the value model: `_choose` consumes an ordering
that carries no information on tu93.

**What is left untested is the representation**, which is the one thing the
6.71% agent does differently — a ResNet over the grid rather than a dozen
summary numbers. Confirmed feasible in-kernel: torch 2.10 and torchvision ship
in the base image, CUDA 12.8 on an RTX PRO 6000 with 94GiB free, and 20 training
steps at batch 32 on 64x64 take 1.31s.

## Carrying `_inert` across a level boundary: rejected (2026-09-24)

`_on_new_level` clears `_inert` every time a level falls, which contradicts
`_inert`'s own rationale — it is deliberately state-key-free because "a cell
that does nothing here almost never does something two states later", and
`_det`'s move lattice already persists across levels. The measured cost looked
like relearning: lp85 clears level 1 in **10 actions against a 17-action human
baseline** (above baseline, hitting the 1.15 cap) and then spends 364, 100 and
777 on levels 2-4.

Keeping it fails.

| | mean | sd | seeds |
| --- | --- | --- | --- |
| baseline | 0.1561% | 0.0280 | 0.1868 x2, 0.1357 x3 |
| keep `_inert` | 0.1357% | **0.0000** | 0.1357 x5 |

delta **-0.0204pp** against a 2 sd bar of 0.0396 — inside noise, so no evidence
either way on the headline. Two things argue against pursuing it regardless.

The candidate's **sd collapses to zero**: carried-forward inert counts dominate
the salted tie-break and the agent goes deterministic again, which would destroy
the variance estimate the bench depends on. The colour-demotion change failed
with the same signature. And it reaches the better of the two score clusters on
**0 of 5 seeds** where the baseline reaches it on 2 of 5 — not significant at
n=5 (p ~= 0.08), but pointing the same way as the negative delta.

## Scope: LLM-as-programmer for a per-game heuristic (2026-09-24)

Eleven agent changes have been measured and reverted. Every one tuned the
existing policy; none changed what kind of agent it is. This is the one
remaining direction with a component already validated, and it is the only one
that **adapts per game at runtime** — which matters because the private set is
out-of-distribution by design, and that is exactly what defeated the static
ranker (21% -> 21% on the 88% of decisions that are simple ids).

**What is already known, and what is not.**

| | status |
| --- | --- |
| `is_goal` | 9/9 on lp85 once the prompt showed a winning board |
| `distance` | matched M0's *privileged* heuristic, 5 real actions on lp85 L1 |
| `simulate` | **failed** — 0.1% of moved objects against 46% for echoing the input |
| source-free | **yes** — `build_prompt` sends the start board, transitions as diffs, and the objects a solved board has that the start lacks. No game file is read |
| generality | **unknown** — `distance` was validated on lp85 L1 only, n=1 |

**The design follows from `simulate` failing.** With no forward model there is no
offline search, so `distance` cannot be used to plan. It can be used as a
*progress signal*: take an action for real, recompute distance on the new frame,
and keep or abandon the direction. That converts the explorer's blind novelty
search into hill-climbing, and a progress signal is precisely what it lacks —
`_choose` plays `untested[0]`, and on tu93 that ordering is indistinguishable
from random (26% against a 25% chance floor, n=138).

**Phases, each with a kill criterion, cheapest first.**

1. **Does `distance` generalise past lp85?** Generate one per game for the 11
   oracle games; walk each oracle trajectory and measure the fraction of steps
   where distance strictly decreases. 281 labelled steps already exist, the
   metric is noiseless, and no agent runs. *Kill if fewer than half the games
   are monotone on most steps.*
2. **Does greedy-on-distance beat the explorer, given a fork?** At each state
   fork every proposal and take the minimum. An upper bound, since the kernel
   has no forkable game. *Kill if it does not beat the explorer's actions-to-
   clear by 2x on games we already clear.*
3. **Does it survive without the fork?** Hill-climb with real actions: act,
   recompute, abandon on a worsening. *Kill if worse than the explorer.*
4. **Gate with `ab.py`**, 5 seeds, the usual 2 sd and per-game depth bars.
5. **Kernel integration.**

**Costs and risks, stated before starting.**

- The explorer build currently ships **no model**. Reinstating vLLM costs the
  10-15 minutes of install and weight load that dropping it saved, plus codegen
  for ~110 games. At 30s each that is ~1.2h of a 7.5h budget. There is slack —
  we spend 6000 actions per game where 16590 are affordable — but it is real.
- **The bootstrap problem is the main risk.** The prompt is far stronger when it
  can show a winning board, and we clear only 5 of 25 public games. On a game
  never won, it falls to the "infer the goal from structure" branch, which is
  unvalidated. This is the same wall the value model hit.
- Generated code can crash or hang; it needs a timeout and a fall-back to the
  current policy per game.
- The kernel runs vLLM 0.19 against 0.26 locally, a divergence that has cost
  builds before.

**Honest prior.** `distance` worked once, on one level of one game, with a
winning board in the prompt. Phase 1 is cheap precisely because that is thin
evidence.

## Ablation: what each mechanism is actually worth (2026-09-25)

Eleven *additions* have been measured and reverted; nothing had ever been
removed. `ablate.py` switches each mechanism off over five seeds and judges it
with `ab.py`'s verdict against the unchanged agent.

| mechanism off | RHAE | delta | per-game |
| --- | --- | --- | --- |
| `USE_PROBE` | 0.3520% | **+0.1959pp** | ls20 5/5 -> 0/5, **g50t 0/5 -> 5/5** |
| `USE_GOAL_INDUCTION` | 0.1843% | +0.0281pp | nothing regressed |
| `USE_NAV` | 0.1561% | **+0.0000pp** | **byte-identical** |
| `USE_INERT` | 0.1421% | -0.0140pp | ar25 lost |
| `USE_AFFORDANCE` | 0.1382% | -0.0180pp | ar25, ls20 lost |
| `USE_CHURN_MASK` | 0.1354% | -0.0207pp | ar25 lost |
| `USE_FRONTIER` | 0.1281% | -0.0280pp | ar25, tu93 lost |

**`nav` never fires, but it is not dead code — its precondition is broken.**
Switching it off is byte-identical on all five seeds, and a door is induced on
**0%** of steps across ten games. Deleting it fails five tests, which build
synthetic key/door boards where it does drive the step, so the suite encodes a
belief that it matters.

The reason it never runs is a located bug. `_nav_move` needs `_det.door`, set
only by `_observe_door`, which runs **only on a level-up** and requires a colour
that *vanished* between the two frames:

```
ls20 level-up: avatar=12  colours gone=[]  adjacent-to-avatar=[5, 9, 12]  overlap=[]
tu93 level-up: avatar=6   colours gone=[]  adjacent-to-avatar=[0, 5, 6]   overlap=[]
```

`gone` is empty at every level-up on every game, because a level-up replaces the
board and the palette carries over. On ls20 the door colour is **9** —
`LS20_DEFAULT.door` — and it is adjacent to the avatar at that exact moment. The
inducer is looking straight at the door and rejecting it on a test that cannot
hold where it is called. It is called once per level-up, with the avatar known,
and finds no candidate every time (1, 1, 1 and 2 calls on ls20, ar25, sp80,
tu93).

The mechanism is right in spirit — a door vanishes when you enter it *mid-level*
— and wired to the wrong event. **Do not delete `nav`; fix the inducer.**

**Removing `probe` is worth +0.1959pp and is a trade.** Every seed shifts by
exactly that amount with an unchanged sd, so it is systematic, not luck: probe
builds the move lattice, so without it `nav` and `affordance` can never fire and
the agent falls back to pure frontier exploration. **g50t unlocks in 5/5 seeds
and ls20 dies in 5/5.** `ab.py` rejects it on the no-regression rule — the rule
exists because exactly this kind of trade has failed to reproduce before — but
no previous trade was +0.1959pp, five times the 0.0396pp bar, and deterministic
on both sides.

**The four "earns its place" verdicts rest on per-game regressions, not on
deltas.** Every one of those deltas is *inside* the 0.0396pp noise bar. What
keeps them is that removing each costs a game outright. `_inert` in particular —
whose docstring has asked since #29 for a re-measurement after the chrome mask —
is worth -0.0140pp and one game. That is a thin case for a mechanism carrying
this much machinery.

## The local bench does not predict the score — tested directly (2026-09-25)

**Read this before proposing anything gated on local RHAE.**

The probe cap was the first change in this project to pass every local bar: a
`sweep.py` **STABLE** verdict across a plateau, +0.1959pp over five seeds at 3.5x
the noise bar, deterministic per game, and confirmed in the kernel at 9 levels
over 7 games against v75's 8 over 6. It was submitted as v76.

| version | agent | local RHAE | public |
| --- | --- | --- | --- |
| v75 | baseline | 0.1561% | 0.13 |
| v76 | `PROBE_LIMIT=1` | **0.3520%** | **0.13** |

**Local more than doubled. Public moved zero.**

This is not a null inside noise. If the metrics were coupled, a 2.25x local gain
should have taken public to roughly 0.29 — about **six times the 0.027 noise
bar** on the public metric. The predicted effect was large and entirely absent.

**Why.** 73% of the local gain was g50t unlocking and the rest was ar25's
efficiency, both selected on the public 25 — the set the ARC-AGI-3 paper calls
"intentionally out-of-distribution relative to the public set" precisely to
resist this. The change helped exactly the games it was chosen on.

**The plateau argument did not save it.** `PROBE_LIMIT` 0, 1, 2 and 3 all beat 4,
which is the STABLE shape this repo treats as evidence of a mechanism rather than
a lucky value. It still did not transfer. **A plateau is insufficient evidence of
transfer**, which is the strongest generalisation heuristic we had.

**What follows.** The local instrumentation is correct for what it measures and
it caught eleven bad changes; it cannot identify which good changes matter. Do
not gate a submission on local RHAE. The remaining distance is architectural —
the leaderboard runs 7-19% against our 0.13 over 3284 teams, and that is not a
candidate-ordering problem.

## Promoted: the bootstrap probe was costing more than it bought (2026-09-25)

The ablation's largest signal, followed through. `_probe_moves` spent one action
per directional move to seed the lattice; `PROBE_LIMIT` caps that, and swept
0-4 the roster reads **0.3827, 0.3827, 0.3081, 0.3074, 0.1868**. Four of five
values beat the shipped setting and the response is a trend, so `sweep.py`
returns **STABLE** — the first such verdict in this project.

Confirmed over five seeds: **0.1561% -> 0.3520%, +0.1959pp**, against a 0.0560pp
two-sigma bar.

**The gain is not the unlock it first looks like.** Per game:

| game | base | PROBE_LIMIT=1 | delta | levels |
| --- | --- | --- | --- | --- |
| **ar25** | 0.0088% | **1.8701%** | +0.0745pp | 1 -> 1, **~570 actions -> ~39** |
| g50t | 0.0000% | 3.5714% | +0.1429pp | 0 -> 1, at its cap |
| ls20 | 0.3738% | 0.0000% | -0.0150pp | 1 -> 0 |
| sp80 | 0.1965% | 0.0345% | -0.0065pp | 1 -> 1 |

ar25 clears the *same* level roughly 14.6x faster against a 32-action human
baseline, and efficiency on a game already won is where the +2.203pp of measured
headroom lives.

**But the result does depend on g50t, and an earlier note here said otherwise.**
Counting both losses rather than only ls20: ar25's +0.0745pp against -0.0215pp
leaves **+0.0530pp without g50t, which is inside the 0.0560pp two-sigma bar**.
So 73% of the gain comes from one game unlocking, and discarding it leaves a
change that is positive but not measurably so. That is the concentration pattern
RRSI's critic exists to reject as benchmark-specific.

What still argues for it: the sweep plateau, the fully deterministic per-game
effect (every game 5/5 or 0/5, no partial flips), and a 10:1 magnitude
asymmetry — +0.2174pp of gains against -0.0215pp of losses, two games each way.
If the hidden set holds ar25/g50t-like and ls20/sp80-like games in similar
proportion the asymmetry carries; if it does not, the losses stand and the gains
may not.

**`ab.py` fails this on the no-regression rule, and it was promoted anyway.**
ls20 goes 5/5 seeds to 0/5, and it goes at every value below four — it is the
one game that needs the full bootstrap. The rule exists because a trade "will
not reproduce on the hidden set", but it was written for *lucky* trades; the
sweep establishes this is a mechanism. Recorded as a deliberate override rather
than a pass, because the hidden set is out-of-distribution by design and the
balance there is unknown.

`PROBE_LIMIT = 1` rather than 0: identical on all 25 games, and one surviving
probe leaves the mechanism available to a hidden game that needs it.

## Why ls20 needs the full bootstrap, and why the general fix is worse (2026-09-25)

ls20 is the one game lost by capping the probe, so it was worth asking what it
uses the bootstrap for. Instrumented at both settings:

| | `PROBE_LIMIT=4` | `PROBE_LIMIT=1` |
| --- | --- | --- |
| clears | step 68 | **never** |
| lattice reaches 4 | step 5 | **never** (tops out at 3) |
| affordance fires | 165x | 126x |

**ls20 needs all four directions.** With a partial lattice its router can never
navigate. ar25 is the opposite: it clears by frontier in ~39 actions and the
lattice only diverts it.

That suggests deferring the probe rather than capping it — let a game the
frontier solves finish first, and still bootstrap one that needs routing. The
obvious form, `PROBE_AFTER = N` steps, is exactly the overfitting this project
keeps finding: a number chosen so ar25's 39-action solve lands first, fitted to
the public 25 while the scored set is out-of-distribution by construction.

So the adaptive form was measured instead — probe only once `_is_stuck()`
reports the board has stopped yielding unseen states, reusing the existing stall
detector and adding no tuned constant. **It is strictly worse:**

| | baseline | `PROBE_LIMIT=1` | demand-driven |
| --- | --- | --- | --- |
| mean | 0.1561% | **0.3520%** | 0.3508% |
| ar25 | 0.0088% | 1.8701% | 1.8701% |
| g50t | 0.0000% | 3.5714% | 3.5714% |
| ls20 | 0.3738% | 0.0000% | 0.0000% |
| tu93 | 0.0357% | 0.0357% | **0.0044%** |

Same gains, ls20 still lost, and tu93 now loses a level too — deferring the
probe means tu93's lattice arrives too late for its second level. Reverted.

**What this says about generalising.** The benefit is not "probe when needed",
it is "probe less". `PROBE_LIMIT` is still a constant selected on the public 25,
but it sits on a plateau — 0, 1, 2 and 3 all beat 4 — rather than at a tuned
optimum, and a plateau is the shape that survives a distribution shift. The
adaptive alternative was the principled answer and the measurement rejected it.

## Reachability: the 19 non-scoring games split in two (2026-09-26)

`reachability.py` reads the agent's own `StateGraph` after a 6000-action run and
asks whether a game that scores nothing has anything left to explore. The answer
partitions them, and neither half is a budget problem in the way the budget
argument assumed.

**Five games have exhausted everything reachable**, with state spaces so small
the agent can barely act:

| game | states | untested |
| --- | --- | --- |
| vc33 | **1** | 0 |
| tr87 | **1** | 0 |
| ft09 | **2** | 0 |
| dc22 | 9 | 0 |
| tn36 | 62 | 0 |

One or two reachable states means essentially nothing the agent proposes changes
the board. That matches what was already known from the other direction — tr87 is
click-inert, and ft09 and vc33 are the games whose winning click is not
proposable. **More budget cannot reach them**; the candidate generator is the
binding constraint.

**Fourteen games have thousands of untested pairs**, but far more than brute
force can cover: cd82 holds 2728 states and **31378 untested pairs** at 6000
actions, cn04 40610, r11l 33078. Doubling the budget roughly doubles coverage and
is still monotone-safe, but it will not exhaust these — and the oracle clears
cd82 in **five actions**, so a short winning path exists inside a space the
search wanders without finding.

**This corrects the premise behind the budget change.** State spaces measured at
900 steps (tu93 156, lp85 431, ls20 562) grow into the thousands by 6000, so
"exhaustive coverage is affordable" is false at scale — the chrome-masked
signature still fragments. The budget increase remains justified by the scorer
being monotone in it, but it should not be expected to unlock these games.

What the partition does say is that **search direction, not search volume, is the
constraint on the 14** — consistent with recall@1 sitting at 18% overall and at
the chance floor on tu93.
## Re-ablated against the new baseline, and the verdicts invert (2026-09-25)

The first ablation measured every mechanism against `PROBE_LIMIT=4`. That
baseline no longer exists, and four of its five verdicts rested on ar25 being
lost — the game that changed most, now clearing in ~39 actions by pure frontier
instead of ~570. Re-run against the shipped agent (0.3520%):

| mechanism off | RHAE | delta | regressions | first ablation |
| --- | --- | --- | --- | --- |
| `USE_INERT` | 0.1343% | **-0.2177pp** | ar25, g50t | -0.0140pp, ar25 |
| `USE_FRONTIER` | 0.1869% | **-0.1651pp** | g50t, tu93 | -0.0280pp |
| `USE_AFFORDANCE` | 0.2795% | -0.0725pp | ar25 | -0.0180pp |
| `USE_CHURN_MASK` | 0.3316% | -0.0204pp | none | -0.0207pp, ar25 |
| `USE_GOAL_INDUCTION` | 0.3802% | +0.0281pp | none | +0.0281pp |

**`_inert` goes from the thinnest case to the strongest**, -0.0140pp to
-0.2177pp, and removing it now costs ar25 *and* g50t. Its docstring has asked
since #29 for a re-measurement after the chrome mask; this is it. The mechanism
follows: with probing capped the agent leans on pure frontier exploration, and
`_inert` is what stops it re-taking dead actions — ar25's 39-action solve
depends on it.

Three mechanisms are now strongly justified where all four previously rested on
a single game each with deltas inside the noise bar. **Reading the first
ablation against the current agent would have been wrong**, which is the same
stale-baseline error the door-inducer thread nearly repeated.

**Neither "REMOVABLE" is removed.** Both sit inside the 0.0560pp bar, so they
are "no evidence" rather than free. `USE_GOAL_INDUCTION` in particular is
*adaptive* — it learns a goal colour from that game's own winning click at
runtime — rather than a constant fitted to the public 25, and adaptive
mechanisms are the ones most likely to carry to an out-of-distribution set.

The rule this suggests: **prune tuned constants aggressively, keep adaptive
mechanisms unless they demonstrably hurt.** Capping the probe was a constant
selected on 25 games and removing it paid; goal induction adapts per game and
should not be cut on a result inside noise.

## The field_box crop is load-bearing; the chrome mask alone is not (2026-09-26)

`frame_signature` crops to the field box *and* `_settled` masks chrome, two
mechanisms for one job — keeping HUD churn out of the state key. Since the crop is
[what blinds five games](#field_box-blinds-the-state-signature-on-the-five-stuck-games),
the obvious test is whether the mask alone carries it. It does not.

| arm | RHAE | sd over 5 seeds |
| --- | --- | --- |
| baseline | 0.3520% | 0.0280 |
| `USE_FIELD_CROP = False` | **0.1859%** | **0.0000** |

`-0.1661pp` on a pooled sd of `0.0198pp`, and **g50t, sp80 and tu93 all stop
scoring in every seed**. That makes the crop the most load-bearing mechanism
measured on this agent — ahead of `_inert` at `-0.2177pp` only because that
ablation was taken against a different baseline.

Whole-board keying fails for the reason `frame_signature`'s own TODO predicts: it
keys on every pixel, so incidental per-frame churn the mask does not catch
fragments one position into many states (live ls20: 741 signatures for 30 avatar
cells). The frontier then never runs out of "unseen" states, so it never routes,
and a cycle is indistinguishable from progress. The crop is not merely a HUD
filter — **it is the denoiser**, and it works by throwing away most of the board.

**The five blind games are closed to signature-level fixes.** Three attempts now:
whole-board `_field` changed no level count on any of the five; whole-board
signature costs three scoring games; and the crop cannot be both narrow enough to
denoise ls20 and wide enough to see tr87's play area, because on tr87 **0% of the
changing cells are inside the box**. A per-game adaptive box is the only remaining
shape, and it would have to re-key the graph as it grows — the exact fragmentation
that just cost three games. Park this line.

**What the two failures share is the diagnostic.** Both this and the regressive-edge
attempt land at `~0.186%` with `sd = 0.0000` — the same floor, where only ar25 and
lp85 still score. A candidate sd of exactly zero across five seeds is now a known
signature of "the change removed the agent's ability to discriminate states", not
of a stable improvement. Treat `sd -> 0` as a failure indicator in its own right.

## Deferring "irreversible" edges costs three games (2026-09-26)

Online-graph-exploration theory says exploration cost on an unknown *directed*
graph is governed by deficiency `d`, the edges needed to make it Eulerian, so the
principled policy is to explore irreversible actions last. The published 3rd-place
ARC-AGI-3 graph explorer lost 16 -> 12 private levels to one instance of this: a
reset action recorded as a self-edge on the start node, which it then kept
re-selecting. Two detectors were tried and **both measured worse**; the mechanism
is reverted.

| arm | RHAE | sd over 5 seeds |
| --- | --- | --- |
| baseline | 0.3520% | 0.0280 |
| defer regressive edges | **0.1861%** | **0.0000** |

`-0.1659pp` against a pooled sd of `0.0198pp` — 8.4 sd, and the gate names three
regressions: **g50t stops scoring in all 5 seeds, sp80 stops scoring, tu93 drops
2 levels to 1.** Losing g50t alone undoes most of the roster's score.

**Why it fails: on these games, returning to an earlier state is how play works.**
The demotion is per primitive, so a single observation of "this discarded
progress somewhere" deprioritises the action everywhere — including where it is
the winning move. A candidate sd of exactly `0.0000` is the tell: the demotion
overrode the seeded tie-break entirely, so all five seeds played one trajectory.
The mechanism did not add caution, it replaced the search.

**The theory is not wrong; the evidence for it is absent here.** Deficiency counts
edges that *cannot be undone*, and the graph holds almost no evidence of those —
early on, every newly-discovered state has no known path back, so the test either
fires on everything or waits for a return route that a frontier walk supplies
anyway. Both proxies tried are proxies for reachability, and both misread ordinary
structure:

- **Return to the remembered level-start signature.** Fires on a legitimate hub.
  It also carried a real bug worth recording: the anchor is captured while
  `_settled` is still unmasked, inside the first `CHURN_WARMUP = 20` steps, then
  compared against chrome-masked frames, so on any board with chrome a genuine
  return can never equal it. Silently dead on exactly the games that score.
- **A drop of more than one step in graph depth.** First-discovery depth is not
  shallowness, so a shortcut into an early-discovered node reads as a reset.

Note what the published bug actually was: a self-edge, i.e. an action that changes
nothing. `_learn_inert` already owns that case, and owns it better — it keys on
byte-identity rather than on a signature that
[the crop can collapse](#field_box-blinds-the-state-signature-on-the-five-stuck-games).
The lesson generalises the earlier one about tuned constants: **prune proxies as
aggressively as constants.** A mechanism justified by theory still has to be
measured on the quantity the theory names, and "came back to somewhere older" is
not "cannot get back".

**What would justify retrying:** instrument edges with no known return path after
a full 6000-action run and count them against the ones these proxies marked. If
that set is large and the proxies caught a small arbitrary slice, a reachability
test is worth building. If it is near-empty, there is nothing here to defer and
the deficiency argument simply does not bind on this roster.

## `field_box` blinds the state signature on the five stuck games (2026-09-26)

The five games reachability found exhausted are **not inert** — almost everything
the agent does moves their board:

| game | available | agent proposes -> live | dense clicks -> live |
| --- | --- | --- | --- |
| vc33 | [6] | 7 -> **7** | 1024 -> **1024** |
| tr87 | [1,2,3,4] | 4 -> **4** | — |
| ft09 | [6] | 12 -> 4 | 1024 -> 72 |
| dc22 | [1,2,3,4,6] | 16 -> **16** | 1024 -> **1024** |
| tn36 | [6] | 12 -> **12** | 1024 -> **1024** |

vc33 has 1024 working clicks out of 1024 and reachability reported **one** state.
The chrome mask is not the cause — signature counts are identical with it off.
The cause is the **crop**: `frame_signature` keys on `field_box`, the modal
colour's extent, and that extent misses where the game happens.

| game | field_box | cells that ever change | inside the box |
| --- | --- | --- | --- |
| **tr87** | rows 0-33, cols 0-63 | 241 | **0 (0%)** |
| **vc33** | rows 1-63, cols 0-51 | 64 | **0 (0%)** |
| dc22 | rows 10-53, cols 0-31 | 100 | 36 (36%) |
| ft09 | rows 0-62, cols 0-63 | 88 | 36 (41%) |
| tn36 | whole board | 61 | 61 (100%) |

tr87 visits **523 distinct boards and hashes them all to one signature**; vc33
51 boards to one. The frontier is blind by construction, so there is nothing for
it to explore and no amount of budget or search direction helps.

**This is one root cause with two symptoms.** The earlier note that vc33's
winning button sits at col 60, outside the field box ending at col 51, is the
same defect seen from the candidate side: the crop both hides winning actions
from `in_field` and collapses the state key.

**Perception is necessary but not sufficient.** Patching `_field` to return the
whole board changes no level count on any of the five — they still clear nothing
in 600 actions. Seeing more states is not the same as finding the goal, so this
is a prerequisite for the other mechanisms rather than a fix on its own. A real
fix should be adaptive — widen the box to contain the cells observed to change,
which is self-correcting and needs no constant — rather than dropping the crop
globally, since the crop exists to keep HUD churn out of the key for the games
that do score.

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
