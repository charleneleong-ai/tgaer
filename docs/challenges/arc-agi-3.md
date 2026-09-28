# ARC-AGI-3 (ARC Prize 2026)

The interactive grid-game benchmark. An agent sees a 64×64 board plus the action
ids available to it and picks one action per turn to complete levels. Scored on
**action efficiency**, not just completion.

Harness-level docs live in the [README](../../README.md); this file is the
challenge-specific record. The long-form investigation log is
[arc-agi3-world-model.md](../arc-agi3-world-model.md), the kernel/submission
mechanics are in [arc-agi3-kaggle.md](../arc-agi3-kaggle.md).

## Scoring, and why it drives everything

`evaluate.py` scores each environment as

```
env  = min(cap, weighted)
cap      = k(k+1)/2 / total_weight            # k levels cleared
weighted = Σ_{l≤k} l · min(1.15, (baseline_l / ours_l)²) / total_weight
```

RHAE is the mean of `env` over the games. Two consequences shape all the work:

- **Efficiency is squared.** Taking 10× the human action count scores 1%. Every
  scoring game we have is efficiency-limited, not level-limited — we capture
  0.02–15% of the `cap` we are already entitled to.
- **Actions on levels that never complete are free.** A level's action count is
  fixed when it clears, so `env` is monotone non-decreasing in the action budget.

## Dataset split — the central constraint

| set | count | role |
| --- | --- | --- |
| public demo | **25** | our entire local roster |
| semi-private | 55 | scored |
| fully private | 55 | scored |

The ARC-AGI-3 paper states the public set is *"a demonstration interface"* and
the private set is *"intentionally out-of-distribution relative to the public
set"* to resist overfitting. **Our local bench is the demo set.**

**The leaderboard never scores our roster.** `arc_agi3_kaggle.py` runs one thread
per game, "110 in the competition rerun" — the 55 semi-private plus the 55 fully
private — and Kaggle splits that one parquet into its two columns:

| column | scores | available |
| --- | --- | --- |
| `publicScore` | ~55 semi-private games, never seen locally | per submission, 1/day |
| `privateScore` | ~55 fully private games | withheld until the close |

So `publicScore` **is** an out-of-distribution reading, not a re-score of the demo
set. That makes the decoupling result stronger than first stated: a 2.25x gain on
the dev set moved a 55-game held-out score by zero. Local RHAE is a regression
guard; `publicScore` is the objective, at one reading per day.

Scores oscillate in a 0.13–0.14 band across agents that differ greatly locally
(v75 0.1561% and v76 0.3520% both scored 0.13; v73 0.1879% scored 0.14), and the
reported granularity is 0.01. **Treat a single 0.01 step as noise**, not a lift;
establishing one needs same-build repeats.

## Current results

Local, 25 games, 5 seeds at 600 actions:

| | RHAE | levels | games scoring |
| --- | --- | --- | --- |
| before the probe cap | 0.1561% ± 0.0280 | 6–8 | 5 |
| **current** | **0.3520% ± 0.0280** | 8 | **6** |

At the 6000-action shipping budget, 3 seeds: 0.1704% ± 0.0296, 9–12 levels.

Kaggle public:

| submission | agent | local RHAE | public |
| --- | --- | --- | --- |
| v77 | explorer, 12000 actions + colour-agnostic roles | 0.3520% | **0.14** |
| v76 | explorer, `PROBE_LIMIT=1` | 0.3520% | 0.13 |
| v75 | explorer, baseline | 0.1561% | 0.13 |
| v73 | explorer | 0.1879% | 0.14 |
| v54 | 27B LLM agent | — | **0.17** (best ever) |

Standing: **0.13 against a 0.30 median and a 19.40 leader over 3284 teams** —
bottom quartile. Frontier LLMs score 0.25–0.37% on this benchmark, so the
leaderboard is far above frontier-model performance; competitors get there by
extreme optimisation, not by a better prompt.

## Levers measured and closed (2026-09-26/27)

Everything below was gated at 5 seeds against a `0.030pp` noise floor and a pooled-sd
no-regression rule. One shipped.

| lever | result | verdict |
| --- | --- | --- |
| action budget 6000 -> 12000 | `+0.0001pp`, one extra level (ar25) | shipped, [#38](https://github.com/charleneleong-ai/tgaer/pull/38) |
| defer irreversible edges (deficiency theory) | `-0.1659pp`, g50t + sp80 + tu93 lost | rejected, [#40](https://github.com/charleneleong-ai/tgaer/pull/40) |
| drop the `field_box` crop | `-0.1661pp`, same three lost | rejected, [#40](https://github.com/charleneleong-ai/tgaer/pull/40) |
| re-validate the avatar latch | tu93 2 levels -> 1 | rejected |
| `(colour, size)` class compression | candidates collapse only 1.4x median | not built |
| colour-agnostic role inference | `+0.0000pp`, **bit-identical** | shipped, [#41](https://github.com/charleneleong-ai/tgaer/pull/41) |

**Four of five mechanism levers are closed, and the one change worth defending moved
local RHAE by exactly nothing.** That is the loop working: each rejection came from
the gate, at roughly 8 sd, and three of them had published theory behind them.

What the failures establish, which is worth more than the fix would have been:

- **The crop is the denoiser, not a HUD filter.** Whole-board keying fragments one
  position into hundreds of states, so the frontier never exhausts and a cycle reads
  as progress. The five signature-blind games are therefore closed to signature-level
  fixes.
- **`sd -> 0` is a failure signature.** Both rejected mechanisms landed at `~0.186%`
  with a candidate sd of exactly zero across five seeds — the change had destroyed
  state discrimination, leaving one trajectory. A tight variance is not stability.
- **Navigation is not always better than blind search.** Giving tu93 a correct avatar
  made it worse; its frontier walk beats its own navigation, and the mis-pinned
  avatar was accidentally protecting it.
- **Cost is state fragmentation, not candidate breadth.** The agent weighs only
  21-101 distinct candidates per frame yet spends 1463-1707 actions, because the graph
  re-tests candidates at every new signature. That is why `_inert` — the one mechanism
  generalising per primitive *across* states — is the most load-bearing part of the
  agent at `-0.2177pp`.

**Eight constants remain fitted to these 25 games** (`PROBE_LIMIT`, `MIN_NOVELTY`,
`WALK_WINDOW`, `STUCK_WINDOW`, `FIELD_SWITCH_MARGIN`, `CHURN_FRACTION`,
`CHURN_WARMUP`, `_RECENT_CELLS`). `MIN_NOVELTY = 0.15` sits inside a 0.11-0.18 gap
measured on five games, which is capacity fitted to noise. Deleting an *inert* one
would be a free out-of-distribution win; `_RECENT_CELLS` was swept first and is not
inert (0 reads 0.3155% against 0.3520%), so no free deletion there.
## What we optimise, and against what bar

**The objective is `publicScore`, not local RHAE.** The scored rerun plays 110 games
and Kaggle splits that parquet in two; `publicScore` is ~55 semi-private games we have
never seen, and our 25-game roster is never scored at all. So:

| signal | role | cadence |
| --- | --- | --- |
| `publicScore` | **the objective** — a genuine held-out reading | 1 per day |
| local RHAE | regression guard only, never the target | minutes |
| wall clock | deadline guard — RHAE cannot see it | minutes |
| `privateScore` | unavailable until the competition closes | never, mid-competition |

`privateScore` is blank on every completed submission, so a final private lift cannot
be verified before the close. A semi-private lift is the strongest evidence available,
and that is what the loop targets.

**A change is worth a submission slot only if it clears all three guards:**

1. **No local regression.** `ab.py` at 5 seeds, no game losing depth or frequency.
   Local RHAE *gain* is not required — a change argued off-roster can be flat here and
   still be right, as the colour-agnostic fix was (bit-identical, twice).
2. **No wall-clock regression.** `ab.py`'s `throughput_verdict` fails a candidate more
   than `1.25x` slower than baseline, or either arm projecting past the kernel's 7.5h.
   This exists because RHAE counts actions, not seconds: one change left RHAE untouched
   while costing `1.96h` of the budget, and the score gate could not see it.
3. **An argument that survives without a local score.** Fewer fitted constants, no
   public-set literals, a mechanism rather than a case. Dev-set gain
   [anti-correlates with out-of-distribution gain](#the-local-bench-does-not-predict-the-score--tested-directly-2026-09-25),
   so a large local win is a reason for suspicion, not confidence.

**The `publicScore` noise bar is being measured.** Scores oscillate 0.13–0.14 across
agents that differ 2.25x locally, and the reported granularity is 0.01 — so until a
same-build repeat quantifies the spread, no single 0.01 step can be read as a lift.
That repeat is what today's slot was spent on. Until it lands, treat any 0.01 move as
noise.

## The result that governs the rest

**A 2.25× local gain moved the public score by zero.** Coupled metrics would have
taken public from 0.13 to roughly 0.29, about six times its noise bar. The
predicted effect was large and entirely absent.

So local RHAE is a **regression guard, not a promotion criterion**. It caught
eleven bad changes; it cannot identify which good changes matter. A submission
must be justified by derivation from the scorer, or by an algorithmic argument —
not by the local number.

## Agents

| agent | where | state |
| --- | --- | --- |
| `ExplorerArcAgi3Agent` | [`arc_agi3_explorer.py`](../../src/tgaer/agents/arc_agi3_explorer.py) | **ships** — model-free, frontier-driven |
| `ExplorerAgent` | [`arc_agi3_kaggle.py`](../../src/tgaer/agents/arc_agi3_kaggle.py) | kernel seam for the above |
| `MyAgent` | same file | 27B LLM agent; clears nothing on the roster but holds the best public score |
| `ArcAgi3LLMAgent` | [`arc_agi3_llm.py`](../../src/tgaer/agents/arc_agi3_llm.py) | Gemini / local vLLM, scores 0 |

The explorer's mechanisms, each measured by ablation:

| mechanism | removing it costs |
| --- | --- |
| `_inert` — demote primitives that changed nothing | **−0.2177pp**, ar25 + g50t |
| frontier routing — walk back to an untested state | **−0.1651pp**, g50t + tu93 |
| affordance — steer toward a salient object | −0.0725pp, ar25 |
| chrome mask — flatten animating cells from the state key | −0.0204pp, nothing |
| goal induction — learn a goal colour from a winning click | +0.0281pp, nothing |

## Bench tools

Under `sia-oss/bench/`:

| tool | what it answers |
| --- | --- |
| `measure.py` | one 25-game run → RHAE |
| `ab.py` | promotion gate: N seeds, 2 pooled sd, per-game frequency **and** depth |
| `sweep.py` | is a gain a trend across a parameter range, or a spike? |
| `ablate.py` | switch each mechanism off; what is it worth? |
| `reachability.py` | is a game budget-limited or out of reachable states? |
| `oracle_labels.py` / `oracle_recall.py` | offline oracle's winning moves, and whether the agent can propose them |
| `m0_suite.py` | uninformed offline planning across the roster |

Measured noise floor: **sd ≈ 0.030pp** seed-to-seed, so the 2σ bar is ≈0.060pp.
Anything smaller is unreadable.

## Rules of thumb earned the hard way

- **Prune tuned constants aggressively; keep adaptive mechanisms** unless they
  demonstrably hurt. A constant fitted to 25 games carries nothing; a mechanism
  that adapts per game at runtime might.
- **A sweep plateau is not proof of transfer.** `PROBE_LIMIT` 0–3 all beat 4 and
  it still moved the public score by zero.
- **Re-ablate after any promoted change.** Ablation verdicts do not survive a
  baseline shift — `_inert` went from −0.0140pp to −0.2177pp.
- **Never spend the daily slot on a kernel whose build is not `COMPLETE`**, and
  read the kernel log for `Mock agent:` and a non-zero level count first. A
  build defaulting to the wrong agent nearly cost a slot.
