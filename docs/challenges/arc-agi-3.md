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
| v76 | explorer, `PROBE_LIMIT=1` | 0.3520% | **0.13** |
| v75 | explorer, baseline | 0.1561% | 0.13 |
| v73 | explorer | 0.1879% | 0.14 |
| v54 | 27B LLM agent | — | **0.17** (best ever) |

Standing: **0.13 against a 0.30 median and a 19.40 leader over 3284 teams** —
bottom quartile. Frontier LLMs score 0.25–0.37% on this benchmark, so the
leaderboard is far above frontier-model performance; competitors get there by
extreme optimisation, not by a better prompt.

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
