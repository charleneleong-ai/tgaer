# ARC-AGI-3 prior art: what the teams above us actually do

Gathered 2026-09-28 from the MLST interview with the winning team
([video](https://www.youtube.com/watch?v=Vg6FBKTlfOw), Machine Learning Street Talk,
"ARC-AGI-3 winning team — Millennia of minds, compressed into words") and the
primary sources it points to.

Our standing: **publicScore 0.13–0.14** against a 0.30 median.

## The scoreboard we are actually on

| team / system | score | architecture |
| --- | --- | --- |
| **Duck harness** (Tufa Labs) — Milestone 1 **winner** | **1.21%** public | LLM-as-programmer in a REPL, Qwen 3.6 27B FP8 |
| AERA (Liew Keong Han) | 0.2116 public 25-game, **0.30** private 55-game | EXPLORE → VERIFY → PLAN, Qwen2.5-**0.5B** |
| **ours** (model-free explorer) | **0.13–0.14** | directed state graph over frame signatures |

Two things to sit with. The winner is ~9x us. And **AERA beats us roughly 2x using a
0.5B model** — so our deficit is not compute and not model size, it is architecture.

## Duck harness — the winning method

Source: [tufalabs.ai/research/duck-harness](https://tufalabs.ai/research/duck-harness/),
code at [github.com/Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness).

- A **REPL**: the agent is a tool-using solver, not a policy emitting actions.
- Game observations are **encoded as Python variables the model inspects via tool
  calls**, with **both image and text representations** of the grid.
- **Pre-built helper functions** are provided for the model to call.
- It **"encodes the game as a programming problem"** — the model writes Python.
- **Context is kept short by automatically evicting the oldest messages.**
- Model: **Qwen 3.6 27B FP8**. Reported on 25 public games x 20 passes:
  **mean 1.6002 ± 0.4475**.

The ±0.4475 is worth noting on its own: even the winning harness has run-to-run
spread of that order, which is the same reason our 0.13 vs 0.14 steps are unreadable
(see the variance probe).

## AERA — the frontier framing, and it matches our own measurement

Source: [arXiv 2605.25931](https://arxiv.org/abs/2605.25931), "Explore Before You
Solve: The Speed–Depth Trade-off in Epistemic Agents for ARC-AGI-3".

Three phases, **EXPLORE → VERIFY → PLAN**. Its central claim is that **RHAE's
quadratic form is a second-order penalty for deviating from the Pareto frontier
between action efficiency and information gain**, and that random / no-explore
baselines score **RHAE = 0.0000**.

That is the same structure we derived independently from `results.json`'s per-game
`cap`: score is action efficiency on levels already cleared, and the quadratic makes
deviation expensive. The paper also states **all 25 public games are reachable by
non-intelligent strategies** — consistent with our explorer clearing 7 of them with
no model at all, and with the public set being a weak signal.

## Preview-competition winners, and a tension worth naming

- **StochasticGoose** (Tufa) won the *Preview Agent Competition*: **12.58%, 18
  levels**, using a **CNN trained with RL to predict which actions cause frame
  changes** (64x64 frames, four-layer conv net).
- **Blind Squirrel** (2nd): a **directed state graph built from observed frames** —
  which is our architecture.

The tension: predicting whether an action changes the frame is exactly the
effect/inertness prediction we closed as worthless here (a perfect oracle is
`+0.0065pp` against a `0.0300pp` floor). The likely reconciliation is the
**objective**, not the mechanism — the Preview scored *levels completed*, where
knowing which actions do something buys progress, while this competition scores
*actions per level squared*, where the cost of finding out is one action. Treat our
closure as scoped to the current metric, not as a refutation of their result.

## What to adopt here

**The explorer line is near its ceiling** — seven levers gated and closed, and the
two games holding 64% of the reachable score now need opposite fixes (`tu93`: 81
states, 80% exhausted, a routing problem; `lp85`: 1624 states, 3% exhausted, a
breadth problem). Prior art says the way up is a different architecture, and it is
one we already started and dropped:

- [`feat/arc-agi3-repl-harness`](../../tree/feat/arc-agi3-repl-harness) (2026-08-22,
  "let the model act from inside its own python tool", ~1077 lines incl. 341 lines of
  tests)
- [`wip/kaggle-agent-python-tool`](../../tree/wip/kaggle-agent-python-tool)
  (2026-09-15, a richer python tool for the Kaggle agent)

Both are LLM-as-programmer, which is the Duck's design. The 27B is already in the
kernel; our earlier 27B failure tested **LLM-as-policy**, the weak architecture, so
it is not evidence against this one.

Concrete gaps between those branches and the published Duck design, in the order
they are worth closing:

1. **Python variables + tool-call inspection** of the observation, rather than the
   board pasted into the prompt.
2. **Both image and text** grid representations.
3. **Pre-built helper functions** the model can call instead of re-deriving.
4. **Oldest-message eviction** to bound context over a long run.

Read `ARC3-Inference/README.md` in the Duck repo before building — the landing page
omits TAAF's details and publishes no ablations, so the repo is the only source for
the loop and tool definitions.
