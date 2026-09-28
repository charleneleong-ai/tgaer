# SOTA technique audit against the rules we actually play under

2026-09-28. Purpose: decide what to build next. Everything here is checked against
the competition's own constraints, because most published ARC-AGI-3 results are not
reachable from inside the scored kernel.

## The rules that decide everything

Verified against [arcprize.org](https://arcprize.org/competitions/2026/arc-agi-3) and
[docs.arcprize.org](https://docs.arcprize.org/arc-prize-2026):

- **Internet is disabled.** *"All accelerated Kaggle sessions have internet disabled."*
- Accelerators: CPU, **T4 x2** (default), **P100**, and **RTX 6000 — "reserved for
  ARC-AGI-3 notebooks."**
- All code must be open-sourced to be prize-eligible.

**So no frontier API.** GPT-5.x and Opus cannot be called from the scored run. Any
method whose numbers come from an API model is a research result on the public set,
not a submittable design.

## Two tiers, and only one is reachable

| system | score | model | in-kernel? |
| --- | --- | --- | --- |
| [Tycho](https://arxiv.org/abs/2607.28287) | **100.0 RHAE, all 183 levels**; Opus 5 used 61% fewer actions than human | GPT-5.6 Sol / Opus 5 | **no** — API |
| [EWM](https://arxiv.org/abs/2605.05138) | 58.12% mean RHAE, 15 games solved | GPT-5.5 high | **no** — API |
| [EWM ablations](https://arxiv.org/abs/2607.15439) | every public level at **41% fewer actions than human**; verification ~99% | gpt-5.6-sol max | **no** — API |
| **[Duck harness](https://tufalabs.ai/research/duck-harness/)** | **1.21%** public (Milestone 1 **winner**) | **Qwen 3.6 27B FP8** | **yes** |
| [AERA](https://arxiv.org/abs/2605.25931) | 0.30 private | **Qwen2.5-0.5B** | **yes** |
| **ours** | **0.13–0.14** | qwen3-14B Q4_K_M | yes |

The benchmark is close to solved *by frontier coding agents*. The competition is not,
because the competition forbids them. **The only published kernel-feasible results are
Duck (1.21%) and AERA (0.30)** — and both beat us.

## Which mechanisms transfer to a local model

The [ablation paper](https://arxiv.org/abs/2607.15439) is the most decision-relevant
source, because it separates the mechanisms rather than shipping them as a bundle:

| mechanism | evidence | verdict for us |
| --- | --- | --- |
| **Verification** (exact replay against recorded observations) | *"ranked first consistently"*, and **"succeeds at lower effort"** | **build first** — the only mechanism shown to pay at reduced model effort, which is our entire regime |
| **Simplification** (MDL-ish refactor toward simpler abstractions) | beat executable-only in **3 of 4** settings | second |
| **Executable world model** (agent-authored Python simulator) | *"not universally beneficial"*; **textual beat flexible-interface executable variants** | do not lead with this |
| **Active abstraction** (Tycho: decide when a model is worth its cost) | 88.49 mean RHAE under a delegation policy | the right framing for RHAE's quadratic; needs the above first |

Headline, verbatim: *"at max, the three imposed mechanisms are not required for
action-efficient public-set completion; verification nevertheless scores higher and
succeeds at lower effort."*

Two consequences worth stating plainly. The fanciest component is **not** the
highest-value one — and the one that survives weak models is the cheapest to build.
**Nobody has published these mechanisms on a small open-weight model**, so whether
verification transfers to a 27B is an open question and is exactly our opening.

## Why verification is the right first build here

Our score is `min(cap, (human/ours)^2)` per level. Two properties:

- A world model or verifier **costs zero environment actions** to consult. Planning
  offline is free in the only currency RHAE charges.
- [The kernel has no game object](../../tree/main/docs/challenges/arc-agi-3.md), so
  deepcopy planning was ruled out. An agent-authored simulator sidesteps that — we do
  not need the env to be forkable if we write our own model of it.

Verification is also the mechanism our current agent most conspicuously lacks: the
explorer executes a frontier plan **blind**, never checking that a step landed where
predicted.

## Cheap wins found while auditing, both needing a check first

1. **We are under-using the GPU.** The kernel loads `qwen3-14b.Q4_K_M.gguf` via
   llama-cpp ([`arc_agi3_kaggle.py:1159`](../../tree/main/src/tgaer/agents/arc_agi3_kaggle.py#L1159)),
   while **RTX 6000 is reserved for this competition** and the winner ran a **27B**.
   Confirm the accelerator and its VRAM, then size the model to it.
2. **Our deadline constant may be too conservative.** We assume **7.5h**
   (`KERNEL_BUDGET_H`, `_RUN_DEADLINE`); secondary sources say runtime is **<12h**.
   The official docs do not state it. If 12h is right, ~4.5h of budget is unused —
   and by RHAE's monotonicity extra actions on uncleared levels are free. **Verify
   before changing anything**, since a wrong deadline costs the whole run.

## Ranked next steps

1. **Verify the two constants above** (accelerator/VRAM, runtime). Cheap, and both
   gate everything else.
2. **Add verification to the existing explorer** — check each planned step against
   the predicted signature and re-plan on mismatch. Small, testable against the
   current gate, independent of any LLM work, and directly attacks `tu93`'s routing
   failure (81 states, 80% exhausted, 35 pairs it never reaches).
3. **Resume the REPL branches** toward the Duck design
   (`feat/arc-agi3-repl-harness`, `wip/kaggle-agent-python-tool`), closing the four
   gaps in [the prior-art doc](arc-agi-3-prior-art.md), sized to the real GPU.
4. **Then** layer simplification, and only after that an executable world model.

Note the ordering is the ablation's, not intuition's: verification before
simplification before executable models.
