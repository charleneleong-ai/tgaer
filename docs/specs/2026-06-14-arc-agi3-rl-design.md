# ARC-AGI-3 RL — design (vendor + extend `arc-agi-3-env`, shaped reward, GSPO)

**Status:** draft / scoping. **Date:** 2026-06-14.

## Why

Every frozen baseline plateaus at **0/7** on `ls20-9607627b` — random, Gemini, Qwen3.6-35B (text),
Qwen3-VL-4B and Qwen3-VL-30B-A3B (vision), all with perception fixed and action feedback. The wall
is **skill-acquisition**, not perception or scale. The only untried lever is training a local policy
with RL — which the benchmark's no-internet/open-weights rules also require.

## What we build on (don't reinvent)

- **Vendored `verifiers` env** — [`environments/arc_agi_3_env/`](../../environments/arc_agi_3_env/VENDORED.md), copied + audited from `ryanznie/arc-agi-3-env`. A `vf.MultiTurnEnv` with `load_environment(game_family, level_index, game_id)`:
  - **local `simple_maze` (2 levels) / `complex_maze` (5 levels)** via `arcengine` — *no API*, *solvable*. This is the RL curriculum.
  - **remote `arc_agi`** — the live game via `ARC_API_KEY`/`ROOT_URL`.
  - reward `1.0 if WIN else 0.0` (+ `num_actions`/`timed_out` metrics); XML action tags.
- **tgaer harness** — [`ArcAgi3LLMAgent`](../../src/tgaer/agents/arc_agi3_llm.py) (legible encoding, action feedback, vision) + [`rendering.py`](../../src/tgaer/envs/arc_agi3/rendering.py) + wandb reasoning/trajectory logging.
- **orak RLVR stack** — `verifiers` 0.1.14 in `lagunaxs2-rlvr` (multi-turn trainer, vLLM rollouts, `laguna_rlvr.rewards`).

So the env, the curriculum, *and* the trainer already exist. The work is **extend + shape + train**, not build-from-scratch.

## Why the vendored env changes everything

The design's biggest open risk was *"is any game winnable, so the reward has variance?"* The local
**mazes answer it: yes.** They're simple enough that a policy can reach `WIN`, so the binary reward
gets cross-rollout variance → GRPO/GSPO has a gradient — **without** us having to solve reward
shaping first. And local games mean **no scorecard-API bottleneck** → fast, cheap rollouts.

## 1. Extend the vendored env

Layer tgaer's harness onto `arc_agi_3_env._format_observation` (it's barebones text today):
- **Legible encoding + per-step action feedback** (diff vs last frame) — port from `ArcAgi3LLMAgent`.
- **Board image** for VL policies — `rendering.grid_to_png_data_url`; multimodal rollouts per `lagunaxs2-rlvr/docs/a100-multimodal-adapter.md`.
- Keep its `load_environment` signature so it stays a drop-in verifiers module.

## 2. Reward

- **Mazes:** keep the binary WIN reward — they're winnable, so it already has signal. Start here.
- **Remote `arc_agi`:** WIN is too sparse (every baseline 0/7). Add shaping — novelty (hash visited
  states), sub-goal credit, anti-stall (reuse the "changed NOTHING" signal) — via
  `laguna_rlvr.rewards.shaped`. **Validate offline that shaped reward has cross-rollout variance
  before spending a training step.**

## 3. Training loop

- **Policy:** start `Qwen3.6-35B-A3B` (LoRA), served on vLLM for rollouts.
- **Algorithm:** GRPO, with **GSPO** = `importance_sampling_level="sequence"` as a swap. Home = the
  orak `verifiers` trainer; **Unsloth+TRL `GRPOTrainer`** is the alt single-A100 backend (does GSPO,
  but you supply the rollout loop verifiers gives free).
- **Single A100-40GB:** LoRA + vLLM rollouts + grad on one card (the orak RLVR pattern).

## Phased plan

1. **Wire the vendored env** into the RLVR venv (`arcengine` dep) and `vf-eval` the **local mazes**
   with our existing policies — confirms the loop + gives a non-zero baseline (mazes should be
   partially solvable even frozen).
2. **Extend observation** (encoding + feedback + image); re-eval mazes.
3. **Train** text policy (GRPO→GSPO) on the **maze curriculum** (simple → complex levels); watch WIN-rate climb.
4. **Shape + curriculum** for remote `arc_agi`; validate reward variance offline; train.
5. **Vision** policy once text shows signal.

## Open questions

- Maze WIN-rate for a frozen 35B-A3B — is it already >0 (so RL has signal on day one)?
- `arcengine` local-game fidelity vs the hidden remote games — does maze skill transfer?
- Vision RL on a 30B-A3B-VL under LoRA on 40GB — feasible but unproven here.
