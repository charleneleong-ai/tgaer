# TGAER – Toward General-Purpose Abstraction & Embodied Reasoning

Monorepo for agent experiments on abstraction-and-reasoning benchmarks. The
active line of work is **ARC-AGI-3** (the interactive grid-game benchmark from
the ARC Prize): an agent sees a 64×64 board + the action ids available to it and
must pick an action each turn to complete levels. ORAK embodied-reasoning
challenges live alongside as git submodules under `challenges/`.

## Layout

```
src/tgaer/
  agents/        agent policies — RandomArcAgi3Agent (floor), ArcAgi3LLMAgent (Gemini / local vLLM)
  envs/arc_agi3/ interactive env adapter + hosted-API transport (three.arcprize.org)
  guards/        env-agnostic trajectory guards (futile-action, repeated-plan) + GuardedAgent
  evaluation/    dispatch (run_eval, routed on env.kind) · eval loop · wandb logging
  cli/           tgaer-eval entrypoint
  core/          Agent / Environment base types
configs/experiments/   one YAML per run (env + agent + guards + wandb)
challenges/            orak + poetiq solver (git submodules)
tests/                hermetic — scripted transport + mocked LLM, no network
```

## How an ARC-AGI-3 eval runs

`run_eval(cfg)` dispatches on `cfg["env"]["kind"]`. For `arc_agi3` it builds the
env over a live transport, wraps the configured agent in a `GuardedAgent` (the
guards inject a planner hint when the agent loops), opens a scorecard, and runs
one episode — reward is the per-step `levels_completed` delta. With a `wandb:`
block enabled it logs per-step metrics + the board rendered as an image.

Agents are selected by `agent.kind`: `random` (seeded floor) or `llm`
(`ArcAgi3LLMAgent`, one litellm client driving either Gemini or a local
OpenAI-compatible endpoint — set `model` + `api_base`).

## Quickstart

```bash
uv sync
# secrets in .env: ARC_API_KEY (required), GEMINI_API_KEY and/or WANDB_API_KEY (optional)

# random baseline against a live game
uv run tgaer-eval run configs/experiments/arc_agi3_guarded.yaml

# local open-weights Qwen3.6 on vLLM (:8000), logged to wandb
uv run tgaer-eval run configs/experiments/arc_agi3_qwen36.yaml

uv run pytest -q          # tests (hermetic)
mise run lint             # ruff format + check
```

## Baselines (game `ls20-9607627b`, 80 steps)

| Agent | score | levels | notes |
|---|---|---|---|
| random | 0.0 | 0/7 | floor |
| Gemini flash-lite | 0.0 | 0/7 | loops (guards fire ~76/80) |
| Qwen3.6 (local) | 0.0 | 0/7 | varied moves, no progress |

All frozen LLMs score 0/7 — expected (even Gemini 3.1 Pro scores ~0.37% on
ARC-AGI-3). Closing the gap needs training (RL), not a bigger frozen prompt; the
local open-weights path (Qwen3.6 on vLLM) is the foundation for that.
