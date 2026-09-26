# TGAER – Toward General-Purpose Abstraction & Embodied Reasoning

A harness for running agents against abstraction-and-reasoning benchmarks. The
harness is benchmark-agnostic: an environment adapter, an agent, optional
trajectory guards, and an eval loop that logs what happened. Each benchmark
brings its own adapter and its own results file under
[`docs/challenges/`](docs/challenges).

## Layout

```
src/tgaer/
  core/          Agent / Environment base types
  agents/        agent policies, per benchmark
  envs/          environment adapters — arc_agi3 (wired), arc, orak
  guards/        env-agnostic trajectory guards (futile-action, repeated-plan)
                 + GuardedAgent, which injects a planner hint when an agent loops
  evaluation/    dispatch (run_eval, routed on env.kind) · eval loops · scorers
                 · wandb logging
  optimization/  dspy signatures + a GEQ loop
  logging/       langsmith / mlflow helpers
  cli/           tgaer-eval entrypoint
configs/experiments/   one YAML per run (env + agent + guards + wandb)
docs/challenges/       per-benchmark results and constraints
challenges/            orak + poetiq solver (git submodules)
tests/                 hermetic — scripted transport + mocked LLM, no network
```

## How an eval runs

`run_eval(cfg)` dispatches on `cfg["env"]["kind"]`. The loader builds the
environment, wraps the configured agent in a `GuardedAgent`, runs episodes, and
returns an `EvalResult`. With a `wandb:` block it logs per-step metrics and, for
grid environments, the board as an image.

Registered kinds are `arc_agi3`, `arc` and `orak`. **Only `arc_agi3` is wired**
— the other two are registered and raise on use, so `available_kinds()` reports
them but they are not yet runnable.

Agents are selected by `agent.kind`. Overrides passed to `run_eval`
(`transport=`, `agent=`) are the dependency-injection seam that keeps the test
suite off the network.

## Quickstart

```bash
uv sync
# secrets in .env: ARC_API_KEY (required for live games),
# GEMINI_API_KEY and/or WANDB_API_KEY (optional)

uv run tgaer-eval run configs/experiments/arc_agi3_guarded.yaml

uv run pytest -q          # 533 tests, hermetic
mise run lint             # ruff format + check
```

## Benchmarks

| benchmark | status | results |
| --- | --- | --- |
| **ARC-AGI-3** (ARC Prize 2026) | active | [`docs/challenges/arc-agi-3.md`](docs/challenges/arc-agi-3.md) |
| ORAK (embodied reasoning) | submodule, adapter not wired | — |
| ARC-AGI-1/2 | adapter not wired | — |

## What this harness is for

The working assumption is that a frozen model plus a prompt does not close these
benchmarks — every frozen LLM we have measured scores at or near zero on
ARC-AGI-3, and published frontier-model numbers agree. What moves the score is
the machinery around the model: search, state abstraction, exploration policy,
and the discipline to tell a real gain from noise.

So the harness carries measurement tooling as a first-class concern rather than
an afterthought. Two lessons from it generalise beyond any one benchmark:

- **Measure the noise floor before trusting a delta.** Seed-to-seed spread on
  ARC-AGI-3's local roster is sd ≈ 0.030pp; comparing single runs promoted
  changes that later evaporated, and eleven consecutive "improvements" failed
  once judged against that floor.
- **A local bench can be sound and still non-predictive.** On ARC-AGI-3 a 2.25×
  local gain moved the public score by zero, because the scored set is built to
  be out-of-distribution from the public one. Local scores are a regression
  guard; promoting on them is a separate claim that needs its own evidence.
