# ls20 induction-telemetry replay — design

**Date:** 2026-06-25 · **Branch:** `feat/arc-agi3-explorer` · Prior: [`2026-06-25-explorer-phase4-live-roster.md`](../experiments/2026-06-25-explorer-phase4-live-roster.md)

## Problem

The Phase 4 live re-sweep was pixel-identical to Phase 3b: the directed bootstrap gave **zero** lift, and the aggregate match means the Phase 4 path *never engaged* on the keyboard games. `_nav_affordance` is gated on the avatar + move lattice being induced first; if induction doesn't fire it returns `None` and the agent falls straight back to blind exploration. The sweep measures **outcome, not internals** — it can't say *which* stage dies: avatar never pins, lattice never completes, or affordance steers wrong.

We need a local, re-runnable diagnostic that surfaces, per step on a **real** ls20 trajectory, whether the avatar pins, how large the lattice grows, and which branch of the `act()` cascade fires.

## Why a real trajectory (not the synthetic sim)

The synthetic `_Ls20LockSim` is exactly where induction *works* — replaying it proves nothing about the live failure. We must reproduce the live trajectory, which needs **real ls20 frames**. The explorer is deterministic (seed=0, no RNG in the decision cascade), so re-feeding a captured obs sequence to a fresh agent reproduces the live actions exactly — capture-then-replay is faithful, and the replay is free and re-runnable.

## Architecture

Live ls20 → `RecordingAgent` → `ls20_capture.jsonl` → replay harness → fresh `ExplorerArcAgi3Agent` (with `self.trace`) → `ls20_telemetry.jsonl` + summary.

**Capture at the agent boundary, not the transport.** The env's `reset_on_game_over` rewrites the frame and injects `obs["terminal"]` *after* the transport returns, so a transport-level recorder would capture raw API frames that differ from what `act()` saw — breaking faithful replay. Recording the `(obs, action)` the explorer actually processes makes replay exact. The sweep runs `guards=[]`, so the explorer's chosen action is the executed one — no guard interference.

### 1. Recording agent — `src/tgaer/agents/arc_agi3_recorder.py`

`RecordingAgent(inner: Agent, path: str)` implements the `Agent` interface. On each `act(obs)` it appends `{"obs": obs, "action": {"id", "x", "y"}}` to a JSONL, then delegates to `inner.act(obs)` and returns its result. `reset()` passes through. Pure passthrough — no behaviour change to the live run. (`obs` is already JSON-serialisable: `frame` is a list of grids, the rest are ints/str/bool.)

### 2. Capture script — `experiments/capture_ls20.py`

Runs **one** ls20 game live: an `ExplorerArcAgi3Agent` wrapped in `RecordingAgent`, passed to `evaluate_arc_agi3_agent` (`guards=[]`, `max_steps=1000`) over an `ArcAgi3Environment` (`MAX_ACTIONS=1000`, `reset_on_game_over=True`) driven by the real `ArcAgi3Client`. Writes `experiments/ls20_capture.jsonl`. The only paid step (~1 game budget). No scorecard — the score is irrelevant; only the obs/action stream matters.

### 3. Agent telemetry — `ExplorerArcAgi3Agent.act()`

Replace the inline `prim = a or b or c or d` cascade (`arc_agi3_explorer.py:265-270`) with explicit assignment that records the **source branch**, and populate `self.trace` each step:

```python
{"step": int, "avatar": int | None, "lattice_size": int,
 "branch": "probe" | "affordance" | "nav" | "choose", "prim": Primitive, "levels": int}
```

`avatar` = `self._det.avatar` (pinned value or `None`). This is the core signal; it is also genuine production observability, not throwaway scaffold.

### 4. Replay harness — `experiments/replay_telemetry.py`

Pure core `replay_traces(records: list[dict]) -> list[dict]`: a **fresh** explorer, for each record feeds the captured obs, asserts the replayed action matches the captured action (proves faithful reproduction; a mismatch means non-determinism crept in and the diagnostic is void), and collects `self.trace`. A thin CLI wrapper loads `ls20_capture.jsonl`, runs `replay_traces`, writes `experiments/ls20_telemetry.jsonl`, and prints a summary: step at which the avatar first pins (or `never`), max lattice size, and a branch-firing histogram. The pure core is what the tests target — no real frames needed.

## What the output tells us

- avatar **never pins** → fix is in `EmpiricalSemantics` controllability (confirmation window / motion-consistency relaxation).
- avatar pins but **lattice stays small** → the probe / lattice stage.
- both healthy but **branch is always `choose`** → `_nav_affordance` target selection (salience prior).

## Testing — `tests/test_arc_agi3_replay.py`

- **Recorder round-trip:** `RecordingAgent` writes one record per `act()`; the record's obs + action read back identical to what was passed in, and the wrapped agent's return value is unchanged (an ACTION6 click and a simple act both round-trip).
- **Replay faithfulness + telemetry on a working induction:** build a scripted moving-avatar obs sequence (via the explorer test's `_board`/`_obs` helpers) that *does* induce the avatar, serialise it as records, run `replay_traces`. Assert it reproduces the actions (no assertion error) *and* the telemetry shows the avatar pinning and a `nav`/`affordance` branch firing — proving the harness surfaces a working induction, so a flat/`None` avatar on real ls20 is a real signal, not a harness artifact.

## Out of scope

- Counterfactual replay (the captured trajectory is fixed; the replay can't explore actions the live agent didn't take).
- The fix itself — this diagnostic only localises the failing stage; the follow-up sweep depends on what it shows.
