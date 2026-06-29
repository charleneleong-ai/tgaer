# ls20 Induction-Telemetry Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a capture-then-replay diagnostic that isolates which stage of the explorer's directed-bootstrap induction fails on a real ls20 trajectory.

**Architecture:** A `RecordingAgent` wrapper logs every `(obs, action)` the explorer processes during one live ls20 game. The explorer gains a per-step `self.trace` exposing avatar / lattice-size / chosen branch. An offline replay harness re-feeds the captured obs to a fresh explorer, asserts faithful reproduction, and summarises where induction dies — free and re-runnable.

**Tech Stack:** Python 3.11+, numpy, pytest. No new dependencies.

## Global Constraints

- Capture at the **agent boundary**, never the transport — the env injects `obs["terminal"]` and rewrites frames after the transport returns.
- The explorer is deterministic (seed=0); replay faithfulness is asserted, not assumed.
- No behaviour change to the live agent — telemetry is write-only side state.
- Spec: [`docs/specs/2026-06-25-ls20-induction-telemetry-replay-design.md`](../specs/2026-06-25-ls20-induction-telemetry-replay-design.md).
- Lint gate before every push: `uv run ruff check .` AND `uv run ruff format --check .`.
- Tests run with `./.venv/bin/python -m pytest -q`; standalone scripts need `PYTHONPATH=src`.

## Decision chain (why this plan)

1. **Phase 4 was refuted, and *how* it failed pointed here.** The re-sweep was pixel-identical to Phase 3b — identical, not noisy, means the directed-bootstrap path *never engaged*. `_nav_affordance` is gated on avatar+lattice induction; if induction doesn't fire it returns `None` and the agent falls back to blind exploration. → **Stop re-sweeping** (a paid sweep measures outcome, not which stage dies).
2. **Diagnose, don't guess.** The failure is one of three stages — avatar never pins / lattice never completes / affordance mis-targets — indistinguishable from a scorecard. → **Build per-step telemetry** (avatar / lattice-size / branch).
3. **Real frames, not the synthetic sim.** The sim is exactly where induction *works*; only real ls20 frames reproduce the live failure. The explorer is deterministic, so replaying captured obs reproduces the live trajectory exactly. → **Capture-then-replay**: one paid capture, then free re-runnable replay.
4. **Capture at the agent boundary, not the transport.** The env injects `obs["terminal"]` and rewrites frames *after* the transport returns, so a transport recorder would capture frames `act()` never saw — breaking faithful replay. The replay asserts reproduced action == captured, so non-determinism fails loud. → **`RecordingAgent`**.
5. **Instrument production, don't reconstruct externally.** Re-deriving the branch in the harness means re-running and mutating agent state. → **`self.trace` on the real agent** (also keeps genuine observability).
6. **Tune-vs-train is downstream of this.** The failure looks like mis-tuned primitives, not missing training, and ARC-AGI-3 resists training (held-out anonymized games, sparse reward) — but we don't guess; this diagnostic distinguishes a small threshold fix from "induction fundamentally unreliable → learned perception head," for ~1 game of budget.

---

### Task 1: Explorer telemetry (`self.trace`)

**Files:**
- Modify: `src/tgaer/agents/arc_agi3_explorer.py` (`__init__` ~line 214; `act` lines 265-274)
- Test: `tests/test_arc_agi3_explorer.py` (new `TestTrace` class)

**Interfaces:**
- Produces: `ExplorerArcAgi3Agent.trace: dict | None` — populated each `act()` with
  `{"step": int, "avatar": int | None, "lattice_size": int, "branch": str, "prim": Primitive, "levels": int}`
  where `branch ∈ {"probe", "affordance", "nav", "choose"}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_arc_agi3_explorer.py`:

```python
class TestTrace:
    def test_probe_branch_tagged_first(self):
        # First directional step seeds the lattice → the probe branch fires.
        agent = ExplorerArcAgi3Agent()
        agent.act(_obs(_board(avatar=(2, 2)), actions=(1, 2, 3, 4)))
        assert agent.trace["branch"] == "probe"
        assert agent.trace["step"] == 1
        assert agent.trace["avatar"] is None  # not yet pinned on first frame

    def test_choose_branch_when_no_induction(self):
        # Click-only game: no avatar/lattice ever, so selection falls to frontier.
        agent = ExplorerArcAgi3Agent()
        board = _board(extra={5: [(4, 4)]})
        agent.act(_obs(board, actions=(6,)))
        assert agent.trace["branch"] == "choose"
        assert agent.trace["lattice_size"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest tests/test_arc_agi3_explorer.py::TestTrace -q`
Expected: FAIL — `AttributeError: 'ExplorerArcAgi3Agent' object has no attribute 'trace'`.

- [ ] **Step 3: Add the `trace` field**

In `__init__`, beside `self.last_reply = None` (~line 214):

```python
        self.last_reply: str | None = None
        self.trace: dict | None = None
        self._step = 0
```

- [ ] **Step 4: Tag the branch and populate `trace`**

Replace the cascade + tail of `act` (lines 265-274) with:

```python
        branch = "choose"
        if prim := self._probe_moves(available, lattice):
            branch = "probe"
        elif prim := self._nav_affordance(arr, available, lattice):
            branch = "affordance"
        elif prim := self._nav_move(arr, available, lattice):
            branch = "nav"
        else:
            prim = self._choose(sig, prims)
        self._step += 1
        self.trace = {
            "step": self._step,
            "avatar": self._det.avatar,
            "lattice_size": len(lattice),
            "branch": branch,
            "prim": prim,
            "levels": int(levels),
        }
        self._graph.take(sig, prim)
        self._prev_sig, self._prev_prim = sig, prim
        self._prev_arr = arr
        self.last_reply = f"[explorer] {prim}"
        return to_arc(prim)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest tests/test_arc_agi3_explorer.py -q`
Expected: PASS (new `TestTrace` + all existing explorer tests — the cascade is behaviourally identical).

- [ ] **Step 6: Commit**

```bash
cd /workspace/tgaer && git add src/tgaer/agents/arc_agi3_explorer.py tests/test_arc_agi3_explorer.py
git commit -m "feat(arc-agi3): per-step induction telemetry (self.trace) on explorer"
```

---

### Task 2: `RecordingAgent` wrapper

**Files:**
- Create: `src/tgaer/agents/arc_agi3_recorder.py`
- Test: `tests/test_arc_agi3_replay.py` (new `TestRecordingAgent` class)

**Interfaces:**
- Consumes: `tgaer.core.agent_base.Agent`, `tgaer.envs.arc_agi3.arc_agi3_api.ArcAction`.
- Produces: `RecordingAgent(inner: Agent, path: str | Path)`; writes one JSONL line per `act()`:
  `{"obs": <obs dict>, "action": {"id": int, "x": int | None, "y": int | None}}`. `reset()` delegates.

- [ ] **Step 1: Write the failing test**

Create `tests/test_arc_agi3_replay.py`:

```python
from __future__ import annotations

import json

from tgaer.agents.arc_agi3_recorder import RecordingAgent
from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction


class _Echo(Agent):
    """Returns a fixed action regardless of obs — lets us assert passthrough."""

    def __init__(self, action: ArcAction) -> None:
        self._action = action

    def act(self, observation):
        return self._action


class TestRecordingAgent:
    def test_records_obs_and_action_and_passes_through(self, tmp_path):
        path = tmp_path / "cap.jsonl"
        inner = _Echo(ArcAction(id=6, x=4, y=7))
        agent = RecordingAgent(inner, path)
        obs = {"frame": [[[0, 1], [2, 3]]], "available_actions": [6], "levels_completed": 0}
        out = agent.act(obs)
        assert out is inner._action  # unchanged passthrough
        rec = json.loads(path.read_text().splitlines()[0])
        assert rec["obs"] == obs
        assert rec["action"] == {"id": 6, "x": 4, "y": 7}

    def test_simple_action_records_null_coords(self, tmp_path):
        path = tmp_path / "cap.jsonl"
        agent = RecordingAgent(_Echo(ArcAction(id=2)), path)
        agent.act({"available_actions": [2]})
        rec = json.loads(path.read_text().splitlines()[0])
        assert rec["action"] == {"id": 2, "x": None, "y": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest tests/test_arc_agi3_replay.py::TestRecordingAgent -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tgaer.agents.arc_agi3_recorder'`.

- [ ] **Step 3: Write the implementation**

Create `src/tgaer/agents/arc_agi3_recorder.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction


class RecordingAgent(Agent):
    """Wraps an agent, appending every ``(obs, action)`` it processes to a JSONL.

    Capture happens at the agent boundary so the recorded obs is exactly what
    ``inner.act`` saw — including the env's post-transport ``terminal`` flag and
    respawn frame, which a transport-level recorder would miss."""

    def __init__(self, inner: Agent, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._path.write_text("")  # truncate any prior capture

    def act(self, observation: Any) -> Any:
        action = self._inner.act(observation)
        rec = {"obs": observation, "action": _action_dict(action)}
        with self._path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return action

    def reset(self) -> None:
        if hasattr(self._inner, "reset"):
            self._inner.reset()


def _action_dict(a: ArcAction) -> dict:
    return {"id": a.id, "x": a.x, "y": a.y}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/tgaer && ./.venv/bin/python -m pytest tests/test_arc_agi3_replay.py::TestRecordingAgent -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /workspace/tgaer && git add src/tgaer/agents/arc_agi3_recorder.py tests/test_arc_agi3_replay.py
git commit -m "feat(arc-agi3): RecordingAgent — capture (obs, action) at agent boundary"
```

---

### Task 3: Replay harness (`replay_traces` + CLI)

**Files:**
- Create: `experiments/replay_telemetry.py`
- Test: `tests/test_arc_agi3_replay.py` (new `TestReplay` class)

**Interfaces:**
- Consumes: `RecordingAgent` records (Task 2 format); `ExplorerArcAgi3Agent.trace` (Task 1).
- Produces:
  - `replay_traces(records: list[dict]) -> list[dict]` — fresh explorer, asserts each replayed
    action equals the captured one (`AssertionError` on divergence), returns the per-step `trace` list.
  - `summarise(traces: list[dict]) -> dict` — `{"avatar_pins_at": int | None, "max_lattice": int, "branches": dict[str, int]}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_arc_agi3_replay.py` (reuses `_board`/`_obs` from the explorer suite via import):

```python
from tests.test_arc_agi3_explorer import _board, _obs
from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
from tgaer.agents.arc_agi3_recorder import _action_dict
from experiments.replay_telemetry import replay_traces, summarise


def _records_from(obs_seq: list[dict]) -> list[dict]:
    """Drive a live explorer over obs_seq to produce capture-format records."""
    agent = ExplorerArcAgi3Agent()
    out = []
    for obs in obs_seq:
        action = agent.act(obs)
        out.append({"obs": obs, "action": _action_dict(action)})
    return out


class TestReplay:
    def test_faithful_replay_reproduces_actions(self):
        # A moving-avatar sequence: probe seeds the lattice, avatar pins.
        seq = [
            _obs(_board(avatar=(2, 2)), actions=(1, 2, 3, 4)),
            _obs(_board(avatar=(3, 2)), actions=(1, 2, 3, 4)),  # down → Δ=(1,0)
            _obs(_board(avatar=(3, 3)), actions=(1, 2, 3, 4)),  # right → Δ=(0,1)
        ]
        records = _records_from(seq)
        traces = replay_traces(records)  # no AssertionError == faithful
        assert len(traces) == len(records)

    def test_summary_reports_avatar_pin_and_branches(self):
        seq = [
            _obs(_board(avatar=(2, 2)), actions=(1, 2, 3, 4)),
            _obs(_board(avatar=(3, 2)), actions=(1, 2, 3, 4)),
            _obs(_board(avatar=(3, 3)), actions=(1, 2, 3, 4)),
            _obs(_board(avatar=(4, 3)), actions=(1, 2, 3, 4)),
        ]
        summary = summarise(replay_traces(_records_from(seq)))
        assert summary["avatar_pins_at"] is not None  # induction fires in-sim
        assert summary["max_lattice"] >= 1
        assert sum(summary["branches"].values()) == len(seq)

    def test_divergence_raises(self):
        records = _records_from([_obs(_board(avatar=(2, 2)), actions=(1, 2, 3, 4))])
        records[0]["action"]["id"] = 99  # corrupt → replayed action won't match
        import pytest

        with pytest.raises(AssertionError):
            replay_traces(records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tgaer && PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_arc_agi3_replay.py::TestReplay -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.replay_telemetry'`.

- [ ] **Step 3: Write the implementation**

Create `experiments/replay_telemetry.py`:

```python
"""Offline replay of a captured ls20 trajectory with induction telemetry.

Re-feeds the obs stream from a RecordingAgent capture to a fresh explorer,
asserts the replayed action matches the captured one (faithful reproduction),
and reports where induction dies: when the avatar pins, how large the move
lattice grows, and which act() branch fires each step. Free and re-runnable —
the only paid step is the one-game capture (experiments/capture_ls20.py).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
from tgaer.agents.arc_agi3_recorder import _action_dict


def replay_traces(records: list[dict]) -> list[dict]:
    agent = ExplorerArcAgi3Agent()
    traces: list[dict] = []
    for i, rec in enumerate(records):
        action = agent.act(rec["obs"])
        got, want = _action_dict(action), rec["action"]
        assert got == want, f"step {i}: replay {got} != captured {want} (non-determinism)"
        traces.append(agent.trace)
    return traces


def summarise(traces: list[dict]) -> dict:
    pins = next((t["step"] for t in traces if t["avatar"] is not None), None)
    return {
        "avatar_pins_at": pins,
        "max_lattice": max((t["lattice_size"] for t in traces), default=0),
        "branches": dict(Counter(t["branch"] for t in traces)),
    }


def main(capture: str, out: str) -> None:
    records = [json.loads(ln) for ln in Path(capture).read_text().splitlines() if ln]
    traces = replay_traces(records)
    Path(out).write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    print(f"[replay] {len(traces)} steps -> {out}", flush=True)
    print(f"[replay] {summarise(traces)}", flush=True)


if __name__ == "__main__":
    cap = sys.argv[1] if len(sys.argv) > 1 else "experiments/ls20_capture.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "experiments/ls20_telemetry.jsonl"
    main(cap, dst)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/tgaer && PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_arc_agi3_replay.py -q`
Expected: PASS (all `TestRecordingAgent` + `TestReplay`).

- [ ] **Step 5: Commit**

```bash
cd /workspace/tgaer && git add experiments/replay_telemetry.py tests/test_arc_agi3_replay.py
git commit -m "feat(arc-agi3): offline induction-telemetry replay harness"
```

---

### Task 4: Live capture script

**Files:**
- Create: `experiments/capture_ls20.py`

**Interfaces:**
- Consumes: `RecordingAgent` (Task 2), `ExplorerArcAgi3Agent`, `ArcAgi3Client`, `ArcAgi3Environment`,
  `evaluate_arc_agi3_agent`.
- Produces: `experiments/ls20_capture.jsonl` (one record per step of one live ls20 game).

> No unit test — this is a thin live-API entrypoint (the recorded format is covered by Task 2/3 tests). Verified by the one live run in the handoff.

- [ ] **Step 1: Write the script**

Create `experiments/capture_ls20.py`:

```python
"""Capture one live ls20 game's (obs, action) stream for offline telemetry replay.

The single paid step of the induction diagnostic: drives the real explorer over
one ls20 game through a RecordingAgent. No scorecard — only the trajectory matters.
Replay it for free with experiments/replay_telemetry.py.

    PYTHONPATH=src ./.venv/bin/python experiments/capture_ls20.py
"""

from __future__ import annotations

import os

import requests

from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
from tgaer.agents.arc_agi3_recorder import RecordingAgent
from tgaer.envs.arc_agi3.arc_agi3_client import BASE_URL, ArcAgi3Client
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

OUT = "/workspace/tgaer/experiments/ls20_capture.jsonl"
MAX_ACTIONS = int(os.environ.get("MAX_ACTIONS", "1000"))


def _ls20_game_id(key: str) -> str:
    games = requests.get(
        f"{BASE_URL}/api/games", headers={"X-API-Key": key}
    ).json()
    return next(g["game_id"] for g in games if g["game_id"].startswith("ls20"))


def main() -> None:
    key = os.environ["ARC_API_KEY"]
    gid = _ls20_game_id(key)
    print(f"[capture] ls20 = {gid}", flush=True)

    client = ArcAgi3Client(api_key=key)
    client.open_scorecard()  # API requires an open card to accept actions
    env = ArcAgi3Environment(
        client, gid, max_actions=MAX_ACTIONS, reset_on_game_over=True
    )
    agent = RecordingAgent(ExplorerArcAgi3Agent(), OUT)
    result = evaluate_arc_agi3_agent(agent, env, {"guards": [], "max_steps": MAX_ACTIONS})
    client.close_scorecard()
    print(f"[capture] DONE score={result.score} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run: `cd /workspace/tgaer && uv run ruff check experiments/capture_ls20.py experiments/replay_telemetry.py src/tgaer/agents/arc_agi3_recorder.py && uv run ruff format --check .`
Expected: no errors. (If format fails, run `uv run ruff format` and fold into the commit.)

- [ ] **Step 3: Commit**

```bash
cd /workspace/tgaer && git add experiments/capture_ls20.py
git commit -m "feat(arc-agi3): live ls20 capture script for telemetry replay"
```

---

### Task 5: Gitignore capture artefacts + simplify pass

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Ignore generated captures**

Add to `.gitignore`:

```
experiments/ls20_capture.jsonl
experiments/ls20_telemetry.jsonl
```

- [ ] **Step 2: Run `/simplify` on the full diff**

Per the per-commit review trigger, run the code-review/simplify pass over the working diff (Tasks 1-4) and fold any findings into the relevant commit (`git commit --amend` / `git rebase`), BEFORE pushing.

- [ ] **Step 3: Full test + lint gate**

Run: `cd /workspace/tgaer && PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_arc_agi3_replay.py tests/test_arc_agi3_explorer.py -q && uv run ruff check . && uv run ruff format --check .`
Expected: all PASS, no lint errors.

- [ ] **Step 4: Commit**

```bash
cd /workspace/tgaer && git add .gitignore && git commit -m "chore(arc-agi3): gitignore replay capture artefacts"
```

---

## Execution Handoff (live capture)

After Tasks 1-5 land and tests are green, run the one paid step, then the free replay:

```bash
cd /workspace/tgaer && set -a && . ./.env && set +a
PYTHONPATH=src ./.venv/bin/python experiments/capture_ls20.py
PYTHONPATH=src ./.venv/bin/python experiments/replay_telemetry.py
```

Read the `[replay] {...}` summary to localise the failing stage (avatar never pins → controllability; pins but small lattice → probe; always `choose` → affordance targeting), then write the findings up and decide the fix.

## Self-Review

- **Spec coverage:** RecordingAgent (Task 2) ✓, capture script (Task 4) ✓, `self.trace` (Task 1) ✓, replay `replay_traces`+summary (Task 3) ✓, recorder round-trip test ✓, replay-faithfulness-on-working-induction test (Task 3) ✓. All spec sections mapped.
- **Placeholders:** none — every code step is complete.
- **Type consistency:** `_action_dict` defined in Task 2, imported in Tasks 3; `agent.trace` keys defined in Task 1 match `summarise` access in Task 3; `replay_traces`/`summarise` signatures match between Task 3 impl and tests.
