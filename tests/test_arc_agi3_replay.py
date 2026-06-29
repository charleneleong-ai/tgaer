from __future__ import annotations

import json

import pytest

from experiments.replay_telemetry import replay_traces, summarise
from tests.test_arc_agi3_explorer import _Ls20Sim
from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
from tgaer.agents.arc_agi3_recorder import RecordingAgent, _action_dict
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
        obs = {
            "frame": [[[0, 1], [2, 3]]],
            "available_actions": [6],
            "levels_completed": 0,
        }
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

    def test_accumulates_multiple_acts(self, tmp_path):
        path = tmp_path / "cap.jsonl"
        agent = RecordingAgent(_Echo(ArcAction(id=1)), path)
        agent.act({"t": 0})
        agent.act({"t": 1})
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert [json.loads(ln)["obs"]["t"] for ln in lines] == [0, 1]

    def test_reset_delegates_to_inner(self, tmp_path):
        calls = []

        class _Resettable(_Echo):
            def reset(self):
                calls.append(1)

        agent = RecordingAgent(_Resettable(ArcAction(id=1)), tmp_path / "c.jsonl")
        agent.reset()
        assert calls == [1]


def _sim_records(budget: int = 120) -> list[dict]:
    """Capture the (obs, action) stream of a fresh explorer solving the closed-loop
    ls20 sim. The avatar responds to the explorer's *own* actions, so the induction
    is faithful — exactly the shape of a live ls20 recording, not a hand-scripted
    trajectory whose action→Δ mapping is incoherent."""
    sim = _Ls20Sim([(6, 6), (1, 6), (6, 1)])
    agent = ExplorerArcAgi3Agent()
    out = []
    for _ in range(budget):
        obs = sim.obs()
        action = agent.act(obs)
        out.append({"obs": obs, "action": _action_dict(action)})
        sim.step(action.id)
        if sim.levels == len(sim.doors):
            break
    return out


class TestReplay:
    def test_faithful_replay_reproduces_actions(self):
        records = _sim_records()
        traces = replay_traces(records)  # no AssertionError == faithful
        assert len(traces) == len(records)  # the full stream replayed

    def test_summary_reports_working_induction(self):
        # The harness must surface a *working* induction, so a flat avatar on real
        # ls20 reads as signal, not a harness artifact: the avatar pins, the lattice
        # grows past the first direction, and a directed branch fires.
        summary = summarise(replay_traces(records := _sim_records()))
        assert summary["avatar_pins_at"] is not None
        assert summary["max_lattice"] >= 2
        assert {"nav", "affordance"} & summary["branches"].keys()
        assert sum(summary["branches"].values()) == len(records)

    def test_divergence_raises(self):
        records = _sim_records()
        records[0]["action"]["id"] = 99  # corrupt → replayed action won't match
        with pytest.raises(AssertionError):
            replay_traces(records)
