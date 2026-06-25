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
