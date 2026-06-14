from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tgaer.agents.arc_agi3_random import RandomArcAgi3Agent
from tgaer.cli import evaluate as cli
from tgaer.envs.arc_agi3.arc_agi3_api import ArcFrame
from tgaer.evaluation import dispatch

CONFIG_PATH = Path("configs/experiments/arc_agi3_guarded.yaml")
GAME_ID = yaml.safe_load(CONFIG_PATH.read_text())["env"]["game_id"]


def _frame(state="NOT_FINISHED", levels=0) -> ArcFrame:
    return ArcFrame(
        game_id=GAME_ID,
        guid="g0",
        frame=[[[0] * 4 for _ in range(4)]],
        state=state,
        levels_completed=levels,
        win_levels=2,
        available_actions=[1, 2, 6],
    )


class _ScriptedTransport:
    def reset(self, game_id):
        return _frame(levels=0)

    def act(self, game_id, guid, action_id, x=None, y=None):
        return _frame(state="WIN", levels=1)


class _ScoredTransport(_ScriptedTransport):
    """Scripted transport that also records scorecard open/close, like the
    live client — to verify the run is wrapped in a scorecard."""

    def __init__(self):
        self.events: list[str] = []

    def open_scorecard(self):
        self.events.append("open")

    def close_scorecard(self):
        self.events.append("close")


def _cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


class TestConfig:
    def test_shipped_config_dispatches_to_a_known_kind(self):
        cfg = _cfg()
        assert cfg["env"]["kind"] in dispatch.available_kinds()
        assert cfg["env"]["game_id"]


class TestDispatch:
    def test_arc_agi3_runs_with_injected_transport(self):
        result = dispatch.run_eval(_cfg(), transport=_ScriptedTransport())
        assert result.score == 1.0
        assert result.details["task_id"] == GAME_ID
        assert result.details["done"] is True

    def test_wraps_run_in_a_scorecard_when_supported(self):
        transport = _ScoredTransport()
        dispatch.run_eval(_cfg(), transport=transport)
        assert transport.events == ["open", "close"]

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown env.kind"):
            dispatch.run_eval({"env": {"kind": "nope"}})

    @pytest.mark.parametrize("kind", ["orak", "arc"])
    def test_registered_but_unwired_kinds_raise_not_implemented(self, kind):
        with pytest.raises(NotImplementedError, match="not wired yet"):
            dispatch.run_eval({"env": {"kind": kind}})

    def test_missing_api_key_surfaces_runtime_error(self, monkeypatch):
        monkeypatch.delenv("ARC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ARC_API_KEY"):
            dispatch.run_eval(_cfg())  # no injected transport -> live path


class TestBuildGuards:
    def test_defaults_to_both_guards(self):
        assert len(dispatch.build_guards({})) == 2

    def test_disabling_one_drops_it(self):
        guards = dispatch.build_guards(
            {"guards": {"futile_action": True, "repeated_plan": False}}
        )
        assert [type(g).__name__ for g in guards] == ["FutileActionGuard"]


class TestRandomAgent:
    def test_picks_available_action_and_is_seed_deterministic(self):
        obs = {"available_actions": [1, 2, 6]}
        a = RandomArcAgi3Agent(seed=7).act(obs)
        assert a == RandomArcAgi3Agent(seed=7).act(obs)
        assert a.id in (1, 2, 6)

    def test_complex_action_gets_in_range_coords(self):
        action = RandomArcAgi3Agent(seed=3).act({"available_actions": [6]})
        assert action.id == 6
        assert 0 <= action.x < 64 and 0 <= action.y < 64

    def test_empty_available_actions_falls_back(self):
        assert RandomArcAgi3Agent(seed=0).act({"available_actions": []}).id == 1


class TestCli:
    def test_dry_run_prints_config_without_running(self):
        result = CliRunner().invoke(cli.app, ["run", str(CONFIG_PATH), "--dry-run"])
        assert result.exit_code == 0
        assert "arc_agi3" in result.stdout

    def test_run_dispatches_and_prints_score(self, monkeypatch):
        monkeypatch.setattr(dispatch, "_live_arc_agi3_transport", _ScriptedTransport)
        result = CliRunner().invoke(cli.app, ["run", str(CONFIG_PATH)])
        assert result.exit_code == 0
        assert "score" in result.stdout

    def test_run_exits_nonzero_on_error(self, monkeypatch):
        monkeypatch.delenv("ARC_API_KEY", raising=False)
        result = CliRunner().invoke(cli.app, ["run", str(CONFIG_PATH)])
        assert result.exit_code == 1
