from __future__ import annotations

from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction, ArcFrame
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

GAME_ID = "ls20-016295f7601e"


def _frame(state="NOT_FINISHED", levels=0, guid="g0") -> ArcFrame:
    return ArcFrame(
        game_id=GAME_ID,
        guid=guid,
        frame=[[[0] * 4 for _ in range(4)]],
        state=state,
        levels_completed=levels,
        win_levels=2,
        available_actions=[1, 2, 3, 4],
    )


class _ScriptedTransport:
    """reset() yields the first frame; act() pops the next scripted frame, or
    repeats the last one forever (a stuck game)."""

    def __init__(self, reset_frame: ArcFrame, act_frames: list[ArcFrame]):
        self._reset = reset_frame
        self._acts = list(act_frames)

    def reset(self, game_id: str) -> ArcFrame:
        return self._reset

    def act(self, game_id, guid, action_id, x=None, y=None) -> ArcFrame:
        if len(self._acts) > 1:
            return self._acts.pop(0)
        return self._acts[0]


class _FixedAgent(Agent):
    """Returns one fixed action every step and records what it saw."""

    def __init__(self, action: ArcAction):
        self._action = action
        self.seen: list = []

    def act(self, observation):
        self.seen.append(observation)
        return self._action


def _stuck_env(max_actions=80) -> tuple[ArcAgi3Environment, _FixedAgent]:
    transport = _ScriptedTransport(_frame(), [_frame()])
    env = ArcAgi3Environment(transport, GAME_ID, max_actions=max_actions)
    return env, _FixedAgent(ArcAction(id=1))


class TestRunToTerminal:
    def test_sums_level_deltas_and_surfaces_outcome(self):
        transport = _ScriptedTransport(
            _frame(levels=0),
            [_frame(levels=1), _frame(state="WIN", levels=2)],
        )
        env = ArcAgi3Environment(transport, GAME_ID)
        result = evaluate_arc_agi3_agent(_FixedAgent(ArcAction(id=2)), env)
        assert result.score == 2.0
        assert result.details["done"] is True
        assert result.details["steps"] == 2
        assert result.details["levels_completed"] == 2
        assert result.details["task_id"] == GAME_ID


class TestStepCaps:
    def test_env_action_cap_bounds_episode(self):
        env, agent = _stuck_env(max_actions=5)
        result = evaluate_arc_agi3_agent(agent, env)
        assert result.details["steps"] == 5
        assert result.details["done"] is True

    def test_cfg_max_steps_is_outer_bound(self):
        env, agent = _stuck_env(max_actions=80)
        result = evaluate_arc_agi3_agent(agent, env, {"max_steps": 4})
        assert result.details["steps"] == 4
        assert result.details["done"] is False


class TestGuardsWired:
    def test_guards_fire_and_inject_hints_on_stuck_agent(self):
        env, agent = _stuck_env(max_actions=6)
        result = evaluate_arc_agi3_agent(agent, env)
        assert result.details["guard_hints_fired"] > 0
        injected = [o for o in agent.seen if isinstance(o, dict) and "guard_hints" in o]
        assert injected, "expected guards to inject hints into the agent's observation"

    def test_guards_can_be_disabled(self):
        env, agent = _stuck_env(max_actions=6)
        result = evaluate_arc_agi3_agent(agent, env, {"guards": []})
        assert result.details["guard_hints_fired"] == 0
        assert all("guard_hints" not in o for o in agent.seen if isinstance(o, dict))
