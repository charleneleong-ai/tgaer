from __future__ import annotations

from tgaer.core.agent_base import Agent
from tgaer.guards import (
    FutileActionGuard,
    GuardedAgent,
    RepeatedPlanGuard,
)


class _RecordingAgent(Agent):
    """Inner agent that returns a scripted action and records what it saw."""

    def __init__(self, action="noop"):
        self._action = action
        self.seen: list = []
        self.contexts: list = []

    def act(self, observation):
        self.seen.append(observation)
        return self._action

    def update_context(self, feedback):
        self.contexts.append(feedback)


def _run(agent: Agent, observations: list) -> list:
    return [agent.act(o) for o in observations]


class TestFutileActionGuard:
    def test_fires_when_window_of_observations_identical(self):
        guard = FutileActionGuard(window=3)
        for _ in range(2):
            guard.observe("same")
            assert guard.hint() is None
        guard.observe("same")
        assert "no observable change" in guard.hint()

    def test_changed_observation_breaks_the_streak(self):
        guard = FutileActionGuard(window=3)
        for _ in range(3):
            guard.observe("same")
        assert guard.hint() is not None
        guard.observe("different")
        assert guard.hint() is None

    def test_reset_clears_window(self):
        guard = FutileActionGuard(window=3)
        for _ in range(3):
            guard.observe("same")
        guard.reset()
        assert guard.hint() is None


class TestRepeatedPlanGuard:
    def test_fires_when_window_of_actions_identical(self):
        guard = RepeatedPlanGuard(window=4)
        for _ in range(3):
            guard.record_action("up")
            assert guard.hint() is None
        guard.record_action("up")
        assert "same action plan" in guard.hint()

    def test_changed_action_breaks_the_streak(self):
        guard = RepeatedPlanGuard(window=4)
        for _ in range(4):
            guard.record_action("up")
        assert guard.hint() is not None
        guard.record_action("down")
        assert guard.hint() is None

    def test_ignores_observations(self):
        guard = RepeatedPlanGuard(window=2)
        guard.observe("anything")
        assert guard.hint() is None


class TestGuardedAgent:
    def test_forwards_inner_action(self):
        inner = _RecordingAgent(action="jump")
        agent = GuardedAgent(inner, [FutileActionGuard(window=3)])
        assert agent.act({"frame": 1}) == "jump"

    def test_quiet_guards_pass_observation_through_unchanged(self):
        inner = _RecordingAgent()
        agent = GuardedAgent(inner, [FutileActionGuard(window=3)])
        obs = {"frame": 1}
        agent.act(obs)
        assert inner.seen[-1] == obs
        assert "guard_hints" not in inner.seen[-1]

    def test_futile_hint_injected_into_dict_observation(self):
        inner = _RecordingAgent()
        agent = GuardedAgent(inner, [FutileActionGuard(window=3)])
        for _ in range(3):
            agent.act({"frame": "stuck"})
        hints = inner.seen[-1]["guard_hints"]
        assert any("no observable change" in h for h in hints)

    def test_repeated_plan_fires_from_recorded_actions(self):
        inner = _RecordingAgent(action="up")
        agent = GuardedAgent(inner, [RepeatedPlanGuard(window=4)])
        # each step yields a different observation so futile would never fire,
        # but the action is constant -> repeated-plan must catch it.
        _run(agent, [{"frame": i} for i in range(5)])
        hints = inner.seen[-1]["guard_hints"]
        assert any("same action plan" in h for h in hints)

    def test_string_observation_gets_hint_appended(self):
        inner = _RecordingAgent()
        agent = GuardedAgent(inner, [FutileActionGuard(window=3)])
        for _ in range(3):
            agent.act("stuck")
        assert "no observable change" in inner.seen[-1]

    def test_update_context_delegates_to_inner(self):
        inner = _RecordingAgent()
        agent = GuardedAgent(inner, [])
        agent.update_context({"reward": 1.0})
        assert inner.contexts == [{"reward": 1.0}]

    def test_reset_clears_all_guard_state(self):
        inner = _RecordingAgent()
        futile = FutileActionGuard(window=3)
        agent = GuardedAgent(inner, [futile])
        for _ in range(3):
            agent.act("stuck")
        agent.reset()
        assert futile.hint() is None


class TestCanRun:
    def test_disabled_guard_is_dropped_at_construction(self):
        class _Off(FutileActionGuard):
            def can_run(self, env):
                return False

        inner = _RecordingAgent()
        agent = GuardedAgent(inner, [_Off(window=3)], env=object())
        for _ in range(3):
            agent.act("stuck")
        assert "guard_hints" not in _as_dict(inner.seen[-1])


def _as_dict(obs):
    return obs if isinstance(obs, dict) else {}
