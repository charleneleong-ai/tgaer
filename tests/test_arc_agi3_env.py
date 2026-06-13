from __future__ import annotations

import pytest

from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction, ArcFrame
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment


def _frame(state="NOT_FINISHED", levels=0, guid="g0", actions=(1, 2, 3, 4)):
    return ArcFrame(
        game_id="ls20-test",
        guid=guid,
        frame=[[[0] * 64 for _ in range(64)]],
        state=state,
        levels_completed=levels,
        win_levels=254,
        available_actions=list(actions),
    )


class _ScriptedTransport:
    """Returns a preset reset frame then a queue of step frames, recording calls."""

    def __init__(self, reset_frame, step_frames):
        self._reset_frame = reset_frame
        self._step_frames = list(step_frames)
        self.calls: list[tuple] = []

    def reset(self, game_id):
        self.calls.append(("reset", game_id))
        return self._reset_frame

    def act(self, game_id, guid, action_id, x=None, y=None):
        self.calls.append(("act", game_id, guid, action_id, x, y))
        return self._step_frames.pop(0)


@pytest.fixture
def env_factory():
    def make(reset_frame=None, step_frames=(), max_actions=80):
        transport = _ScriptedTransport(reset_frame or _frame(), list(step_frames))
        env = ArcAgi3Environment(
            transport, game_id="ls20-test", max_actions=max_actions
        )
        return env, transport

    return make


class TestReset:
    def test_returns_obs_with_frame_and_available_actions(self, env_factory):
        env, _ = env_factory(reset_frame=_frame(actions=(1, 2)))
        obs = env.reset()
        assert obs["available_actions"] == [1, 2]
        assert len(obs["frame"][0]) == 64  # one 64-row grid


class TestStep:
    def test_reward_is_levels_completed_delta(self, env_factory):
        env, _ = env_factory(
            reset_frame=_frame(levels=0),
            step_frames=[_frame(levels=17), _frame(levels=17)],
        )
        env.reset()
        assert env.step(ArcAction(1)).reward == 17.0
        assert env.step(ArcAction(1)).reward == 0.0

    @pytest.mark.parametrize("terminal", ["WIN", "GAME_OVER"])
    def test_done_on_terminal_state(self, env_factory, terminal):
        env, _ = env_factory(step_frames=[_frame(state=terminal)])
        env.reset()
        assert env.step(ArcAction(1)).done is True

    def test_not_done_while_unfinished(self, env_factory):
        env, _ = env_factory(step_frames=[_frame(state="NOT_FINISHED")])
        env.reset()
        assert env.step(ArcAction(1)).done is False

    def test_done_when_max_actions_reached(self, env_factory):
        env, _ = env_factory(step_frames=[_frame() for _ in range(3)], max_actions=3)
        env.reset()
        assert env.step(ArcAction(1)).done is False
        assert env.step(ArcAction(1)).done is False
        assert env.step(ArcAction(1)).done is True  # 3rd step hits the cap

    def test_complex_action_forwards_xy_to_transport(self, env_factory):
        env, transport = env_factory(step_frames=[_frame()])
        env.reset()
        env.step(ArcAction(6, x=12, y=34))
        assert transport.calls[-1] == ("act", "ls20-test", "g0", 6, 12, 34)

    def test_simple_action_forwards_no_coords(self, env_factory):
        env, transport = env_factory(step_frames=[_frame()])
        env.reset()
        env.step(ArcAction(3))
        assert transport.calls[-1] == ("act", "ls20-test", "g0", 3, None, None)


class TestTaskId:
    def test_returns_game_id(self, env_factory):
        env, _ = env_factory()
        assert env.task_id() == "ls20-test"
