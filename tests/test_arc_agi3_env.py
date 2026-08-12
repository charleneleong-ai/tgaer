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
    def make(reset_frame=None, step_frames=(), max_actions=80, **env_kwargs):
        transport = _ScriptedTransport(reset_frame or _frame(), list(step_frames))
        env = ArcAgi3Environment(
            transport, game_id="ls20-test", max_actions=max_actions, **env_kwargs
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


class TestResetOnGameOver:
    """With reset_on_game_over, a GAME_OVER respawns the game and surfaces a
    terminal flag instead of ending the episode — so the agent can learn the
    fatal transition and keep playing within budget."""

    def _env(self, env_factory, **kw):
        return env_factory(
            reset_frame=_frame(guid="fresh"),
            step_frames=[_frame(state="GAME_OVER"), _frame()],
            **kw,
        )

    def test_flag_respawns_and_continues(self, env_factory):
        env, transport = self._env(env_factory, reset_on_game_over=True)
        env.reset()
        tr = env.step(ArcAction(1))
        assert tr.done is False
        assert tr.state["terminal"] is True
        assert transport.calls.count(("reset", "ls20-test")) == 2  # initial + respawn

    def test_respawn_at_budget_cap_still_ends(self, env_factory):
        env, _ = self._env(env_factory, max_actions=1, reset_on_game_over=True)
        env.reset()
        # the death lands on the final allowed step → no respawn, episode ends
        assert env.step(ArcAction(1)).done is True


class TestTaskId:
    def test_returns_game_id(self, env_factory):
        env, _ = env_factory()
        assert env.task_id() == "ls20-test"
