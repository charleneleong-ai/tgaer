from __future__ import annotations

from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction, ArcFrame
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent
from tgaer.envs.arc_agi3.rendering import ARC_PALETTE, grid_to_rgb
from tgaer.evaluation.wandb_logger import build_logger


def _frame(state="NOT_FINISHED", levels=0) -> ArcFrame:
    return ArcFrame(
        game_id="g",
        guid="x",
        frame=[[[0, 1], [2, 3]]],
        state=state,
        levels_completed=levels,
        win_levels=2,
        available_actions=[1, 2],
    )


class _WinAfter:
    """Transport that wins on the 3rd action — bounds the episode at 3 steps."""

    def __init__(self):
        self.n = 0

    def reset(self, game_id):
        return _frame()

    def act(self, game_id, guid, action_id, x=None, y=None):
        self.n += 1
        return _frame(state="WIN" if self.n >= 3 else "NOT_FINISHED", levels=self.n)


class _Agent:
    last_reasoning = "I will pick action 1 because it looks safe."
    last_reply = '{"id": 1}'

    def reset(self):
        pass

    def act(self, _obs):
        return ArcAction(id=1)


class _FakeLogger:
    def __init__(self):
        self.steps = []
        self.summary = None

    def log_step(self, **kw):
        self.steps.append(kw)

    def finish(self, summary):
        self.summary = summary


class TestBuildLogger:
    def test_disabled_or_missing_returns_none(self):
        assert build_logger(None) is None
        assert build_logger({"enabled": False}) is None
        assert build_logger({}) is None


class TestGridToRgb:
    def test_renders_last_grid_with_palette(self):
        rgb = grid_to_rgb([[[0, 1], [2, 3]]])
        assert rgb.shape == (2, 2, 3)
        assert tuple(rgb[0, 0]) == ARC_PALETTE[0]
        assert tuple(rgb[1, 1]) == ARC_PALETTE[3]

    def test_out_of_range_index_clamped(self):
        rgb = grid_to_rgb([[[99]]])
        assert tuple(rgb[0, 0]) == ARC_PALETTE[-1]

    def test_empty_frame_is_none(self):
        assert grid_to_rgb(None) is None
        assert grid_to_rgb([]) is None


class TestEvalLoggingHook:
    def test_logger_gets_a_step_per_action_and_one_finish(self):
        from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment

        env = ArcAgi3Environment(_WinAfter(), "g", max_actions=80)
        logger = _FakeLogger()
        result = evaluate_arc_agi3_agent(_Agent(), env, {"guards": []}, logger=logger)
        assert len(logger.steps) == result.details["steps"] == 3
        assert logger.summary is not None and logger.summary["score"] == result.score
        # frames + action ids are threaded through
        assert logger.steps[0]["action_id"] == 1
        assert logger.steps[-1]["frame"] is not None
        # the agent's reasoning trace is threaded through to the logger
        assert logger.steps[0]["reasoning"] == _Agent.last_reasoning

    def test_no_logger_is_a_noop(self):
        from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment

        env = ArcAgi3Environment(_WinAfter(), "g", max_actions=80)
        # Must not raise when logger is omitted.
        assert (
            evaluate_arc_agi3_agent(_Agent(), env, {"guards": []}).details["steps"] == 3
        )
