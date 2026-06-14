from __future__ import annotations

import pytest

from tgaer.agents.arc_agi3_llm import ArcAgi3LLMAgent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction

AVAILABLE = [1, 2, 3, 6]


def _agent(reply: str, available=AVAILABLE) -> ArcAgi3LLMAgent:
    """Agent whose LLM call returns a canned string — no network."""
    a = ArcAgi3LLMAgent(seed=0)
    a._complete = lambda _prompt, _img=None: reply  # type: ignore[method-assign]
    return a


def _obs(available=AVAILABLE) -> dict:
    return {
        "frame": [[[0] * 4 for _ in range(4)]],
        "available_actions": available,
        "levels_completed": 0,
        "state": "NOT_FINISHED",
    }


class TestParsing:
    def test_plain_json_simple_action(self):
        act = _agent('{"id": 2}').act(_obs())
        assert act == ArcAction(id=2)

    def test_json_wrapped_in_prose_and_fences(self):
        reply = (
            'Sure — I think the move is:\n```json\n{"id": 3, "x": null, "y": null}\n```'
        )
        assert _agent(reply).act(_obs()) == ArcAction(id=3)

    def test_action6_carries_coords(self):
        act = _agent('{"id": 6, "x": 10, "y": 40}').act(_obs())
        assert (act.id, act.x, act.y) == (6, 10, 40)

    def test_action6_coords_clamped_to_grid(self):
        act = _agent('{"id": 6, "x": 999, "y": -5}').act(_obs())
        assert 0 <= act.x <= 63 and 0 <= act.y <= 63

    def test_action6_missing_coords_get_filled(self):
        act = _agent('{"id": 6}').act(_obs())
        assert act.id == 6 and act.x is not None and act.y is not None


class TestFallback:
    @pytest.mark.parametrize("reply", ["", "no json here", "{bad json", '{"id": 99}'])
    def test_unparseable_or_unavailable_falls_back_to_available(self, reply):
        # id 99 is not in AVAILABLE; garbage has no usable action — both must
        # still yield a *legal* action rather than crash the episode.
        act = _agent(reply).act(_obs())
        assert act.id in AVAILABLE

    def test_llm_exception_falls_back(self):
        a = ArcAgi3LLMAgent(seed=0)

        def _boom(_prompt, _img=None):
            raise RuntimeError("gemini down")

        a._complete = _boom  # type: ignore[method-assign]
        assert a.act(_obs()).id in AVAILABLE

    def test_empty_available_actions_defaults_to_action1(self):
        act = _agent('{"id": 5}', available=[]).act(_obs(available=[]))
        assert act.id == 1


class TestPromptContext:
    def test_prompt_lists_available_actions_and_grid(self):
        seen = {}
        a = ArcAgi3LLMAgent(seed=0)
        a._complete = lambda prompt, _img=None: seen.setdefault("p", prompt) and '{"id": 1}'  # type: ignore[method-assign]
        a.act(_obs())
        assert "2" in seen["p"] and "3" in seen["p"]  # available action ids surfaced

    def test_history_is_bounded(self):
        a = _agent('{"id": 1}')
        for _ in range(20):
            a.act(_obs())
        assert len(a._history) <= a._max_history


class TestPerception:
    def test_render_grid_prefixes_row_indices(self):
        a = ArcAgi3LLMAgent(seed=0)
        out = a._render_grid([[0, 1], [12, 15]])
        assert out.splitlines() == [" 0|01", " 1|cf"]  # 12->c, 15->f

    def test_prompt_states_board_dims(self):
        a = ArcAgi3LLMAgent(seed=0)
        prompt = a._build_prompt(_obs(), AVAILABLE)
        assert "4 rows x 4 cols" in prompt

    def test_diff_first_move(self):
        a = ArcAgi3LLMAgent(seed=0)
        assert "first move" in a._diff_feedback([[0]]).lower()

    def test_diff_no_change_flags_it(self):
        a = ArcAgi3LLMAgent(seed=0)
        a._prev_grid = [[0, 0], [0, 0]]
        a._history = ["3"]
        assert "NOTHING" in a._diff_feedback([[0, 0], [0, 0]])

    def test_diff_reports_changed_cells(self):
        a = ArcAgi3LLMAgent(seed=0)
        a._prev_grid = [[0, 0], [0, 0]]
        a._history = ["2"]
        msg = a._diff_feedback([[0, 5], [0, 0]])
        assert "1 cells" in msg and "(0,1)" in msg

    def test_prev_grid_tracked_across_steps(self):
        a = _agent('{"id": 1}')
        a.act(_obs())
        assert a._prev_grid == _obs()["frame"][-1]


class TestVision:
    def test_vision_passes_an_image_to_complete(self):
        seen = {}
        a = ArcAgi3LLMAgent(seed=0, vision=True)
        a._complete = lambda _p, img=None: seen.setdefault("img", img) or '{"id": 1}'  # type: ignore[method-assign]
        a.act(_obs())
        assert seen["img"] is not None and seen["img"].startswith("data:image/png;base64,")

    def test_no_vision_sends_no_image(self):
        seen = {}
        a = ArcAgi3LLMAgent(seed=0, vision=False)
        a._complete = lambda _p, img=None: seen.setdefault("img", img) or '{"id": 1}'  # type: ignore[method-assign]
        a.act(_obs())
        assert seen["img"] is None


class TestLocalEndpoint:
    def test_api_base_sets_default_local_key(self):
        a = ArcAgi3LLMAgent(model="openai/qwen", api_base="http://localhost:8000/v1")
        assert a._api_base == "http://localhost:8000/v1" and a._api_key == "local"

    def test_no_api_base_leaves_key_none(self):
        a = ArcAgi3LLMAgent(model="gemini/gemini-3.1-flash-lite")
        assert a._api_base is None and a._api_key is None
