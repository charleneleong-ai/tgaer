"""Tests for the Kaggle submission agent.

Previously a throwaway script in /tmp; promoted here so the v32 fixes stay
regression-guarded. `agents.agent` is not importable from here, so MyAgent picks
up the local stub base and the dispatch tests can fake `arc_env` — the seam that
actually carries the action payload to the gateway.
"""
from __future__ import annotations

import io
import json
import random
import threading
from typing import Any

import pytest
from arcengine import FrameData, GameAction, GameState

from tgaer.agents import arc_agi3_kaggle as ma


class FakeLLM:
    """Stands in for llama_cpp.Llama, recording every create_chat_completion call."""

    def __init__(
        self,
        text: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self._responses = list(responses) if responses else []
        self.default_text = text
        self.default_tool_calls = tool_calls
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        msg: dict[str, Any] = {}
        if self.default_text is not None:
            msg["content"] = self.default_text
        if self.default_tool_calls:
            msg["tool_calls"] = self.default_tool_calls
        return {"choices": [{"message": msg}]}


def mk_frame(
    state: GameState = GameState.NOT_FINISHED,
    grid: list[list[int]] | None = None,
    available: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    levels: int = 0,
) -> FrameData:
    return FrameData(
        game_id="sk48",
        frame=[grid] if grid else [],
        state=state,
        levels_completed=levels,
        available_actions=list(available),
    )


def make_agent(
    text: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    responses: list[dict[str, Any]] | None = None,
) -> ma.MyAgent:
    agent = ma.MyAgent()
    agent._llm = FakeLLM(text=text, tool_calls=tool_calls, responses=responses)
    return agent


def tool_call(name: str, arguments: str = "{}") -> list[dict[str, Any]]:
    return [{"function": {"name": name, "arguments": arguments}}]


def tool_response(name: str, arguments: str = "{}") -> dict[str, Any]:
    return {"choices": [{"message": {"tool_calls": tool_call(name, arguments)}}]}


def text_response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


SMALL_GRID = [[0, 1], [1, 0]]


def user_text(call: dict[str, Any]) -> str:
    """The text of a call's last user message, multipart or not."""
    content = call["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if part["type"] == "text")


@pytest.fixture(autouse=True)
def isolated_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the environment and the cheap policies out of the model-path tests.

    ARC_LLM_BASE_URL would silently reroute inference, and probe/exploit answer
    turns before the model is ever asked. TestActionPolicy re-enables them.
    """
    monkeypatch.setattr(ma, "REMOTE_BACKEND", None)
    monkeypatch.setattr(ma, "PROBE_ACTIONS", False)
    monkeypatch.setattr(ma, "EXPLOIT_REPEATS", 0)


@pytest.fixture
def agent() -> ma.MyAgent:
    return make_agent("action(['RIGHT'])")


class TestResetHandling:
    """RESET is required before the first real move and after a game over."""

    @pytest.mark.parametrize(
        ("state", "grid"),
        [(GameState.NOT_PLAYED, []), (GameState.GAME_OVER, [[1, 0], [0, 1]])],
    )
    def test_unplayable_state_resets(self, state: GameState, grid: list[list[int]]) -> None:
        assert make_agent("action(['RIGHT'])").choose_action([], mk_frame(state, grid)) is (
            GameAction.RESET
        )

    def test_repeated_reset_falls_back_to_real_input(self) -> None:
        a = make_agent("action(['RIGHT'])")
        a._last_action_id = 0
        a._prev_levels = 0
        act = a.choose_action([], mk_frame(GameState.NOT_PLAYED, []))
        assert act is not GameAction.RESET


class TestRawTextParsing:
    """The raw-text fallback path extracts an action from free-form model output."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Here is my analysis. action(['DOWN'])", GameAction.ACTION2),
            ("Note: I see a wall.\naction(['LEFT'])", GameAction.ACTION3),
            ("action([RIGHT", GameAction.ACTION4),
            ("I will move UP now", GameAction.ACTION1),
        ],
    )
    def test_parses_action(self, text: str, expected: GameAction) -> None:
        assert make_agent(text).choose_action([], mk_frame(grid=SMALL_GRID)) is expected

    def test_explicit_reset_parses_when_available(self) -> None:
        act = make_agent("action(['RESET'])").choose_action(
            [], mk_frame(grid=SMALL_GRID, available=(0, 1, 2, 3, 4, 5, 6))
        )
        assert act is GameAction.RESET

    @pytest.mark.parametrize(
        "text",
        [
            "action([{'action': 'MOUSE', 'row': 3, 'col': 7}])",
            "action([{'action': 'MOUSE', 'col': 7, 'row': 3}])",
        ],
    )
    def test_mouse_coords_are_order_independent(self, text: str) -> None:
        a = make_agent(text)
        act = a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert act is GameAction.ACTION6
        assert a._pending_data == {"x": 7, "y": 3}

    @pytest.mark.parametrize("unusable", ["garbage not parseable", None])
    def test_fallback_is_varied_and_never_resets(self, unusable: str | None) -> None:
        """RESET mid-game restarts the level, so the fallback must never pick it."""
        acts = set()
        for seed in range(40):
            a = make_agent(unusable) if unusable else ma.MyAgent()
            if not unusable:
                a._llm = None
            a._rng = random.Random(seed)
            acts.add(a.choose_action([], mk_frame(grid=SMALL_GRID, available=(0, 1, 2))))
        assert GameAction.RESET not in acts
        assert len(acts) > 1


class TestToolCalling:
    """Structured function calling is the primary action path."""

    def test_simple_tool_call(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION4

    def test_tool_prompt_is_used_for_the_tool_call(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert "Call EXACTLY ONE function" in user_text(a._llm.calls[0])

    @pytest.mark.parametrize("bad_name", ["NOPE", "ACTION7"])
    def test_unusable_tool_call_falls_back_to_raw_text(self, bad_name: str) -> None:
        a = make_agent(responses=[tool_response(bad_name), text_response("action(['DOWN'])")])
        act = a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 2, 3, 4, 5, 6)))
        assert act is GameAction.ACTION2
        assert "EXACTLY ONE line in this format" in user_text(a._llm.calls[1])

    def test_tools_cover_exactly_the_available_actions(self, agent: ma.MyAgent) -> None:
        tools = agent._build_tools([1, 2, 3, 4, 5, 6])
        assert [t["function"]["name"] for t in tools] == [
            "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE",
        ]
        mouse = next(t for t in tools if t["function"]["name"] == "MOUSE")
        assert mouse["function"]["parameters"]["required"] == ["x", "y"], (
            "a click's coordinates must stay required, and the optional mechanic "
            "note must not join them: a required note would make every call that "
            "omits it unparseable, losing the coordinates"
        )

    def test_notes_are_offered_on_every_action_when_enabled(
        self, agent: ma.MyAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offered on all of them, since the note rides on whichever action the
        model picks; bounded in the schema because MAX_OUTPUT_TOKENS is shared
        with the thinking block and a long note truncates the call."""
        monkeypatch.setattr(ma, "MECHANIC_NOTES", True)
        for tool in agent._build_tools([1, 6]):
            note = tool["function"]["parameters"]["properties"]["note"]
            assert note["maxLength"] == ma.MAX_NOTE_CHARS

    def test_no_note_property_when_the_feature_is_off(
        self, agent: ma.MyAgent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ma, "MECHANIC_NOTES", False)
        for tool in agent._build_tools([1, 6]):
            assert "note" not in tool["function"]["parameters"]["properties"]


class TestTemplateQuirks:
    """Regressions for what the competition GPU exposed but a stub LLM cannot.

    llama-cpp's qwen3 template returned an empty `tool_calls` while the model
    had emitted a correct <tool_call> block in `content`, and prefixed replies
    with <think>. Together those disabled tool calling on every real turn.
    """

    QWEN_TOOL_TEXT = (
        '<think>\n\n<tool_call>\n{"name": "MOUSE", "arguments": {"x": 32, "y": 32}}\n</tool_call>'
    )

    def test_tool_call_left_as_text_is_recovered(self) -> None:
        a = make_agent(self.QWEN_TOOL_TEXT)
        act = a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert act is GameAction.ACTION6
        assert a._pending_data == {"x": 32, "y": 32}

    def test_recovery_costs_no_second_inference(self) -> None:
        a = make_agent(self.QWEN_TOOL_TEXT)
        a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert len(a._llm.calls) == 1

    def test_structured_tool_calls_still_win(self) -> None:
        """A populated tool_calls field must not be overridden by text parsing."""
        a = make_agent(self.QWEN_TOOL_TEXT, tool_calls=tool_call("RIGHT"))
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION4

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("<think>hmm</think>action(['UP'])", "action(['UP'])"),
            # An unclosed tag must keep what follows — that is where the real
            # reply lives in qwen3's output.
            ("<think>\n\naction(['UP'])", "action(['UP'])"),
            ("action(['UP'])", "action(['UP'])"),
        ],
    )
    def test_strip_thinking(self, raw: str, expected: str) -> None:
        assert ma.strip_thinking(raw) == expected

    def test_thinking_wrapped_action_still_parses(self) -> None:
        a = make_agent("<think>I should go right</think>\naction(['RIGHT'])")
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION4

    def test_reasoning_inside_think_cannot_trigger_the_keyword_fallback(self) -> None:
        """Without stripping, 'UP' in the reasoning would win over the real choice."""
        a = make_agent("<think>Maybe UP? No.</think>\naction(['DOWN'])")
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION2

    @pytest.mark.parametrize("system", ["TOOL_SYSTEM", "RAW_TEXT_SYSTEM"])
    def test_thinking_is_disabled_in_both_prompts(self, system: str) -> None:
        assert getattr(ma, system).endswith("/no_think")

    def test_malformed_tool_call_json_is_ignored(self) -> None:
        a = make_agent('<tool_call>\n{"name": "MOUSE", "argu\n</tool_call>')
        act = a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert act in set(GameAction) and act is not GameAction.RESET


class TestModelPool:
    """One instance per thread at a time, grown lazily, never more than max."""

    def test_reuses_one_instance_when_calls_do_not_overlap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loads = []
        monkeypatch.setattr(ma, "load_llama", lambda: loads.append(1) or object())
        pool = ma.ModelPool(4)
        for _ in range(5):
            with pool.acquire() as llm:
                assert llm is not None
        assert len(loads) == 1, "sequential use must not grow the pool"
        assert pool.size == 1

    def test_grows_to_max_under_concurrency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Holding all slots at once must create exactly max_size instances."""
        monkeypatch.setattr(ma, "load_llama", object)
        pool = ma.ModelPool(3)
        barrier, seen = threading.Barrier(3, timeout=10), []

        def worker() -> None:
            with pool.acquire() as llm:
                seen.append(id(llm))
                barrier.wait()  # every slot occupied simultaneously

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert pool.size == 3
        assert len(set(seen)) == 3, "concurrent holders must get distinct instances"

    def test_never_exceeds_max_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ma, "load_llama", object)
        pool = ma.ModelPool(2)

        def worker() -> None:
            for _ in range(20):
                with pool.acquire() as llm:
                    assert llm is not None

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert pool.size <= 2

    def test_no_instance_is_lent_to_two_threads_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ma, "load_llama", object)
        pool, held, clashes = ma.ModelPool(4), set(), []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(25):
                with pool.acquire() as llm:
                    with lock:
                        if id(llm) in held:
                            clashes.append(id(llm))
                        held.add(id(llm))
                    with lock:
                        held.discard(id(llm))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not clashes

    def test_failed_load_yields_none_and_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = []
        monkeypatch.setattr(ma, "load_llama", lambda: attempts.append(1) and None)
        pool = ma.ModelPool(3)
        for _ in range(4):
            with pool.acquire() as llm:
                assert llm is None
        assert len(attempts) == 1, "a second attempt would fail identically"

    def test_agent_falls_back_to_random_when_pool_has_no_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ma, "MODEL_POOL", ma.ModelPool(1))
        monkeypatch.setattr(ma, "load_llama", lambda: None)
        a = ma.MyAgent()
        a._rng = random.Random(0)
        act = a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert act is not GameAction.RESET


class TestContextBudget:
    """v32 regression guard: a board is ~1.2-2.4k tokens, so nothing may accumulate.

    Before v32 the chat history kept 10 messages each carrying a full board, which
    blew past n_ctx within a few steps; every call then raised and the agent
    silently degraded to random moves for the rest of the game.
    """

    def test_every_call_is_system_plus_one_user_message(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        for step in range(20):
            a.choose_action([], mk_frame(grid=[[step % 2, 1], [1, 0]]))
        assert a._llm.calls, "expected the model to be called"
        for call in a._llm.calls:
            assert [m["role"] for m in call["messages"]] == ["system", "user"]
            assert call["max_tokens"] == ma.MAX_OUTPUT_TOKENS

    def test_prompt_size_does_not_grow_across_turns(self) -> None:
        """Turn 20 must cost what turn 5 did on the same board.

        The pre-v32 history appended a whole board per turn (~4.8k chars), so a
        few chars of slack separates "constant" from the bug decisively. Held on
        one board so the comparison isolates accumulation from board content,
        and measured after the first turns, once the dead-action note settles.
        """
        a = make_agent(tool_calls=tool_call("RIGHT"))
        board = mk_frame(grid=[[0, 1], [1, 0]])
        for _ in range(40):
            a.choose_action([], board)
        sizes = [len(user_text(c)) for c in a._llm.calls[2:]]
        assert sizes, "the model must be consulted often enough to compare turns"
        # Bounded, not constant: the notes legitimately change as actions are
        # ruled out and a no-effect streak is flagged. What must never happen is
        # the board accumulating, which added ~4.8k chars per turn.
        assert max(sizes) < min(sizes) * 1.5, f"prompt grew across turns: {sizes}"


class FakeEnv:
    """Records exactly what reached arc_env.step — the real dispatch seam."""

    def __init__(self) -> None:
        self.steps: list[tuple[GameAction, dict[str, int] | None, Any]] = []

    def step(
        self, action: GameAction, data: dict[str, int] | None = None, reasoning: Any = None
    ) -> None:
        self.steps.append((action, data, reasoning))


class TestActionDispatch:
    """GameAction members are process-wide singletons, so payloads travel as arguments."""

    def test_click_coords_reach_step_as_data(self) -> None:
        a = make_agent(tool_calls=tool_call("MOUSE", '{"x": 11, "y": 22}'))
        a.arc_env = FakeEnv()
        act = a.choose_action([], mk_frame(grid=SMALL_GRID))
        a.do_action_request(act)
        action, data, _ = a.arc_env.steps[0]
        assert action is GameAction.ACTION6
        assert data == {"x": 11, "y": 22}

    def test_dispatch_never_mutates_the_shared_enum(self) -> None:
        """Nothing may be written to the singleton — that is what made games collide."""
        GameAction.ACTION6.set_data({"x": 0, "y": 0})
        a = make_agent(tool_calls=tool_call("MOUSE", '{"x": 11, "y": 22}'))
        a.arc_env = FakeEnv()
        a.do_action_request(a.choose_action([], mk_frame(grid=SMALL_GRID)))
        assert (GameAction.ACTION6.action_data.x, GameAction.ACTION6.action_data.y) == (0, 0)

    def test_concurrent_games_each_dispatch_their_own_coords(self) -> None:
        seen: dict[int, dict[str, int] | None] = {}
        barrier = threading.Barrier(8)

        def play(idx: int) -> None:
            a = make_agent(tool_calls=tool_call("MOUSE", f'{{"x": {idx}, "y": {idx + 1}}}'))
            a.arc_env = FakeEnv()
            act = a.choose_action([], mk_frame(grid=SMALL_GRID))
            barrier.wait()  # maximize the interleaving window before dispatch
            a.do_action_request(act)
            seen[idx] = a.arc_env.steps[0][1]

        threads = [threading.Thread(target=play, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == {i: {"x": i, "y": i + 1} for i in range(8)}

    def test_reasoning_is_normalized_to_a_dict(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        a.arc_env = FakeEnv()
        a.do_action_request(a.choose_action([], mk_frame(grid=SMALL_GRID)))
        _, data, reasoning = a.arc_env.steps[0]
        assert data is None
        assert isinstance(reasoning, dict)

    def test_coordinateless_click_gets_real_coords_not_a_stale_payload(self) -> None:
        """The keyword parser can name MOUSE without coords; (0,0) would be a lie."""
        a = make_agent("I should use the MOUSE here")
        a.arc_env = FakeEnv()
        act = a.choose_action([], mk_frame(grid=SMALL_GRID, available=(6,)))
        a.do_action_request(act)
        _, data, _ = a.arc_env.steps[0]
        assert data is not None
        assert 0 <= data["x"] < ma.GRID_SIZE and 0 <= data["y"] < ma.GRID_SIZE

    def test_simple_action_clears_stale_click_data(self) -> None:
        a = make_agent(
            responses=[tool_response("MOUSE", '{"x": 5, "y": 6}'), tool_response("RIGHT")]
        )
        a.choose_action([], mk_frame(grid=SMALL_GRID))
        a.choose_action([], mk_frame(grid=[[1, 1], [0, 0]]))
        assert a._pending_data is None


class TestRunBudget:
    """The framework caps actions per game; the agent must not lower that cap."""

    def test_max_actions_is_not_below_the_framework_default(self) -> None:
        assert ma.MyAgent.MAX_ACTIONS >= ma.Agent.MAX_ACTIONS

    @pytest.mark.parametrize(
        ("state", "deadline", "expected"),
        [
            (GameState.WIN, None, True),
            (GameState.NOT_FINISHED, -1.0, True),
            (GameState.NOT_FINISHED, None, False),
        ],
    )
    def test_is_done(
        self,
        agent: ma.MyAgent,
        monkeypatch: pytest.MonkeyPatch,
        state: GameState,
        deadline: float | None,
        expected: bool,
    ) -> None:
        if deadline is not None:
            monkeypatch.setattr(ma, "_RUN_DEADLINE", deadline)
        assert agent.is_done([], mk_frame(state, SMALL_GRID)) is expected

    def test_past_deadline_skips_inference_so_queued_games_drain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        monkeypatch.setattr(ma, "_RUN_DEADLINE", -1.0)
        a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert a._llm.calls == []


class TestGridEncoding:
    """Boards are cropped to their non-empty bounding box to cut prefill cost."""

    def test_crop_reports_origin_and_bbox(self, agent: ma.MyAgent) -> None:
        grid = [[0] * 8 for _ in range(8)]
        grid[2][3], grid[2][4] = 5, 10
        encoded, bbox = agent._encode_grid(grid)
        assert encoded.startswith("EJ")
        assert "; shown region = rows 2..2" in encoded
        assert bbox == (2, 3, 2, 4)

    def test_full_board_has_no_crop_marker(self, agent: ma.MyAgent) -> None:
        full = [[(r * 8 + c) % 16 for c in range(8)] for r in range(8)]
        encoded, bbox = agent._encode_grid(full)
        assert "[board is" not in encoded
        assert bbox is None

    @pytest.mark.parametrize("grid", [[], [[0, 0], [0, 0]]])
    def test_degenerate_grids_encode_without_bbox(
        self, agent: ma.MyAgent, grid: list[list[int]]
    ) -> None:
        encoded, bbox = agent._encode_grid(grid)
        assert bbox is None
        assert encoded in {"empty", "all zeros"}

    def test_change_detection_tracks_the_previous_board(self, agent: ma.MyAgent) -> None:
        agent.choose_action([], mk_frame(grid=SMALL_GRID))
        assert agent._grid_changed(SMALL_GRID) is False
        assert agent._grid_changed([[0, 1], [1, 9]]) is True


class TestPromptShape:
    """The prompt stays v26-style (the 0.17 build) plus the full-board click guide."""

    def test_raw_text_prompt_markers(self, agent: ma.MyAgent) -> None:
        prompt = agent._build_prompt(
            "AB\nCD", "  #0 A 2px", (2, 3, 5, 7),
            ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"], "NOT_FINISHED",
            "1 cell(s) changed: (0,0) .->A"
        )
        assert "EXACTLY ONE line in this format" in prompt
        assert "FULL-BOARD coordinates" in prompt
        assert "row=2 + r, col=3 + c" in prompt
        assert "Reasoning:" not in prompt

    def test_tool_prompt_markers(self, agent: ma.MyAgent) -> None:
        prompt = agent._build_prompt(
            "AB\nCD", "  #0 A 2px", None, ["UP", "DOWN", "MOUSE"], "NOT_FINISHED",
            "no change", tool_mode=True
        )
        assert "Call EXACTLY ONE function" in prompt
        assert "Reasoning:" not in prompt
        assert "Recent actions" not in prompt

    def test_no_click_guide_when_whole_board_is_shown(self, agent: ma.MyAgent) -> None:
        prompt = agent._build_prompt("AB", "  #0 A 1px", None, ["UP", "MOUSE"],
                                    "NOT_FINISHED", "n/a")
        assert "FULL-BOARD coordinates" not in prompt


class TestHTTPChatBackend:
    """The vLLM/OpenAI client must send what Qwen3.6 needs to answer at all."""

    def sent_payload(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        class FakeResponse:
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"choices":[{"message":{"content":"action([\'UP\'])"}}]}'

        def fake_urlopen(request: Any, timeout: float = 0) -> FakeResponse:
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            return FakeResponse()

        monkeypatch.setattr(ma, "urlopen", fake_urlopen)
        backend = ma.HTTPChatBackend("http://127.0.0.1:8000/v1/", "arc-agent")
        backend.create_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "UP"}}],
            tool_choice="required",
        )
        return captured

    def test_disables_thinking_via_chat_template_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without this Qwen3.6 spends the whole token budget inside <think>."""
        body = self.sent_payload(monkeypatch)["body"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_forwards_tools_and_tool_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = self.sent_payload(monkeypatch)["body"]
        assert body["tool_choice"] == "required"
        assert body["tools"][0]["function"]["name"] == "UP"

    def test_posts_to_the_chat_completions_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self.sent_payload(monkeypatch)["url"] == (
            "http://127.0.0.1:8000/v1/chat/completions"
        )



class TestBackendSelection:
    """Which backend `_model()` picks — the branch the real submission runs on.

    Every other test injects `_llm` or patches the pool, so without this the
    production path (no injected model, a vLLM server configured) has no
    coverage at all.
    """

    def test_remote_backend_is_used_and_the_pool_is_never_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLLM(tool_calls=tool_call("RIGHT"))
        monkeypatch.setattr(ma, "REMOTE_BACKEND", fake)
        monkeypatch.setattr(
            ma, "load_llama", lambda: pytest.fail("pool loaded despite a configured server")
        )
        a = ma.MyAgent()  # no _llm injected: the real rerun shape
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION4
        assert len(fake.calls) == 1

    def test_injected_model_wins_over_a_configured_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score_local injects per game; it must not reach a server it didn't start."""
        server = FakeLLM(tool_calls=tool_call("LEFT"))
        monkeypatch.setattr(ma, "REMOTE_BACKEND", server)
        a = make_agent(tool_calls=tool_call("RIGHT"))
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION4
        assert server.calls == []

    def test_pool_is_used_when_no_server_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLLM(tool_calls=tool_call("DOWN"))
        monkeypatch.setattr(ma, "MODEL_POOL", ma.ModelPool(1))
        monkeypatch.setattr(ma, "load_llama", lambda: fake)
        a = ma.MyAgent()
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION2


class TestLevelMemory:
    """Transitions reduced to prompt text, with chrome separated from the game.

    Scoring squares the action count, so the costly mistake is repeating a move
    that achieves nothing — and a moving timer makes a useless move look useful.
    """

    def board(self, *rows: str) -> ma.Grid:
        return tuple(tuple(int(c) for c in row) for row in rows)

    def test_first_look_has_no_transition(self) -> None:
        assert "first look" in ma.LevelMemory().describe_last()

    def test_no_change_is_reported_as_achieving_nothing(self) -> None:
        mem, board = ma.LevelMemory(), self.board("01", "10")
        mem.record("UP", board, board)
        assert "NOTHING changed" in mem.describe_last()
        assert mem.dead_actions == {"UP"}

    def test_small_change_names_cells_and_symbols(self) -> None:
        mem = ma.LevelMemory()
        mem.record("UP", self.board("00", "00"), self.board("00", "03"))
        summary = mem.describe_last()
        assert "1 cell(s) changed" in summary
        assert "(1,1) .->C" in summary

    def test_large_change_reports_extent_not_every_cell(self) -> None:
        mem = ma.LevelMemory()
        before = tuple(tuple(0 for _ in range(20)) for _ in range(20))
        after = tuple(tuple(5 for _ in range(20)) for _ in range(20))
        mem.record("SPACE", before, after)
        summary = mem.describe_last()
        assert "400 cells changed" in summary
        assert "rows 0..19" in summary
        assert summary.count("->") <= ma.LevelMemory.MAX_LISTED_CELLS

    def test_a_ticking_timer_is_classified_as_chrome_not_progress(self) -> None:
        """A cell that moves under every action must not count as the move working."""
        mem = ma.LevelMemory()
        for step, action in enumerate(["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]):
            before = self.board(f"0{step % 10}", "00")
            after = self.board(f"0{(step + 1) % 10}", "00")  # only the timer cell moves
            mem.record(action, before, after)
        assert (0, 1) in mem.hud_cells
        assert "NOTHING changed" in mem.describe_last()
        assert mem.dead_actions, "an action that only ticks the timer is dead"

    def test_a_real_move_alongside_a_timer_still_counts(self) -> None:
        mem = ma.LevelMemory()
        for step, action in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
            mem.record(action, self.board(f"0{step}", "00"), self.board(f"0{step + 1}", "00"))
        # Same ticking timer, but this time a game cell moves too.
        mem.record("UP", self.board("04", "00"), self.board("05", "07"))
        summary = mem.describe_last()
        assert "(1,1) .->G" in summary
        assert "NOTHING" not in summary
        assert mem.dead_actions == set(), "a real effect clears the dead list"

    def test_one_off_changes_are_never_called_chrome(self) -> None:
        """Only cells moving under many different actions are chrome."""
        mem = ma.LevelMemory()
        mem.record("UP", self.board("00", "00"), self.board("01", "00"))
        mem.record("DOWN", self.board("01", "00"), self.board("01", "00"))
        mem.record("LEFT", self.board("01", "00"), self.board("01", "00"))
        assert mem.hud_cells == set()

    def test_reset_forgets_the_previous_level(self) -> None:
        mem, board = ma.LevelMemory(), self.board("01", "10")
        mem.record("UP", board, board)
        mem.reset()
        assert mem.dead_actions == set()
        assert mem.transitions == 0
        assert "first look" in mem.describe_last()

    def test_notes_list_dead_actions_and_chrome(self) -> None:
        mem = ma.LevelMemory()
        for step, action in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
            mem.record(action, self.board(f"0{step}", "00"), self.board(f"0{step + 1}", "00"))
        notes = mem.prompt_notes()
        assert "change NOTHING" in notes
        assert "timer or counter" in notes

    def test_empty_notes_add_nothing_to_the_prompt(self) -> None:
        assert ma.LevelMemory().prompt_notes() == ""


class TestMemoryInAgent:
    """The agent feeds real transitions in and the prompt carries them out."""

    def test_effect_reaches_the_prompt(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        a.choose_action([], mk_frame(grid=[[0, 0], [0, 0]]))
        a.choose_action([], mk_frame(grid=[[0, 0], [0, 3]]))
        assert "Effect of that action: 1 cell(s) changed" in user_text(a._llm.calls[-1])

    def test_repeated_useless_action_is_flagged_in_the_prompt(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        still = mk_frame(grid=[[0, 1], [1, 0]])
        for _ in range(3):
            a.choose_action([], still)
        prompt = user_text(a._llm.calls[-1])
        assert "change NOTHING" in prompt
        assert "RIGHT" in prompt

    def test_memory_resets_on_a_new_level(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        still = mk_frame(grid=[[0, 1], [1, 0]])
        for _ in range(3):
            a.choose_action([], still)
        assert a.memory.dead_actions
        a.choose_action([], mk_frame(grid=[[0, 1], [1, 0]], levels=1))
        assert a.memory.dead_actions == set()


class TestSegmentation:
    """Objects, not digits — the view the Duck harness makes primary."""

    def test_finds_separate_objects_of_the_same_colour(self) -> None:
        seg = ma.segment(((1, 0, 1), (0, 0, 0), (1, 0, 1)))
        assert len(seg["nodes"]) == 4
        assert all(n["pixels"] == 1 for n in seg["nodes"])

    def test_four_connected_cells_are_one_object(self) -> None:
        seg = ma.segment(((1, 1, 0), (1, 0, 0), (0, 0, 0)))
        assert len(seg["nodes"]) == 1
        assert seg["nodes"][0]["pixels"] == 3

    def test_hash_is_position_invariant_so_objects_track_across_frames(self) -> None:
        """The same shape moved one cell must be recognisable as the same object."""
        left = ma.segment(((0, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 0)))
        right = ma.segment(((0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 0, 0)))
        assert left["nodes"][0]["hash"] == right["nodes"][0]["hash"]

    def test_different_shapes_hash_differently(self) -> None:
        pair = ma.segment(((1, 1),))["nodes"][0]["hash"]
        single = ma.segment(((1, 0),))["nodes"][0]["hash"]
        assert pair != single

    def test_enclosed_object_is_reported_as_a_child(self) -> None:
        board = ((1, 1, 1, 1), (1, 0, 2, 1), (1, 1, 1, 1))
        seg = ma.segment(board)
        ring = next(n for n in seg["nodes"] if n["colour"] == "A")
        inner = next(n for n in seg["nodes"] if n["colour"] == "B")
        assert inner["id"] in ring["children"]

    def test_touching_objects_are_adjacent(self) -> None:
        seg = ma.segment(((1, 2),))
        assert seg["adjacency"] == [[0, 1]]

    def test_background_is_not_an_object(self) -> None:
        assert ma.segment(((0, 0), (0, 0)))["nodes"] == []

    def test_description_lists_largest_first(self) -> None:
        text = ma.describe_segmentation(ma.segment(((1, 1, 1), (0, 0, 0), (2, 0, 0))))
        assert text.index("3px") < text.index("1px")


class TestPythonSandbox:
    """Model-written inspection code, run against the board."""

    def test_prints_come_back(self) -> None:
        assert run_out("print(len(grid))", grid=((1, 2), (3, 4))) == "2"

    def test_can_query_the_segmentation(self) -> None:
        out = run_out(
            "print(sorted(n['pixels'] for n in objects['nodes']))",
            grid=((1, 1, 0), (0, 0, 2)),
        )
        assert out == "[1, 2]"

    def test_errors_return_as_text_instead_of_losing_the_turn(self) -> None:
        assert "NameError" in run_out("print(nope)", grid=((1,),))

    def test_silence_is_explained(self) -> None:
        assert "no output" in run_out("x = 1", grid=((1,),))

    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "__import__('os').system('echo hi')",
            "open('/etc/passwd').read()",
        ],
    )
    def test_imports_and_io_are_unavailable(self, code: str) -> None:
        """Model-written code runs inside the submission kernel; keep it inert."""
        out = run_out(code, grid=((1,),))
        assert "Error" in out or "error" in out

    def test_output_is_truncated(self) -> None:
        out = run_out("print('x' * 5000)", grid=((1,),))
        assert "truncated" in out
        assert len(out) < 2000


def run_out(code: str, grid: ma.Grid) -> str:
    return ma.run_python(code, {"grid": grid, "objects": ma.segment(grid)})


class TestReplLoop:
    """The model may inspect the board before committing to an action.

    REPL_STEPS defaults to 0 (it cost throughput without gaining levels), so
    these enable it explicitly rather than depending on the default.
    """

    @pytest.fixture(autouse=True)
    def enable_repl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ma, "REPL_STEPS", 2)

    def python_call(self, code: str) -> list[dict[str, Any]]:
        return [{"function": {"name": "python", "arguments": json.dumps({"code": code})}}]

    def test_inspection_output_is_fed_back_then_an_action_is_taken(self) -> None:
        a = make_agent(responses=[
            {"choices": [{"message": {"tool_calls": self.python_call("print(len(grid))")}}]},
            tool_response("RIGHT"),
        ])
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION4
        second_call = a._llm.calls[1]["messages"]
        assert any("python output:" in m["content"] for m in second_call)
        assert any("2" in m["content"] for m in second_call if "python output" in m["content"])

    def test_the_last_turn_cannot_call_python(self) -> None:
        """Otherwise a model that only inspects would never act."""
        a = make_agent(responses=[
            {"choices": [{"message": {"tool_calls": self.python_call("print(1)")}}]},
            {"choices": [{"message": {"tool_calls": self.python_call("print(2)")}}]},
            tool_response("DOWN"),
        ])
        a.choose_action([], mk_frame(grid=SMALL_GRID))
        offered = [
            [t["function"]["name"] for t in call["tools"]] for call in a._llm.calls
        ]
        assert "python" in offered[0]
        assert "python" not in offered[-1]

    def test_acting_immediately_costs_one_call(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert len(a._llm.calls) == 1


class TestBoardImage:
    """The served model has a vision tower; the board goes as a picture too."""

    def test_render_produces_a_png(self) -> None:
        png = ma.render_board_png(((0, 1), (2, 3)), cell_px=4)
        assert png is not None
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_image_scales_with_the_board(self) -> None:
        pytest.importorskip("PIL")
        from PIL import Image
        png = ma.render_board_png(tuple((0,) * 10 for _ in range(6)), cell_px=5)
        assert Image.open(io.BytesIO(png)).size == (50, 30)

    def test_colours_differ_per_value_so_objects_are_separable(self) -> None:
        pytest.importorskip("PIL")
        from PIL import Image
        img = Image.open(io.BytesIO(ma.render_board_png(((1, 2),), cell_px=4)))
        assert img.getpixel((1, 1)) != img.getpixel((5, 1))

    def test_turn_message_carries_text_and_image(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        a.choose_action([], mk_frame(grid=SMALL_GRID))
        content = a._llm.calls[0]["messages"][-1]["content"]
        assert isinstance(content, list)
        kinds = [part["type"] for part in content]
        assert kinds == ["text", "image_url"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert a.stats["image_sent"] == 1

    def test_disabling_images_sends_plain_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ma, "SEND_IMAGE", False)
        a = make_agent(tool_calls=tool_call("RIGHT"))
        a.choose_action([], mk_frame(grid=SMALL_GRID))
        assert isinstance(a._llm.calls[0]["messages"][-1]["content"], str)

    def test_missing_pil_degrades_to_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A render failure must cost the turn nothing."""
        monkeypatch.setattr(ma, "render_board_png", lambda *a, **k: None)
        a = make_agent(tool_calls=tool_call("RIGHT"))
        assert a.choose_action([], mk_frame(grid=SMALL_GRID)) is GameAction.ACTION4
        assert isinstance(a._llm.calls[0]["messages"][-1]["content"], str)
        assert a.stats["image_unavailable"] == 1


class TestActionPolicy:
    """Probe every action once, then repeat what works — both without inference.

    A trace showed the agent taking 41 MOUSE clicks in a 4x4 patch while the
    movement keys sat untried, and never repeating anything that did work.
    """

    @pytest.fixture(autouse=True)
    def enable_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ma, "PROBE_ACTIONS", True)
        monkeypatch.setattr(ma, "EXPLOIT_REPEATS", 3)

    def test_probes_each_action_before_asking_the_model(self) -> None:
        a = make_agent(tool_calls=tool_call("MOUSE", '{"x": 1, "y": 1}'))
        seen = [
            a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 2, 3)))
            for _ in range(3)
        ]
        assert set(seen) == {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3}
        assert a._llm.calls == [], "probing must not cost an inference"

    def test_model_is_consulted_once_everything_is_probed(self) -> None:
        """Cheap policies may hold a run of turns, never the whole game."""
        a = make_agent(tool_calls=tool_call("UP"))
        for _ in range(ma.MAX_POLICY_STREAK + 6):
            a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 2, 3)))
        assert a._llm.calls, "the model must be asked once probing is done"

    def test_an_action_that_works_is_repeated(self) -> None:
        """8 of 25 public games are won by one action repeated many times."""
        a = make_agent(tool_calls=tool_call("UP"))
        a._prev_levels = 0  # past the level-reset branch, which clears the model
        a.actions.tried = {"UP": 1, "DOWN": 1}  # probing already done
        a.actions.worked = {"UP": 1}
        a.actions.last_effective = "UP"
        a._exploit_left = 3
        before = len(a._llm.calls)
        acts = [
            a.choose_action([], mk_frame(grid=[[i % 2, 1], [1, 0]], available=(1, 2)))
            for i in range(3)
        ]
        assert acts == [GameAction.ACTION1] * 3
        assert len(a._llm.calls) == before, "repeats must not cost inference"

    def test_exploitation_stops_when_the_action_stops_working(self) -> None:
        a = make_agent(tool_calls=tool_call("DOWN"))
        a._prev_levels = 0
        a.actions.tried = {"UP": 1, "DOWN": 1}
        a.actions.last_effective = "UP"
        a._exploit_left = 5
        still = mk_frame(grid=SMALL_GRID, available=(1, 2))
        for _ in range(4):
            a.choose_action([], still)
        assert a.actions.last_effective != "UP", "a dead action must stop being exploited"

    def test_repeats_are_capped_when_the_action_keeps_working(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every arrow changes a movement board, so re-arming on success never
        drained the counter and the agent held one direction until the move
        budget died — 53% of every action in the v59 run, zero levels."""
        monkeypatch.setattr(ma, "MAX_POLICY_STREAK", 99)  # or the streak cap passes this
        a = make_agent(tool_calls=tool_call("UP"))
        a._prev_levels = 0
        a.actions.tried = {"UP": 1, "DOWN": 1}
        a.actions.worked = {"UP": 1}
        a.actions.last_effective = "UP"
        a._exploited = {"UP"}
        a._exploit_left = ma.EXPLOIT_REPEATS
        for i in range(ma.EXPLOIT_REPEATS + 3):
            a.choose_action([], mk_frame(grid=[[i % 2, 1], [1, 0]], available=(1, 2)))
        assert a.stats["exploit"] <= ma.EXPLOIT_REPEATS
        assert not a.stats["policy_yield"], "the cap must be what stops it, not the streak"

    def test_a_newly_productive_click_cell_is_exploited_too(self) -> None:
        """Every click shares the family MOUSE, so keying the burst on the family
        would arm for the first cell that ever worked and no other — a later
        productive cell found by click_search would never be repeated."""
        a = make_agent(tool_calls=tool_call("MOUSE", '{"x": 9, "y": 9}'))
        a._prev_levels = 0
        a.actions.tried = {"UP": 1, "MOUSE": 1}
        a.actions.last_effective = "MOUSE"

        def productive_click(cell: tuple[int, int], board: int) -> None:
            """A click at `cell` that just changed the board, fed back as history."""
            a._last_action_name = f"MOUSE@{cell[0]},{cell[1]}"
            a._last_click = cell
            a._last_grid = ((1 - board, 1 - board), (1 - board, 1 - board))
            a.choose_action([], mk_frame(grid=[[board, board], [board, board]],
                                         available=(1, 6)))

        productive_click((1, 1), 1)      # the first cell that ever worked
        a._exploit_left = 0              # its burst is spent
        productive_click((9, 9), 0)      # a different cell, found later, also works
        assert a._exploit_left, "a newly productive cell must arm its own burst"

    def test_summary_reports_what_each_action_did(self) -> None:
        model = ma.ActionModel()
        model.record("UP", True)
        model.record("UP", False)
        model.record("SPACE", False)
        summary = model.summary(["UP", "SPACE", "LEFT"])
        assert "UP: worked 1/2" in summary
        assert "SPACE: worked 0/1" in summary
        assert "LEFT" not in summary, "untried actions have nothing to report"

    def test_reset_forgets_the_previous_level(self) -> None:
        model = ma.ActionModel()
        model.record("UP", True)
        model.reset()
        assert model.tried == {} and model.last_effective is None

    def test_click_only_games_are_left_to_the_model(self) -> None:
        """The coordinate is the decision, so probe/exploit have nothing to add.

        tn36 offers only MOUSE; the policy took 201 actions in 1.2s at random
        positions and never asked the model once.
        """
        a = make_agent(tool_calls=tool_call("MOUSE", '{"x": 7, "y": 9}'))
        act = a.choose_action([], mk_frame(grid=SMALL_GRID, available=(6,)))
        assert act is GameAction.ACTION6
        assert a._llm.calls, "a click-only game must consult the model"
        assert a._pending_data == {"x": 7, "y": 9}

    def test_exploiting_a_click_repeats_the_same_target(self) -> None:
        a = make_agent(tool_calls=tool_call("MOUSE", '{"x": 99, "y": 99}'))
        a._prev_levels = 0  # past the level-reset branch
        a.actions.tried = {"UP": 1, "MOUSE": 1}  # probing already done
        a.actions.last_effective = "MOUSE"
        a._exploit_left = 2
        a._last_click = (11, 22)
        act = a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 6)))
        assert act is GameAction.ACTION6
        assert a._pending_data == {"x": 11, "y": 22}, "a repeat must reuse the target"
        assert a._llm.calls == [], "a repeat must not cost an inference"

    def test_probing_skips_mouse(self) -> None:
        a = make_agent(tool_calls=tool_call("MOUSE", '{"x": 1, "y": 1}'))
        seen = [
            a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 2, 6)))
            for _ in range(2)
        ]
        assert set(seen) == {GameAction.ACTION1, GameAction.ACTION2}
        assert a._llm.calls == [], "probing simple actions costs no inference"


class TestClickSearch:
    """Clicking objects, not pixels: 4096 cells collapse to tens of candidates."""

    def board(self) -> ma.Grid:
        # Two separate objects, different sizes.
        rows = [[0] * 8 for _ in range(8)]
        for c in range(4):
            rows[1][c] = 1
        rows[5][6] = 2
        return tuple(tuple(r) for r in rows)

    def test_targets_skip_background_and_prefer_larger_pieces(self) -> None:
        """Clicking sk48's six biggest objects changed nothing; they are scenery."""
        rows = [[3] * 8 for _ in range(8)]      # a big background region
        for c in range(4):
            rows[1][c] = 1                       # a 4-cell piece
        rows[5][6] = 2                           # a single pixel
        seg = ma.segment(tuple(tuple(r) for r in rows))
        targets = ma.ClickSearch.targets(seg, board_cells=64)
        assert len(targets) == 2, "the background region must not be a target"
        assert targets[0][0] == 1, "the 4-cell piece should come before the single pixel"

    def test_walks_objects_without_repeating(self) -> None:
        seg = ma.segment(self.board())
        search = ma.ClickSearch()
        first = search.next_target(seg)
        search.record(first, changed_gameplay=False)
        second = search.next_target(seg)
        assert second is not None and second != first

    def test_reports_exhaustion_once_every_object_is_tried(self) -> None:
        seg = ma.segment(self.board())
        search = ma.ClickSearch()
        while (cell := search.next_target(seg)) is not None:
            search.record(cell, changed_gameplay=False)
        assert search.exhausted

    def test_a_click_that_works_reopens_the_candidates(self) -> None:
        """The board moved, so every object may behave differently now."""
        seg = ma.segment(self.board())
        search = ma.ClickSearch()
        first = search.next_target(seg)
        search.record(first, changed_gameplay=False)
        second = search.next_target(seg)
        search.record(second, changed_gameplay=True)
        assert search.next_target(seg) == first, "candidates reopen after a real change"
        assert not search.exhausted

    def test_agent_walks_objects_once_it_is_stuck(self) -> None:
        a = make_agent(tool_calls=tool_call("MOUSE", '{"x": 0, "y": 0}'))
        a._prev_levels = 0
        still = mk_frame(grid=[list(r) for r in self.board()], available=(6,))
        seen = set()
        for _ in range(6):
            act = a.choose_action([], still)
            assert act is GameAction.ACTION6
            seen.add((a._pending_data["x"], a._pending_data["y"]))
        assert a.stats["click_search"] > 0, "a stuck agent must walk the objects"
        assert len(seen) > 1, "it must not click the same pixel every turn"

    def test_policies_cannot_take_over_the_whole_game(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cn04 ran 201 actions in 1.0s without one inference; tn36 did it before.

        A ticking timer reads as success before HUD detection engages, and
        exploiting re-armed itself on every success, so the cheap policies held
        the game forever. They may fill gaps; they may not take over.
        """
        monkeypatch.setattr(ma, "MAX_POLICY_STREAK", 4)
        monkeypatch.setattr(ma, "EXPLOIT_REPEATS", 8)
        a = make_agent(tool_calls=tool_call("UP"))
        a._prev_levels = 0
        a.actions.tried = {"UP": 1, "DOWN": 1}
        a.actions.last_effective = "UP"
        a._exploit_left = 999  # as if every turn had looked successful
        for i in range(12):
            a.choose_action([], mk_frame(grid=[[i % 2, 1], [1, 0]], available=(1, 2)))
        assert a._llm.calls, "the model must get a turn"
        assert a.stats["policy_yield"] >= 2, "the streak cap must fire repeatedly"


class TestBudgetAwareness:
    """Some actions spend a finite move budget and lose the level at zero.

    sk48 decrements a life counter inside its arrow-key handler and never in
    the click handler, so arrows can end the game while clicks are free. The
    agent infers this from the on-screen meter rather than from any game's
    source, which is why it carries over to games we have not seen.
    """

    def spend(self, model: ma.ActionModel, family: str, times: int, *, meter: bool) -> None:
        for _ in range(times):
            model.record(family, changed_gameplay=False, touched_hud=meter)

    def test_an_action_that_moves_the_meter_is_costly(self) -> None:
        model = ma.ActionModel()
        self.spend(model, "UP", 4, meter=True)
        self.spend(model, "MOUSE", 4, meter=False)
        assert model.costly("UP")
        assert not model.costly("MOUSE")

    def test_untried_actions_are_not_assumed_costly(self) -> None:
        assert not ma.ActionModel().costly("LEFT")

    def test_free_actions_are_explored_first(self) -> None:
        model = ma.ActionModel()
        self.spend(model, "UP", 4, meter=True)
        self.spend(model, "MOUSE", 4, meter=False)
        assert model.free_first(["UP", "MOUSE"])[0] == "MOUSE"

    def test_summary_marks_which_actions_spend(self) -> None:
        model = ma.ActionModel()
        self.spend(model, "UP", 4, meter=True)
        self.spend(model, "MOUSE", 2, meter=False)
        summary = model.summary(["UP", "MOUSE"])
        assert "SPENDS the move budget" in summary
        assert "MOUSE" in summary and "(free)" in summary

    def test_a_stuck_agent_explores_with_the_free_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ma, "PROBE_ACTIONS", True)
        a = make_agent(tool_calls=tool_call("UP"))
        a._prev_levels = 0
        a.actions.tried = {"UP": 4, "SPACE": 4}
        a.actions.spent = {"UP": 4}          # arrows move the meter
        still = mk_frame(grid=SMALL_GRID, available=(1, 5))
        for _ in range(5):
            a.choose_action([], still)
        assert a.stats["explore_free"] > 0
        assert a._last_action_name == "SPACE", "exploring must use the free action"

    def test_the_prompt_warns_which_actions_spend_the_budget(self) -> None:
        a = make_agent(tool_calls=tool_call("UP"))
        a._prev_levels = 0
        a.actions.tried = {"UP": 4}
        a.actions.spent = {"UP": 4}
        a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 6)))
        assert "SPEND the limited move budget" in user_text(a._llm.calls[-1])


class TestUndoDetection:
    """An action that reverts the board, found by watching rather than by name.

    sk48's ACTION7 restores a snapshot and costs no budget. Which action does
    this differs per game, so it is inferred: free, and it returns the board to
    the state before the previous action.
    """

    A = ((0, 0), (0, 0))
    B = ((1, 0), (0, 0))

    def test_spots_an_action_that_reverts_the_board(self) -> None:
        undo = ma.UndoDetector()
        undo.observe("RIGHT", self.A, costly=True)
        undo.observe("RIGHT", self.B, costly=True)
        undo.observe("ACTION7", self.A, costly=False)
        assert undo.candidate == "ACTION7"

    def test_a_costly_action_is_never_called_undo(self) -> None:
        """Reversal is only worth taking because it is free."""
        undo = ma.UndoDetector()
        undo.observe("RIGHT", self.A, costly=True)
        undo.observe("RIGHT", self.B, costly=True)
        undo.observe("LEFT", self.A, costly=True)
        assert undo.candidate is None

    def test_an_action_that_changes_nothing_is_not_undo(self) -> None:
        undo = ma.UndoDetector()
        undo.observe("SPACE", self.A, costly=False)
        undo.observe("SPACE", self.A, costly=False)
        undo.observe("SPACE", self.A, costly=False)
        assert undo.candidate is None

    def test_a_false_positive_can_be_ruled_out(self) -> None:
        undo = ma.UndoDetector()
        undo.observe("RIGHT", self.A, costly=True)
        undo.observe("RIGHT", self.B, costly=True)
        undo.observe("ACTION7", self.A, costly=False)
        undo.rule_out("ACTION7")
        assert undo.candidate is None
        undo.observe("ACTION7", self.A, costly=False)
        assert undo.candidate is None, "a ruled-out action must stay ruled out"

    def test_history_stays_bounded(self) -> None:
        undo = ma.UndoDetector()
        for i in range(50):
            undo.observe("UP", ((i % 2, 0), (0, 0)), costly=True)
        assert len(undo.history) <= 3

    def test_reset_forgets_the_previous_level(self) -> None:
        undo = ma.UndoDetector()
        undo.observe("RIGHT", self.A, costly=True)
        undo.observe("RIGHT", self.B, costly=True)
        undo.observe("ACTION7", self.A, costly=False)
        undo.reset()
        assert undo.candidate is None and undo.history == []

    def test_agent_reverts_a_wasted_costly_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ma, "PROBE_ACTIONS", False)
        a = make_agent(tool_calls=tool_call("UP"))
        a._prev_levels = 0
        a.actions.tried = {"UP": 4, "ACTION7": 4}
        a.actions.spent = {"UP": 4}            # arrows move the meter
        a.undo.candidate = "ACTION7"
        a.memory._no_effect_streak = 1
        a._last_action_name = "UP"
        a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 7)))
        assert a.stats["undo"] == 1
        assert a._last_action_name == "ACTION7"


class TestObjectTracking:
    """Turns described as movement, because these games are about arrangement.

    Reading sk48's source showed the agent was playing the wrong game: the
    player pushes coloured tiles until they match a fixed reference, and the
    win compares what sits under each piece. "96 cells changed" cannot express
    that; "the blue piece was pushed up" can.
    """

    def grid(self, cells: dict[tuple[int, int], int], size: int = 8) -> ma.Grid:
        rows = [[0] * size for _ in range(size)]
        for (r, c), v in cells.items():
            rows[r][c] = v
        return tuple(tuple(r) for r in rows)

    def test_a_push_reads_as_two_objects_moving(self) -> None:
        tracker = ma.ObjectTracker()
        tracker.update(ma.segment(self.grid({(4, 2): 1, (3, 2): 2})))
        events = tracker.update(ma.segment(self.grid({(3, 2): 1, (2, 2): 2})))
        assert any("moved up" in e for e in events)
        assert len(events) == 2, "the player and the piece it pushed"

    def test_direction_and_distance_are_reported(self) -> None:
        tracker = ma.ObjectTracker()
        tracker.update(ma.segment(self.grid({(1, 1): 3})))
        events = tracker.update(ma.segment(self.grid({(1, 4): 3})))
        assert "moved right by (0,3)" in events[0]

    def test_an_object_that_leaves_is_reported_as_vanished(self) -> None:
        tracker = ma.ObjectTracker()
        tracker.update(ma.segment(self.grid({(1, 1): 3})))
        assert "vanished" in tracker.update(ma.segment(self.grid({})))[0]

    def test_a_new_object_is_reported_as_appeared(self) -> None:
        tracker = ma.ObjectTracker()
        tracker.update(ma.segment(self.grid({})))
        assert "appeared" in tracker.update(ma.segment(self.grid({(2, 2): 5})))[0]

    def test_a_still_board_reports_nothing(self) -> None:
        tracker = ma.ObjectTracker()
        board = ma.segment(self.grid({(1, 1): 3}))
        tracker.update(board)
        assert tracker.update(board) == []

    def test_scenery_is_not_tracked(self) -> None:
        """Background reshapes as pieces cross it and would churn every turn."""
        big = {(r, c): 7 for r in range(8) for c in range(8)}
        tracker = ma.ObjectTracker()
        tracker.update(ma.segment(self.grid(big)), board_cells=64)
        assert tracker.update(ma.segment(self.grid({**big, (0, 0): 0})), board_cells=64) == []

    def test_movers_rank_the_pieces_in_play(self) -> None:
        tracker = ma.ObjectTracker()
        for step in range(4):
            tracker.update(ma.segment(self.grid({(step, 1): 3, (6, 6): 4})))
        assert tracker.movers(), "a piece that keeps moving must be identifiable"

    def test_the_prompt_carries_movement_and_progress(self) -> None:
        a = make_agent(tool_calls=tool_call("RIGHT"))
        a.choose_action([], mk_frame(grid=[list(r) for r in self.grid({(4, 2): 1})]))
        a.choose_action([], mk_frame(grid=[list(r) for r in self.grid({(3, 2): 1})]))
        prompt = user_text(a._llm.calls[-1])
        assert "What moved:" in prompt and "moved up" in prompt
        assert "Progress: level" in prompt


class TestMechanicNotes:
    """One line of the model's own understanding, carried across turns.

    Every other note in the prompt is harness bookkeeping, so the model
    re-derived the mechanic from scratch each turn. orak's 2048 agent is given
    the rules outright; ARC-AGI-3 hides them, so the model writes its own.
    """

    @pytest.fixture(autouse=True)
    def enable_notes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pinned so the suite stays green when the flag is off for an A/B."""
        monkeypatch.setattr(ma, "MECHANIC_NOTES", True)

    def test_a_note_reaches_the_next_turn_s_prompt(self) -> None:
        a = make_agent(tool_calls=tool_call("UP", '{"note": "arrows push tiles"}'))
        a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 2)))
        a.choose_action([], mk_frame(grid=[[1, 0], [0, 1]], available=(1, 2)))
        assert "arrows push tiles" in user_text(a._llm.calls[-1])

    def test_the_prompt_says_so_when_no_theory_exists_yet(self) -> None:
        a = make_agent(tool_calls=tool_call("UP"))
        a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 2)))
        assert "not yet recorded a theory" in user_text(a._llm.calls[-1])

    def test_a_new_theory_replaces_the_old_one(self) -> None:
        """A log would grow without bound; carrying history is what overflowed
        n_ctx in v31 and reduced the agent to random play."""
        mem = ma.LevelMemory()
        mem.record_mechanic("clicking swaps colours")
        mem.record_mechanic("arrows push tiles")
        assert mem.mechanic == "arrows push tiles"

    def test_a_level_change_keeps_the_theory_but_drops_the_board(self) -> None:
        """Levels of one game share a mechanic: sk48's eight levels carry
        identical flags and differ only in piece count. Wiping the theory threw
        it away exactly when the finished level had just confirmed it."""
        mem = ma.LevelMemory()
        mem.record_mechanic("arrows push tiles")
        mem.record("UP", ((0, 0),), ((0, 0),))  # a dead action on this board
        mem.reset()
        assert mem.mechanic == "arrows push tiles"
        assert not mem._dead, "board-specific findings must not survive the level"

    @pytest.mark.parametrize("junk", ["", "   ", None])
    def test_an_empty_note_is_not_recorded(self, junk: object) -> None:
        mem = ma.LevelMemory()
        mem.record_mechanic("arrows push tiles")
        assert mem.record_mechanic(junk) is False
        assert mem.mechanic == "arrows push tiles", "junk must not erase a real theory"

    def test_a_long_note_is_bounded(self) -> None:
        mem = ma.LevelMemory()
        mem.record_mechanic("x " * 500)
        assert len(mem.mechanic) <= ma.MAX_NOTE_CHARS

    def test_repeating_the_same_theory_is_not_counted_as_new(self) -> None:
        """The stat must measure the model revising its understanding, not it
        echoing the line back; an inert feature and a useless one must differ."""
        a = make_agent(tool_calls=tool_call("UP", '{"note": "arrows push tiles"}'))
        for _ in range(3):
            a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 2)))
        assert a.stats["mechanic_note"] == 1


class TestForwardModel:
    """Learning what an action does, so sequences can be searched unplayed.

    The scored run talks to a remote gateway with no local environment, so the
    game cannot be forked the way the local sk48 solver forked it. Undo does
    not help either: it restores sprite positions and never refunds the move
    budget. A predicted sequence is the only kind we can afford to search.
    """

    def test_nothing_is_predicted_from_a_single_sighting(self) -> None:
        """One sighting cannot be told from a fluke, and search misled by a
        confident wrong model is worse off than search with no model."""
        fm = ma.ForwardModel()
        fm.observe("UP", [(-6, 0)])
        assert fm.predict("UP") is None

    def test_a_repeated_translation_becomes_a_prediction(self) -> None:
        fm = ma.ForwardModel()
        for _ in range(2):
            fm.observe("UP", [(-6, 0)])
        assert fm.predict("UP") == (-6, 0)

    def test_a_push_does_not_confuse_the_effect(self) -> None:
        """A push moves pusher and pushed by the same vector; a stray object
        drifting elsewhere must not outvote them."""
        fm = ma.ForwardModel()
        for _ in range(2):
            fm.observe("UP", [(-6, 0), (-6, 0), (3, 1)])
        assert fm.predict("UP") == (-6, 0)

    @pytest.mark.parametrize(
        "moves", [[(-6, 0), (1, 0)], [(0, -6), (0, 1)], [(2, 2), (-2, -2)]]
    )
    def test_a_tie_is_not_evidence(self, moves: list[tuple[int, int]]) -> None:
        """Breaking a tie by picking a vector chose the largest tuple, so a
        player moving up 6 lost to an enemy drifting down 1 — and since
        negative deltas always lose that comparison, UP and LEFT were the
        systematically mislearned directions."""
        assert ma.ForwardModel.summarise(moves) is None

    def test_a_tied_turn_teaches_the_model_nothing(self) -> None:
        fm = ma.ForwardModel()
        for _ in range(3):
            fm.observe("UP", [(-6, 0), (1, 0)])
        assert fm.effects.get("UP") in (None, {}), "a guess must not be recorded"

    def test_an_action_that_stops_working_is_relearned(self) -> None:
        """A piece that moved up 6 all game moves 0 at a wall. On lifetime
        counts the successes outvote the wall for many turns, leaving the model
        confident exactly where it has started being wrong."""
        fm = ma.ForwardModel()
        for _ in range(ma.ForwardModel.WINDOW):
            fm.observe("UP", [(-6, 0)])
        for _ in range(4):
            fm.observe("UP", [])
        assert fm.predict("UP") != (-6, 0), "the wall must overtake stale history"

    def test_an_inconsistent_action_is_not_predicted(self) -> None:
        fm = ma.ForwardModel()
        for move in [(-6, 0), (0, 6), (6, 0), (0, -6)]:
            fm.observe("SPACE", [move])
        assert fm.predict("SPACE") is None, "no majority means no usable prediction"

    def test_accuracy_counts_only_genuine_predictions(self) -> None:
        """Scoring must happen before learning. Learning from an outcome and
        then claiming to have predicted it measures nothing at all."""
        fm = ma.ForwardModel()
        fm.observe("UP", [(-6, 0)])          # nothing predictable yet
        fm.observe("UP", [(-6, 0)])          # still below the threshold
        assert fm.predicted == 0, "unpredicted turns must not count as correct"
        fm.observe("UP", [(-6, 0)])          # now predicted, and right
        fm.observe("UP", [(9, 9)])           # predicted, and wrong
        assert (fm.predicted, fm.correct) == (2, 1)
        assert fm.accuracy == 0.5

    def test_accuracy_is_zero_before_any_prediction(self) -> None:
        assert ma.ForwardModel().accuracy == 0.0

    def test_an_action_that_moves_nothing_is_learnable(self) -> None:
        """A wall is information: knowing UP does nothing here prunes it from
        every candidate sequence."""
        fm = ma.ForwardModel()
        for _ in range(2):
            fm.observe("UP", [])
        assert fm.predict("UP") == (0, 0)

    def test_the_model_survives_a_level_change(self) -> None:
        """sk48's eight levels carry identical flags and differ only in piece
        count, so the physics learned on one level hold on the next."""
        a = make_agent(tool_calls=tool_call("UP"))
        for _ in range(2):
            a.forward.observe("UP", [(-6, 0)])
        a.choose_action([], mk_frame(grid=SMALL_GRID, available=(1, 2), levels=1))
        assert a.forward.predict("UP") == (-6, 0)

    @staticmethod
    def board_with(row: int) -> list[list[int]]:
        grid = [[0] * 8 for _ in range(8)]
        grid[row][2] = 1
        return grid

    @pytest.mark.parametrize(
        ("kwargs", "path"),
        [({"tool_calls": tool_call("UP")}, "tool call"),
         ({"text": "action(['UP'])"}, "raw text")],
    )
    def test_a_turn_is_observed_exactly_once(
        self, kwargs: dict[str, Any], path: str
    ) -> None:
        """prompt_for runs twice whenever the tool call misses — once eagerly
        and once through the raw-text thunk. Observing there re-consumed the
        frame, every object self-matched at zero delta, and the phantom (0,0)
        tied the real translation forever. The tool-call path alone hid it.
        """
        a = make_agent(**kwargs)
        a.choose_action([], mk_frame(grid=self.board_with(4)))
        a.choose_action([], mk_frame(grid=self.board_with(3)))
        assert a.forward.effects.get("UP") == {(-1, 0): 1}, f"via {path}"

    def test_an_unmatched_frame_teaches_nothing(self) -> None:
        """Nothing matched is not the same as nothing moved: a piece can change
        colour or shape, outgrow the scenery cut, or swap with a twin. Learning
        'this action does nothing' there prunes a working action from every
        candidate sequence."""
        a = make_agent(tool_calls=tool_call("UP"))
        a.choose_action([], mk_frame(grid=self.board_with(4)))
        blank = [[0] * 8 for _ in range(8)]
        a.choose_action([], mk_frame(grid=blank))
        assert not a.forward.effects.get("UP"), "an unmatched frame is not a no-op"


class TestRollout:
    """Predicting where pieces end up, which is what a sequence needs.

    Summing translations cannot express a wall: told UP moves a piece -6, a
    five-step rollout adds -6 five times and walks it off the board.
    """

    @staticmethod
    def model(vector: tuple[int, int] = (-1, 0), moving: str = "a") -> ma.ForwardModel:
        fm = ma.ForwardModel()
        for _ in range(2):
            fm.observe("UP", [vector])
        fm.moving.add(moving)
        return fm

    def test_a_moving_piece_is_carried(self) -> None:
        cells = {"a": frozenset({(4, 2)})}
        assert self.model().step(cells, "UP", 8, 8) == {"a": frozenset({(3, 2)})}

    def test_scenery_stays_put(self) -> None:
        cells = {"a": frozenset({(4, 2)}), "wall": frozenset({(0, 7)})}
        after = self.model().step(cells, "UP", 8, 8)
        assert after["wall"] == frozenset({(0, 7)}), "only pieces seen to move may move"

    def test_a_piece_stops_at_the_board_edge(self) -> None:
        """The whole point: a vector sum walks it off the board instead."""
        cells = {"a": frozenset({(0, 2)})}
        assert self.model().step(cells, "UP", 8, 8) == {"a": frozenset({(0, 2)})}

    def test_a_piece_stops_at_an_obstacle(self) -> None:
        cells = {"a": frozenset({(4, 2)}), "wall": frozenset({(3, 2)})}
        after = self.model().step(cells, "UP", 8, 8)
        assert after["a"] == frozenset({(4, 2)}), "a blocked piece must not pass through"

    def test_an_unknown_action_is_not_guessed(self) -> None:
        assert self.model().step({"a": frozenset({(4, 2)})}, "DOWN", 8, 8) is None

    def test_nothing_known_to_move_is_not_guessed(self) -> None:
        """Predicting no change would look identical to predicting a wall."""
        fm = self.model(moving="somethingelse")
        assert fm.step({"a": frozenset({(4, 2)})}, "UP", 8, 8) is None

    def test_a_known_no_op_leaves_the_board_alone(self) -> None:
        fm = ma.ForwardModel()
        for _ in range(2):
            fm.observe("UP", [])
        fm.moving.add("a")
        cells = {"a": frozenset({(4, 2)})}
        assert fm.step(cells, "UP", 8, 8) == cells

    def test_a_sequence_rolls_forward_until_it_is_blocked(self) -> None:
        """Three UPs from row 2 reach row 0 and stop, rather than reaching -1."""
        fm, cells = self.model(), {"a": frozenset({(2, 2)})}
        for _ in range(3):
            cells = fm.step(cells, "UP", 8, 8)
        assert cells == {"a": frozenset({(0, 2)})}
