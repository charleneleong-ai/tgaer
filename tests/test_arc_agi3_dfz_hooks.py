"""Harness hooks for dfranzen's notebook; they patch his ToolAgent, so the tests need DFZ_SRC."""

from __future__ import annotations

import heapq
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = os.environ.get("DFZ_SRC")
if not SRC:
    pytest.skip(
        "DFZ_SRC (dfranzen's solution checkout) is not set", allow_module_level=True
    )
sys.path[:0] = [str(Path(SRC) / "ARC3-Inference"), str(REPO / "sia-oss/bench/dfz")]
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "test-model")

import inference.agent.tool_agent as ta  # noqa: E402
from endgame_rotation import ENDGAME_FRACTION, EndgameRotation  # noqa: E402
from inference.agent.runtime_state import (  # noqa: E402
    Frame,
    HistoryEntry,
    write_runtime_state,
)
from level_reattempt import THRESHOLD_TOKENS, LevelReattempt  # noqa: E402
from reasoning_window import DRAIN_TURNS, KEEP_TURNS, ReasoningWindow  # noqa: E402
from stall_probe import PROBE_AFTER_TOKENS, StallProbe  # noqa: E402

START, MOVED = ((0, 0),), ((0, 1),)


def opener(self: Any, n: int, **_: Any) -> str:
    return f"opener {n}"


class FakeSession:
    """The harness side: owns the board and state file; step_env is a bound method, as in the solver."""

    def __init__(self, path: Path, level: int = 2) -> None:
        self.path, self.level, self.grid, self.resets = path, level, START, 0
        self.entries = [
            HistoryEntry(action="ACTION1", frame=Frame(START, 0, level - 1))
        ]
        self.act("ACTION4")  # the level-completing action lands on the new level

    def act(self, action: str) -> None:
        self.grid = START if action == "RESET" else MOVED
        frame = Frame(self.grid, len(self.entries), self.level)
        self.entries.append(HistoryEntry(action=action, frame=frame))
        self.write_runtime_state()

    def write_runtime_state(self) -> None:
        write_runtime_state(
            self.path, current_frame=self.entries[-1].frame, history=self.entries
        )

    def _execute_auto_reset(self) -> None:
        self.resets += 1
        self.act("RESET")

    def should_stop(self) -> bool:
        return False

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {}


def record_turn(agent: Any, state_path: Path, n: int, *_: Any, **__: Any) -> None:
    prompt = agent._build_user_prompt(n)
    agent.prompts.append(prompt)
    agent._history_messages += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"reply {n}"},
    ]


class TestLevelReattempt:
    @pytest.fixture
    def agent(self) -> Any:
        attrs = {"analyze": record_turn, "_build_user_prompt": opener}
        cls = type("Reattempting", (ta.ToolAgent,), attrs)
        LevelReattempt.install(cls)
        a = cls()
        a.prompts = []
        a._history_messages = [
            {"role": "user", "content": "level 1 opener"},
            {"role": "assistant", "content": "level 1 work"},
        ]
        a._kept_functions = {"old_helper": "def old_helper(): ..."}
        return a

    @pytest.fixture
    def game(self, tmp_path: Path) -> FakeSession:
        return FakeSession(tmp_path / "state.json")

    @staticmethod
    def turn(agent: Any, game: FakeSession, n: int, tokens: int = 0) -> str:
        agent._session_generated_tokens += tokens
        agent.analyze(game.path, n, ["ACTION1"], game.step_env)
        return agent.prompts[-1]

    def test_a_stall_resets_the_board_and_restores_the_level_start(
        self, agent: Any, game: FakeSession
    ) -> None:
        self.turn(agent, game, 1)
        game.act("ACTION1")
        agent._kept_functions["wrong_model"] = "def wrong_model(): ..."
        self.turn(agent, game, 2, THRESHOLD_TOKENS - 1)
        assert game.resets == 0
        prompt = self.turn(agent, game, 3, 1)
        assert game.resets == 1 and game.grid == START and "pressed RESET" in prompt
        assert [m["content"] for m in agent._history_messages] == [
            "level 1 opener",
            "level 1 work",
            prompt,
            "reply 3",
        ]
        assert set(agent._kept_functions) == {"old_helper"}
        assert agent._context_was_trimmed

    def test_the_snapshot_survives_trimming_of_earlier_messages(
        self, agent: Any, game: FakeSession
    ) -> None:
        self.turn(agent, game, 1)
        game.act("ACTION1")
        agent._history_messages = agent._history_messages[-2:]
        assert "[Fresh look]" in self.turn(agent, game, 2, THRESHOLD_TOKENS)
        assert agent._history_messages[0]["content"] == "level 1 opener"

    def test_at_most_two_per_level_and_a_new_level_resets_the_count(
        self, agent: Any, game: FakeSession
    ) -> None:
        self.turn(agent, game, 1)
        fresh = []
        for n in range(2, 7):
            game.act("ACTION1")
            fresh.append("[Fresh look]" in self.turn(agent, game, n, THRESHOLD_TOKENS))
        assert fresh.count(True) == 2 and game.resets == 2
        game.level = 3
        game.act("ACTION4")
        self.turn(
            agent, game, 7, THRESHOLD_TOKENS
        )  # a new level only takes the snapshot
        game.act("ACTION1")
        assert "[Fresh look]" in self.turn(agent, game, 8, THRESHOLD_TOKENS)

    @pytest.mark.parametrize("last_action", ["RESET", None])
    def test_no_reset_without_an_action_on_this_level(
        self, agent: Any, game: FakeSession, last_action: str | None
    ) -> None:
        self.turn(agent, game, 1)
        if last_action:
            game.act(last_action)
        prompt = self.turn(agent, game, 2, THRESHOLD_TOKENS)
        assert game.resets == 0 and "left as it is" in prompt

    def test_a_failing_reset_still_lets_the_turn_run(
        self, agent: Any, game: FakeSession
    ) -> None:
        self.turn(agent, game, 1)
        game.act("ACTION1")

        def broken() -> None:
            raise RuntimeError("engine down")

        game._execute_auto_reset = broken
        assert "left as it is" in self.turn(agent, game, 2, THRESHOLD_TOKENS)


def reasoning_turns(n: int, key: str = "reasoning_content") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(n):
        out += [
            {"role": "user", "content": f"opener {i}"},
            {"role": "assistant", "content": f"reply {i}", key: f"thinking {i}"},
        ]
    return out


class TestReasoningWindow:
    @pytest.fixture
    def agent(self) -> Any:
        attrs = {
            "_persistent_history_messages": lambda self, messages, **k: messages[1:]
        }
        cls = type("Windowed", (ta.ToolAgent,), attrs)
        ReasoningWindow.install(cls)
        return cls()

    def persist(
        self, agent: Any, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return agent._persistent_history_messages([{"role": "system"}, *history])

    def test_below_the_batch_size_history_is_untouched(self, agent: Any) -> None:
        history = reasoning_turns(KEEP_TURNS + DRAIN_TURNS)
        assert self.persist(agent, history) == history

    @pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
    def test_a_full_batch_strips_all_but_the_last_turns(
        self, agent: Any, key: str
    ) -> None:
        n = KEEP_TURNS + DRAIN_TURNS + 1
        history = self.persist(agent, reasoning_turns(n, key))
        kept = [m[key] for m in history if ReasoningWindow.has_reasoning(m)]
        assert kept == [f"thinking {i}" for i in range(n - KEEP_TURNS, n)]
        assert history[1] == {"role": "assistant", "content": "reply 0"}
        assert agent._context_was_trimmed

    def test_the_prefix_is_stable_until_the_next_batch(self, agent: Any) -> None:
        once = self.persist(agent, reasoning_turns(KEEP_TURNS + DRAIN_TURNS + 1))
        more = once + reasoning_turns(DRAIN_TURNS - 1)
        assert self.persist(agent, more) == more

    def test_originals_are_not_mutated(self, agent: Any) -> None:
        history = reasoning_turns(KEEP_TURNS + DRAIN_TURNS + 1)
        self.persist(agent, history)
        assert all(
            "reasoning_content" in m for m in history if m["role"] == "assistant"
        )


class TestEndgameRotation:
    @pytest.fixture
    def gate(self) -> Any:
        cls = type("RotatingGate", (ta._PriorityGate,), {})
        EndgameRotation.install(cls)
        return cls(1)

    @staticmethod
    def queue(
        gate: Any, items: list[tuple[int, int]], total: float, left: float
    ) -> None:
        now = time.monotonic()
        gate._clock_start, gate._clock_deadline = now - (total - left), now + left
        gate._free = 0
        for priority, token in items:
            heapq.heappush(gate._waiting, (-priority, token))

    @staticmethod
    def admit(gate: Any) -> int:
        before = set(gate._admitted)
        gate._free = 1
        with gate._cond:
            gate._pump()
        (token,) = gate._admitted - before
        return token

    def test_outside_the_window_priority_wins(self, gate: Any) -> None:
        self.queue(gate, [(10, 1), (90, 2), (50, 3)], 1000, 500)
        assert self.admit(gate) == 2

    def test_inside_the_window_the_longest_waiting_goes_first(self, gate: Any) -> None:
        self.queue(gate, [(10, 1), (90, 2), (50, 3)], 1000, ENDGAME_FRACTION * 1000 - 1)
        assert [self.admit(gate) for _ in range(3)] == [1, 2, 3]


class TestStallProbe:
    def test_objects_skip_the_background_and_keep_one_cell_per_kind(self) -> None:
        grid = [[0] * 8 for _ in range(8)]
        grid[1][1] = 3
        grid[6][4] = grid[6][5] = 5
        kinds = StallProbe.objects(tuple(tuple(r) for r in grid))
        assert [(c, s) for c, s, _, _ in kinds] == [(3, 1), (5, 2)]

    def test_no_probe_before_the_stall_threshold(self, tmp_path: Path) -> None:
        cls = type("Probing", (ta.ToolAgent,), {"analyze": lambda *a, **k: None})
        StallProbe.install(cls)
        game, agent = FakeSession(tmp_path / "state.json"), cls()
        calls: list[dict[str, Any]] = []
        agent.analyze(game.path, 1, ["ACTION1"], calls.append)
        agent._session_generated_tokens += PROBE_AFTER_TOKENS - 1
        agent.analyze(game.path, 2, ["ACTION1"], calls.append)
        assert calls == []
