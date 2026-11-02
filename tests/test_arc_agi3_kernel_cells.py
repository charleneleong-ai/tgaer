"""Notebook cells patched into the dfranzen M2 kernel: the scorecard keep-alive and the turn archive.

Both cells run inside the Kaggle notebook against the harness modules, so the tests load them by path against
minimal stand-ins for `inference.framework.solver` and `inference.agent.{tool_agent,python_tool_sandbox}`.
"""

from __future__ import annotations

import http.server
import sys
import textwrap
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

CELLS = Path(__file__).resolve().parents[1] / "sia-oss/bench/dfz/kernel_cells"
ANCHOR = '    _refresh_state(initial.get("state") or {})\n'


def step(action: str, level: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        action=action, frame=types.SimpleNamespace(level=level)
    )


class FakeSession:
    def __init__(self, i: int) -> None:
        self.game = types.SimpleNamespace(
            game_run=types.SimpleNamespace(game_id=f"g{i}")
        )
        self.actions = 0

    def should_stop(self) -> bool:
        return False

    def _execute_action(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.actions += 1
        return {}

    def _execute_auto_reset(self) -> None:
        self._execute_action()

    def _play_inner(self) -> None:
        pass


class Health:
    def __init__(self) -> None:
        self.up = False
        state = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200 if state.up else 503)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def install_modules(
    monkeypatch: pytest.MonkeyPatch, **modules: types.ModuleType
) -> None:
    for name in ("inference", "inference.agent", "inference.framework"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def run_cell(name: str, **globals_: Any) -> dict[str, Any]:
    ns: dict[str, Any] = {"__name__": "cell", **globals_}
    exec(compile((CELLS / f"{name}.py").read_text(), name, "exec"), ns)
    return ns


class TestKeepAlive:
    @pytest.fixture
    def cell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> Iterator[tuple[dict[str, Any], Health, list[FakeSession]]]:
        solver = types.ModuleType("inference.framework.solver")
        solver._HarnessGameSession = type("S", (FakeSession,), {})
        install_modules(monkeypatch, **{"inference.framework.solver": solver})
        for key, value in {
            "POLL_S": "0.02",
            "IDLE_LOADING_S": "0.1",
            "IDLE_RUNNING_S": "0.3",
        }.items():
            monkeypatch.setenv(f"OURS_KEEPALIVE_{key}", value)
        health = Health()
        ns = run_cell(
            "keepalive",
            SERVED_MODEL_HOST="127.0.0.1",
            SERVED_MODEL_PORT=health.server.server_port,
            NOTEBOOK_START_TIME=time.time(),
            TRUE_SUBMISSION=True,
        )
        sessions = [solver._HarnessGameSession(i) for i in range(3)]
        for s in sessions:
            s._play_inner()
        yield ns, health, sessions
        health.server.shutdown()

    def test_resets_while_loading_and_on_stalls_but_never_during_play(
        self, cell
    ) -> None:
        ns, health, sessions = cell
        time.sleep(0.5)
        loading = ns["KEEPALIVE"]["resets"]
        assert loading >= 3 and sum(s.actions for s in sessions) == loading
        assert len({s.actions for s in sessions}) <= 2  # round robin spreads the RESETs

        health.up = True
        start = time.time()
        while time.time() - start < 0.6:
            sessions[0]._execute_action()
            time.sleep(0.02)
        assert ns["KEEPALIVE"]["resets"] == loading

        time.sleep(0.8)
        assert ns["KEEPALIVE"]["resets"] > loading


class TestTurnArchive:
    @pytest.fixture
    def ns(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        tool_agent = types.ModuleType("inference.agent.tool_agent")
        sandbox = types.ModuleType("inference.agent.python_tool_sandbox")
        sandbox._SANDBOX_BOOTSTRAP = (
            f"def main():\n    initial = {{}}\n{ANCHOR}    pass\n"
        )

        class ToolAgent:
            keep: list = []

            def _persistent_history_messages(self, messages, tools=None):
                return list(self.keep)

            def _run_python_tool(self, state_path, arguments):
                return tool_agent.run_sandboxed_python(
                    code=arguments["code"], initial_state=self.state
                )

        tool_agent.ToolAgent = ToolAgent
        tool_agent.run_sandboxed_python = lambda **kwargs: kwargs
        tool_agent._build_system_prompt = lambda **kwargs: "SYSTEM"
        install_modules(
            monkeypatch,
            **{
                "inference.agent.tool_agent": tool_agent,
                "inference.agent.python_tool_sandbox": sandbox,
            },
        )
        return run_cell("turn_archive")

    @staticmethod
    def search(ns: dict[str, Any], agent: Any, level: int, *calls: str) -> list[Any]:
        """Run the injected sandbox snippet on the state the wrapper would send, like the real bootstrap does."""
        agent.state = {"current_frame": {"level": level}}
        sent = agent._run_python_tool(None, {"code": "search_turns()"})["initial_state"]
        bootstrap = ns["_ta_sandbox"]._SANDBOX_BOOTSTRAP
        snippet = textwrap.dedent(bootstrap[bootstrap.index(ANCHOR) + len(ANCHOR) :])
        frame = types.SimpleNamespace(level=level)
        env: dict[str, Any] = {
            "initial": {"state": sent},
            "runtime_globals": {"current_frame": frame},
        }
        exec(snippet, env)
        return [
            eval(c, {"search_turns": env["runtime_globals"]["search_turns"]})
            for c in calls
        ]

    def test_archives_dropped_turns_once_with_the_level_they_were_written_on(
        self, ns
    ) -> None:
        agent = ns["_ta_mod"].ToolAgent()
        agent._session_runtime_dir = "game-1"
        system = {"role": "system", "content": "SYSTEM"}
        user = {"role": "user", "content": "board"}

        def turn(level: int, history: list, keep: list) -> None:
            agent._ours_level, agent.keep = level, keep
            assert agent._persistent_history_messages([system, *history]) == keep

        push = {
            "role": "assistant",
            "reasoning": "the blue block is pushed by ACTION3",
            "content": "",
        }
        result = {"role": "tool", "content": "blue block moved 3 cells"}
        gate = {
            "role": "assistant",
            "content": [{"type": "text", "text": "the gate opens on the red wall"}],
        }

        turn(1, [user, push, result], [user, push, result])
        turn(2, [user, push, result, gate], [gate])
        turn(
            2, [user, push, result, gate], [gate]
        )  # the same drop again archives nothing new
        turn(2, [gate], [])

        entries = agent._ours_turn_archive["entries"]
        assert [(e["level"], e["role"]) for e in entries] == [
            (1, "assistant"),
            (1, "tool"),
            (2, "assistant"),
        ]

        here, everywhere, gates = self.search(
            ns,
            agent,
            2,
            "search_turns('blue')",
            "search_turns('blue', level=None)",
            "search_turns('gate')",
        )
        assert here == [] and len(everywhere) == 2 and len(gates) == 1
        assert all(x.startswith("[unverified: your turn") for x in everywhere + gates)
        assert ns["TURN_ARCHIVE"]["searches"] == 1

    def test_a_new_game_starts_an_empty_archive(self, ns) -> None:
        agent, capture = ns["_ta_mod"].ToolAgent(), ns["_ta_capture"]
        agent._session_runtime_dir = "game-1"
        capture(agent, [{"role": "assistant", "content": "old game"}], [])
        agent._session_runtime_dir = "game-2"
        assert self.search(ns, agent, 1, "search_turns('', level=None)") == [[]]

    def test_prompt_line_is_appended(self, ns) -> None:
        assert (
            ns["_ta_mod"]
            ._build_system_prompt(tool_output_tokens=1)
            .startswith("SYSTEM\n- `search_turns(")
        )


class TestSolvedLevelMemory:
    @pytest.fixture
    def ns(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        tool_agent = types.ModuleType("inference.agent.tool_agent")
        runtime_state = types.ModuleType("inference.agent.runtime_state")
        # level 1: a failed attempt, RESET, then the winning attempt; the last action lands on level 2
        entries = [
            step("ACTION1", 1),
            step("ACTION2", 1),
            step("RESET", 1),
            step("ACTION3", 1),
            step("ACTION4", 2),
        ]
        runtime_state.load_runtime_state = lambda path: ({}, entries)

        class ToolAgent:
            def analyze(self, state_path):
                return "turn"

            def _build_user_prompt(self):
                return "board"

            def _note_history_evicted(self):
                pass

            def _trim_messages_for_context(self, messages):
                return messages[:1] + messages[2:]

        tool_agent.ToolAgent = ToolAgent
        install_modules(
            monkeypatch,
            **{
                "inference.agent.tool_agent": tool_agent,
                "inference.agent.runtime_state": runtime_state,
            },
        )
        return run_cell("solved_level_memory")

    def test_asks_for_a_rule_then_pins_it_only_when_the_history_drains(
        self, ns
    ) -> None:
        agent = ns["_ta_slm"].ToolAgent()
        agent._system_prompt = "SYSTEM"
        agent._history_messages = [
            {"role": "assistant", "content": "pushed the block onto the pad"}
        ]
        agent._last_step_summary = {"level": 2}

        agent.analyze("state")
        (rec,) = agent._slm_levels
        assert (rec["level"], rec["actions"]) == (1, ["ACTION3", "ACTION4"])
        assert "`Rule for level 1:`" in agent._build_user_prompt()

        agent._history_messages.append(
            {
                "role": "assistant",
                "content": "Rule for level 1: push the block onto the pad, then step off it.",
            }
        )
        agent.analyze("state")
        assert rec["rule"] == "push the block onto the pad, then step off it."
        assert agent._build_user_prompt() == "board"
        assert agent._system_prompt == "SYSTEM"  # nothing changes until a drain

        agent._note_history_evicted()
        assert agent._system_prompt.startswith("SYSTEM\n\n=== SOLVED LEVELS")
        assert "push the block onto the pad" in agent._system_prompt
        system, user = {"role": "system", "content": "SYSTEM"}, {"role": "user"}
        trimmed = agent._trim_messages_for_context([system, user, user])
        assert trimmed[0]["content"] == agent._system_prompt

    @pytest.mark.parametrize(
        "text",
        [
            "Rule for level 1: too short",
            "Rule for level 2: right level number, but the wrong one asked for",
            "no rule line at all",
        ],
    )
    def test_ignores_missing_short_or_wrong_level_rules(self, ns, text) -> None:
        assert ns["_slm_find_rule"](text, 1) is None
