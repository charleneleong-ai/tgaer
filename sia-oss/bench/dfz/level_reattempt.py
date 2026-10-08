"""Re-attempt a stalled level with fresh eyes: reset the board, drop the level's own conversation, keep earlier levels.

When a game has generated more than THRESHOLD_TOKENS on its current level without clearing it, the harness
issues RESET (the same path as its post-death auto-reset), restores the history snapshot taken just before
the level's first turn, and restores the retained helper functions to their level-start set. The retry is a
fresh sample from the level-start board. Runs at the start of a turn, before the opener is built.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from inference.agent.runtime_state import load_runtime_state

THRESHOLD_TOKENS = 80_000
MAX_REATTEMPTS = 2
NOTE = (
    "\n[Fresh look] You spent about {tokens:,} generated tokens on this level without clearing it. {board} "
    "Your notes from that attempt were removed so you can re-examine the level without its assumptions; "
    "everything from earlier levels is kept. Re-derive what the goal is from what you can see before acting."
)
RESET_NOTE = "The harness pressed RESET, so the level is back at its starting board (that cost one action)."
NO_RESET_NOTE = "The board was left as it is."


class LevelReattempt:
    """Per-agent level bookkeeping, installed on dfranzen's ToolAgent class."""

    @staticmethod
    def on_level_start(agent: Any, level: int) -> None:
        agent._reattempt_level = level
        agent._level_history = list(agent._history_messages)
        agent._level_functions = copy.deepcopy(
            getattr(agent, "_kept_functions", {}) or {}
        )
        agent._reattempt_base = agent._session_generated_tokens
        agent._reattempts = 0

    @staticmethod
    def acted_on_level(entries: list[Any], level: int) -> bool:
        """True once a non-RESET action was taken on this level, so RESET restarts the level, not the game."""
        tail = entries[-2:]
        return (
            len(tail) == 2
            and all(e.frame is not None and e.frame.level == level for e in tail)
            and str(tail[-1].action or "").upper() != "RESET"
        )

    @staticmethod
    def reset_board(step_env: Any, entries: list[Any], level: int) -> bool:
        session = getattr(step_env, "__self__", None)
        if (
            session is None
            or not LevelReattempt.acted_on_level(entries, level)
            or session.should_stop()
        ):
            return False
        try:
            session._execute_auto_reset()
            session.write_runtime_state()
        except Exception as exc:  # a failed reset must never cost the model its turn
            print(f"[level-reattempt] reset failed: {exc!r}", flush=True)
            return False
        return True

    @staticmethod
    def maybe_reattempt(
        agent: Any, step_env: Any, entries: list[Any], level: int
    ) -> tuple[int, bool] | None:
        if agent._reattempts >= MAX_REATTEMPTS:
            return None
        spent = agent._session_generated_tokens - agent._reattempt_base
        if spent < THRESHOLD_TOKENS:
            return None
        agent._reattempt_base = agent._session_generated_tokens
        reset = LevelReattempt.reset_board(step_env, entries, level)
        agent._history_messages = list(agent._level_history)
        agent._kept_functions = copy.deepcopy(agent._level_functions)
        agent._reattempts += 1
        agent._note_history_evicted()
        return spent, reset

    @staticmethod
    def install(agent_cls: type) -> None:
        analyze = agent_cls.analyze
        build_prompt = agent_cls._build_user_prompt

        def _analyze(
            agent: Any,
            state_path: Path,
            action_num: int,
            valid_actions: list[str] | None = None,
            step_env: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            frame, entries = (
                load_runtime_state(state_path)
                if Path(state_path).exists()
                else (None, [])
            )
            if frame is not None:
                if agent.__dict__.get("_reattempt_level") != frame.level:
                    LevelReattempt.on_level_start(agent, frame.level)
                elif outcome := LevelReattempt.maybe_reattempt(
                    agent, step_env, entries, frame.level
                ):
                    spent, reset = outcome
                    print(
                        f"[level-reattempt] re-attempt {agent._reattempts} after {spent} tokens, reset={reset}",
                        flush=True,
                    )
                    agent._reattempt_note = NOTE.format(
                        tokens=spent, board=RESET_NOTE if reset else NO_RESET_NOTE
                    )
            return analyze(
                agent, state_path, action_num, valid_actions, step_env, *args, **kwargs
            )

        def _build_user_prompt(agent: Any, action_num: int, **kwargs: Any) -> str:
            return build_prompt(agent, action_num, **kwargs) + agent.__dict__.pop(
                "_reattempt_note", ""
            )

        agent_cls.analyze = _analyze
        agent_cls._build_user_prompt = _build_user_prompt
