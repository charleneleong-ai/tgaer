"""Kaggle submission wrapper for the REPL agent.

Adapts our ArcAgi3ReplAgent to the Kaggle ARC-AGI-3 Agent interface.
This file is the ONLY file you edit for Kaggle submissions.

Usage:
    1. Copy this file to the ARC-AGI-3-Kaggle-Starter repo as agent/my_agent.py
    2. Run: make play-local
    3. Run: make submit
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework
try:
    from agents.agent import Agent
except ImportError:
    # Local testing — provide a stub base class
    class Agent:  # type: ignore[no-redef]
        """Stub base class for local testing."""
        game_id: str = ""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        @property
        def name(self) -> str:
            return f"Agent.{self.game_id}"


class MyAgent(Agent):
    """REPL agent using LLM with function calling for ARC-AGI-3.

    Uses OpenAI function calling with a python tool, segmentation view,
    and structured world model — aligned with Tufa Labs' Duck harness.
    """

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Import our agent (lazy to avoid circular imports)
        from tgaer.agents.arc_agi3_repl import ArcAgi3ReplAgent

        self._agent = ArcAgi3ReplAgent(
            model=os.environ.get("REPL_MODEL", "openai/gpt-4o-mini"),
            vision=True,
            max_tool_steps=int(os.environ.get("REPL_MAX_TOOL_STEPS", "8")),
            temperature=0.6,
        )
        self._prev_levels = -1
        self._step = 0

    @property
    def name(self) -> str:
        return f"{super().name}.repl"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop when we win the whole game."""
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Convert FrameData to our format, call REPL agent, convert back."""

        # Reset on new level
        if latest_frame.levels_completed != self._prev_levels:
            self._prev_levels = latest_frame.levels_completed
            self._step = 0
            self._agent._history.clear()
            self._agent._messages.clear()
            self._agent._world_model = {k: "" for k in self._agent._world_model}
            self._agent._levels = latest_frame.levels_completed

        self._step += 1

        # Convert FrameData to our observation format
        # FrameData.frame is list[list[list[int]]] — list of grids
        grids = latest_frame.frame if latest_frame.frame else []
        grid = grids[-1] if grids else []

        # Map GameState to our string format
        state_map = {
            GameState.NOT_PLAYED: "NOT_FINISHED",
            GameState.NOT_FINISHED: "NOT_FINISHED",
            GameState.WIN: "WIN",
            GameState.GAME_OVER: "GAME_OVER",
        }
        state = state_map.get(latest_frame.state, "NOT_FINISHED")

        # Build observation
        obs = {
            "frame": [grid] if grid else [],
            "available_actions": latest_frame.available_actions or [1, 2, 3, 4, 5, 6],
            "levels_completed": latest_frame.levels_completed,
            "state": state,
        }

        # Call our REPL agent
        arc_action = self._agent.act(obs)

        # Convert ArcAction to GameAction
        game_action = self._arc_to_game_action(arc_action)
        return game_action

    def _arc_to_game_action(self, arc_action: Any) -> GameAction:
        """Convert our ArcAction to Kaggle's GameAction."""
        action_map = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }

        game_action = action_map.get(arc_action.id, GameAction.ACTION1)

        if game_action.is_complex() and arc_action.x is not None:
            game_action.set_data({"x": arc_action.x, "y": arc_action.y})
            game_action.reasoning = {"why": f"REPL agent click at ({arc_action.x}, {arc_action.y})"}
        else:
            game_action.reasoning = f"REPL agent action {arc_action.id}"

        return game_action
