from __future__ import annotations

from typing import Any, Dict

from tgaer.core.agent_base import Agent
from tgaer.optimization.dspy_signatures import build_orak_program


class PlanningAgent(Agent):
    """Planning-oriented agent for ORAK-like environments."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.program = build_orak_program(
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 512),
        )
        self.history: str = ""

    def act(self, observation: Any) -> Any:
        mission_description = observation.get("mission_description", "")
        observation_summary = observation.get("observation_summary", "")
        step_res = self.program.step(
            mission_description=mission_description,
            observation_summary=observation_summary,
            history=self.history,
        )
        # Update text history naively for now
        self.history += f"Action: {step_res['next_action']}\n"
        return step_res["next_action"]
