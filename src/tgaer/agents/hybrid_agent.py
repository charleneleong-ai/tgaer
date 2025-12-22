from __future__ import annotations

from typing import Any, Dict

from tgaer.core.agent_base import Agent
from tgaer.optimization.dspy_signatures import build_arc_program


class HybridAgent(Agent):
    """Hybrid neuro-symbolic agent stub for ARC-like tasks."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        # Build a DSPy program for ARC by default; you can make this conditional.
        self.program = build_arc_program(
            temperature=config.get("temperature", 0.2),
            max_tokens=config.get("max_tokens", 512),
        )

    def act(self, observation: Any) -> Any:
        """Observation is expected to include task_description, examples, input_grid."""
        task_description = observation.get("task_description", "")
        examples = observation.get("examples", "")
        input_grid = observation.get("input_grid", "")
        out = self.program(
            task_description=task_description,
            examples=examples,
            input_grid=input_grid,
        )
        return out.get("output_grid")
