from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Agent(ABC):
    """Base class for all agents (LLM-only, hybrid, planning, etc.)."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        ...

    def update_context(self, feedback: Dict[str, Any]) -> None:
        """Optional hook, e.g. for GEQ, memory, RL updates, etc."""
        return
