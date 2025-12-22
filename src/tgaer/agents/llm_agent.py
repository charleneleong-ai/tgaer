from __future__ import annotations

from typing import Any, Dict

from tgaer.core.agent_base import Agent


class LLMAgent(Agent):
    """Simple LLM-only agent stub."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        # TODO: init LLM client

    def act(self, observation: Any) -> Any:
        # TODO: call LLM with prompt built from observation
        raise NotImplementedError("LLMAgent.act is not implemented yet.")
