from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class OrakStepResult:
    state: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class OrakClient:
    """Stub ORAK client – replace with real API integration."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def start_mission(self, mission_id: str, seed: int | None = None) -> Any:
        # TODO: call real API
        return {"mission_id": mission_id, "seed": seed}

    def apply_action(self, mission_id: str, action: Any) -> OrakStepResult:
        # TODO: call real API
        return OrakStepResult(state={}, reward=0.0, done=True, info={})

    def render_state(self, mission_id: str, state: Any) -> Any:
        # TODO: pretty-print or convert to text/grid
        return {"mission_id": mission_id, "state": state}
