from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional


@dataclass
class Transition:
    state: Any
    action: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class Environment(ABC):
    """Generic environment interface for ARC / ORAK / other tasks."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        ...

    @abstractmethod
    def step(self, action: Any) -> Transition:
        ...

    @abstractmethod
    def render(self) -> Any:
        ...

    @abstractmethod
    def task_id(self) -> Hashable:
        ...
