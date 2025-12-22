from __future__ import annotations

from typing import Any, Dict, List


class SimpleMemory:
    """Tiny memory stub for agents to store key/value facts."""

    def __init__(self) -> None:
        self._store: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        self._store.append(item)

    def all(self) -> List[Dict[str, Any]]:
        return list(self._store)
