from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EvalResult:
    score: float
    details: Dict[str, Any]
