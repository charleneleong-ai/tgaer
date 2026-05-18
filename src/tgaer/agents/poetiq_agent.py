from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

from tgaer.core.agent_base import Agent


def _ensure_poetiq_on_path(submodule_path: Path) -> None:
    if not submodule_path.exists():
        raise FileNotFoundError(
            f"Poetiq submodule not found at {submodule_path}. "
            "Run `git submodule update --init challenges/poetiq-arc-agi-solver`."
        )
    p = str(submodule_path)
    if p not in sys.path:
        sys.path.insert(0, p)


_DEFAULT_SUBMODULE = (
    Path(__file__).resolve().parents[3] / "challenges" / "poetiq-arc-agi-solver"
)


class PoetiqAgent(Agent):
    """Wraps poetiq-arc-agi-solver as a TGAER Agent.

    Expects observation = {"train_pairs": [{"input", "output"}, ...],
                           "test_inputs": [grid, ...],
                           "task_id": str}.
    Returns action = [{"attempt_1": grid, "attempt_2": grid}, ...] (one per test input).
    Requires GEMINI_API_KEY / OPENAI_API_KEY etc. depending on the active CONFIG_LIST.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        submodule = Path(config.get("submodule_path", _DEFAULT_SUBMODULE)).expanduser().resolve()
        _ensure_poetiq_on_path(submodule)

        from dotenv import load_dotenv  # type: ignore[import-not-found]
        load_dotenv(submodule / ".env")

        from arc_agi.solve import solve  # type: ignore[import-not-found]
        from arc_agi.io import build_kaggle_two_attempts  # type: ignore[import-not-found]

        self._solve = solve
        self._build_kaggle = build_kaggle_two_attempts

    def act(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        train_pairs = observation["train_pairs"]
        test_inputs = observation["test_inputs"]
        task_id = observation.get("task_id")

        train_in = [pair["input"] for pair in train_pairs]
        train_out = [pair["output"] for pair in train_pairs]

        results = asyncio.run(
            self._solve(train_in, train_out, test_inputs, problem_id=task_id)
        )
        return self._build_kaggle(results, test_inputs)
