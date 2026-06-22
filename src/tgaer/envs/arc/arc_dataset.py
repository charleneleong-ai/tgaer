from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

Grid = List[List[int]]


@dataclass(frozen=True)
class ArcPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train_pairs: List[ArcPair]
    test_inputs: List[Grid]
    test_outputs: Optional[List[Grid]] = field(default=None)


_SPLIT_FILES = {
    "train": ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
    "evaluation": (
        "arc-agi_evaluation_challenges.json",
        "arc-agi_evaluation_solutions.json",
    ),
    "test": ("arc-agi_test_challenges.json", None),
}


def load_arc_tasks(
    dataset_path: str | Path, split: str = "evaluation"
) -> List[ArcTask]:
    if split not in _SPLIT_FILES:
        raise ValueError(
            f"Unknown split: {split!r}. Expected one of {list(_SPLIT_FILES)}."
        )

    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"ARC dataset_path does not exist: {root}")

    challenges_name, solutions_name = _SPLIT_FILES[split]
    challenges = json.loads((root / challenges_name).read_text(encoding="utf-8"))

    solutions = None
    if solutions_name is not None:
        solutions_path = root / solutions_name
        if solutions_path.exists():
            solutions = json.loads(solutions_path.read_text(encoding="utf-8"))

    tasks: List[ArcTask] = []
    for task_id, blob in challenges.items():
        train_pairs = [
            ArcPair(input=ex["input"], output=ex["output"])
            for ex in blob.get("train", [])
        ]
        test_inputs = [ex["input"] for ex in blob.get("test", [])]
        test_outputs = solutions.get(task_id) if solutions is not None else None
        tasks.append(
            ArcTask(
                task_id=task_id,
                train_pairs=train_pairs,
                test_inputs=test_inputs,
                test_outputs=test_outputs,
            )
        )
    return tasks
