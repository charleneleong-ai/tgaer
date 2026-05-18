from __future__ import annotations

from typing import Any, List, Optional

from tgaer.core.env_base import Environment, Transition
from tgaer.envs.arc.arc_dataset import ArcTask

Grid = List[List[int]]


def _score_predictions(kaggle_preds: List[dict], gt_outputs: List[Grid]) -> float:
    if not gt_outputs:
        return 0.0
    correct = 0
    for i, gt in enumerate(gt_outputs):
        if i >= len(kaggle_preds):
            continue
        pack = kaggle_preds[i] or {}
        a1 = pack.get("attempt_1")
        a2 = pack.get("attempt_2")
        if (a1 is not None and a1 == gt) or (a2 is not None and a2 == gt):
            correct += 1
    return correct / len(gt_outputs)


class ArcEnvironment(Environment):
    """Single-shot ARC environment.

    reset() yields {train_pairs, test_inputs} as observation.
    step(action) accepts Kaggle two-attempts predictions
    (list of {"attempt_1": grid, "attempt_2": grid}) and scores them
    against held-out test_outputs. done=True after one step.
    """

    def __init__(self, task: ArcTask) -> None:
        self._task = task
        self._done = False

    def reset(self, seed: Optional[int] = None) -> dict:
        self._done = False
        return {
            "task_id": self._task.task_id,
            "train_pairs": [{"input": p.input, "output": p.output} for p in self._task.train_pairs],
            "test_inputs": list(self._task.test_inputs),
        }

    def step(self, action: Any) -> Transition:
        if self._task.test_outputs is None:
            reward = 0.0
            info = {"scored": False, "reason": "no test_outputs available (test split)"}
        else:
            reward = _score_predictions(action or [], self._task.test_outputs)
            info = {"scored": True, "num_tests": len(self._task.test_outputs)}
        self._done = True
        return Transition(state=None, action=action, reward=reward, done=True, info=info)

    def render(self) -> Any:
        return {
            "task_id": self._task.task_id,
            "num_train": len(self._task.train_pairs),
            "num_test": len(self._task.test_inputs),
        }

    def task_id(self) -> str:
        return self._task.task_id
