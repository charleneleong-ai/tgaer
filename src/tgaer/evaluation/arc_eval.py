from __future__ import annotations

from typing import Any, Dict

from tgaer.core.agent_base import Agent
from tgaer.core.env_base import Environment
from tgaer.evaluation.metrics import EvalResult


def evaluate_arc_agent(
    agent: Agent, env: Environment, cfg: Dict[str, Any]
) -> EvalResult:
    observation = env.reset()
    action = agent.act(observation)
    transition = env.step(action)
    return EvalResult(
        score=float(transition.reward),
        details={
            "task_id": observation.get("task_id"),
            "done": transition.done,
            **(transition.info or {}),
        },
    )
