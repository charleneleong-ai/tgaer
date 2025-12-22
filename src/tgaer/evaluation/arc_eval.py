from __future__ import annotations

from typing import Any, Dict

from tgaer.core.agent_base import Agent
from tgaer.core.env_base import Environment
from tgaer.evaluation.metrics import EvalResult


def evaluate_arc_agent(agent: Agent, env: Environment, cfg: Dict[str, Any]) -> EvalResult:
    _ = env.reset()
    observation = env.render()
    # In practice, you will adapt observation to the HybridAgent expectation
    if not isinstance(observation, dict):
        observation = {"input_grid": str(observation)}
    action = agent.act(observation)
    transition = env.step(action)
    score = float(transition.reward)
    return EvalResult(score=score, details={"done": transition.done})
