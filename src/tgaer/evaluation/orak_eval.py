from __future__ import annotations

from typing import Any, Dict

from tgaer.core.agent_base import Agent
from tgaer.core.env_base import Environment
from tgaer.evaluation.metrics import EvalResult


def evaluate_orak_agent(agent: Agent, env: Environment, cfg: Dict[str, Any]) -> EvalResult:
    state = env.reset()
    total_reward = 0.0
    max_steps = cfg.get("max_steps_per_mission", 50)
    done = False

    for _ in range(max_steps):
        observation = env.render()
        if not isinstance(observation, dict):
            observation = {"observation_summary": str(observation)}
        action = agent.act(observation)
        transition = env.step(action)
        total_reward += transition.reward
        done = transition.done
        if done:
            break

    return EvalResult(score=total_reward, details={"done": done})
