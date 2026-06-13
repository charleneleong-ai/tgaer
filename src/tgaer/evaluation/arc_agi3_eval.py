from __future__ import annotations

from typing import Any

from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.metrics import EvalResult
from tgaer.guards import FutileActionGuard, Guard, GuardedAgent, RepeatedPlanGuard


def _default_guards() -> list[Guard]:
    return [FutileActionGuard(), RepeatedPlanGuard()]


def evaluate_arc_agi3_agent(
    agent: Agent,
    env: ArcAgi3Environment,
    cfg: dict[str, Any] | None = None,
) -> EvalResult:
    """Run one interactive ARC-AGI-3 episode, with trajectory guards wrapped
    around ``agent`` so degenerate loops get a planner hint mid-episode.

    The episode ends on the env's own terminal/action-cap signal or after
    ``cfg["max_steps"]`` (an outer safety bound). Score is the cumulative
    reward (the sum of per-step ``levels_completed`` deltas). Pass
    ``cfg["guards"]`` to override the guard set (``[]`` disables them).
    """
    cfg = cfg or {}
    guards = cfg["guards"] if "guards" in cfg else _default_guards()
    guarded = GuardedAgent(agent, guards, env=env)
    max_steps = cfg.get("max_steps", 1000)

    observation = env.reset()
    guarded.reset()
    total_reward = 0.0
    steps = 0
    done = False
    info: dict[str, Any] = {}

    while not done and steps < max_steps:
        action = guarded.act(observation)
        transition = env.step(action)
        observation = transition.state
        total_reward += transition.reward
        info = transition.info or {}
        done = transition.done
        steps += 1

    return EvalResult(
        score=total_reward,
        details={
            **info,
            "task_id": env.task_id(),
            "done": done,
            "steps": steps,
            "guard_hints_fired": guarded.hint_count,
        },
    )
