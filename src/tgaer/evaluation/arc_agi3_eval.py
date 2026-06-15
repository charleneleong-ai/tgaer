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
    logger: Any | None = None,
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
    prev_hints = 0
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
        if logger is not None:
            hints = guarded.hint_count
            logger.log_step(
                step=steps,
                action_id=getattr(action, "id", None),
                reward=transition.reward,
                score=total_reward,
                levels_completed=info.get("levels_completed"),
                guard_fired=hints > prev_hints,
                frame=observation.get("frame"),
                # instruct models have no separate <think> trace — fall back to
                # the reply (which now carries the rationale) so the column fills.
                reasoning=getattr(agent, "last_reasoning", None)
                or getattr(agent, "last_reply", None),
                reply=getattr(agent, "last_reply", None),
            )
            prev_hints = hints

    details = {
        **info,
        "task_id": env.task_id(),
        "done": done,
        "steps": steps,
        "guard_hints_fired": guarded.hint_count,
    }
    if logger is not None:
        logger.finish({"score": total_reward, **details})
    return EvalResult(score=total_reward, details=details)
