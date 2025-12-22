from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, List

from tgaer.core.agent_base import Agent
from tgaer.core.env_base import Environment
from tgaer.evaluation.metrics import EvalResult
from tgaer.logging.mlflow_utils import log_eval_run
from tgaer.logging.langsmith_utils import traced_run


class GEQOptimizer:
    """Generic GEQ loop for evaluation-driven config search."""

    def __init__(
        self,
        make_agent: Callable[[Dict[str, Any]], Agent],
        eval_fn: Callable[[Agent, Environment], EvalResult],
    ) -> None:
        self.make_agent = make_agent
        self.eval_fn = eval_fn

    def run(
        self,
        envs: List[Environment],
        search_space: List[Dict[str, Any]],
        run_name: str,
    ) -> Dict[str, Any]:
        best_cfg: Dict[str, Any] | None = None
        best_score = float("-inf")

        for trial_idx, cfg in enumerate(search_space):
            agent = self.make_agent(cfg)
            trial_scores: List[float] = []

            with traced_run(
                name=f"{run_name}_trial_{trial_idx}",
                metadata={"config": cfg},
            ):
                for env in envs:
                    result = self.eval_fn(agent, env)
                    trial_scores.append(result.score)

            log_eval_run(
                run_name=f"{run_name}_trial_{trial_idx}",
                cfg=cfg,
                scores=trial_scores,
                extra_metrics={"num_envs": float(len(envs))},
            )

            avg_score = sum(trial_scores) / len(trial_scores) if trial_scores else 0.0
            if avg_score > best_score:
                best_score = avg_score
                best_cfg = cfg

        return {"best_cfg": best_cfg or {}, "best_score": best_score}


def expand_search_space(search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Very simple grid search expansion for search_space dict of lists."""
    keys = list(search_space.keys())
    values_lists = [search_space[k] for k in keys]
    configs: List[Dict[str, Any]] = []
    for combo in product(*values_lists):
        cfg = dict(zip(keys, combo))
        configs.append(cfg)
    return configs
