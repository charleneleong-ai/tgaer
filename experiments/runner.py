from __future__ import annotations

from typing import Any, Dict, List

from tgaer.envs.arc.arc_env import ArcEnvironment
from tgaer.envs.arc.arc_dataset import load_arc_tasks
from tgaer.envs.orak.orak_env import OrakEnvironment
from tgaer.envs.orak.orak_api import OrakClient

from tgaer.agents.llm_agent import LLMAgent
from tgaer.agents.hybrid_agent import HybridAgent
from tgaer.agents.planning_agent import PlanningAgent

from tgaer.optimization.geq_loop import GEQOptimizer, expand_search_space
from tgaer.evaluation.arc_eval import evaluate_arc_agent
from tgaer.evaluation.orak_eval import evaluate_orak_agent
from tgaer.logging.mlflow_utils import init_mlflow


def _build_envs(env_cfg: Dict[str, Any]):
    kind = env_cfg["kind"]
    if kind == "arc":
        tasks = load_arc_tasks(env_cfg["dataset_path"], split=env_cfg.get("split", "train"))
        envs = [ArcEnvironment(task) for task in tasks[: env_cfg.get("num_tasks", len(tasks))]]
        return envs

    if kind == "orak":
        client = OrakClient(base_url=env_cfg["api_endpoint"])
        mission_ids = env_cfg["missions"]
        envs = [OrakEnvironment(client=client, mission_id=m) for m in mission_ids]
        return envs

    raise ValueError(f"Unknown env kind: {kind}")


def _build_agent(agent_cfg: Dict[str, Any]):
    kind = agent_cfg["kind"]
    if kind == "llm_only":
        return LLMAgent(config=agent_cfg)
    if kind == "hybrid":
        return HybridAgent(config=agent_cfg)
    if kind == "planning":
        return PlanningAgent(config=agent_cfg)
    raise ValueError(f"Unknown agent kind: {kind}")


def _build_eval_fn(env_kind: str):
    if env_kind == "arc":
        return evaluate_arc_agent
    if env_kind == "orak":
        return evaluate_orak_agent
    raise ValueError(f"Unknown env kind for eval: {env_kind}")


def run_experiment(cfg: Dict[str, Any]) -> None:
    experiment_name = cfg.get("experiment_name", "tgaer_experiment")
    init_mlflow(experiment_name=experiment_name)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    opt_cfg = cfg.get("optimization", {})
    eval_cfg = cfg["evaluation"]

    envs = _build_envs(env_cfg)
    eval_fn = _build_eval_fn(env_cfg["kind"])

    if opt_cfg.get("kind") == "geq":
        def make_agent(overrides: Dict[str, Any]):
            merged = {**agent_cfg, **overrides}
            return _build_agent(merged)

        search_space = expand_search_space(opt_cfg["search_space"])
        optimizer = GEQOptimizer(make_agent=make_agent, eval_fn=lambda a, e: eval_fn(a, e, eval_cfg))
        result = optimizer.run(
            envs=envs,
            search_space=search_space,
            run_name=experiment_name,
        )
        print(f"[RESULT] Best config: {result['best_cfg']}")
        print(f"[RESULT] Best score: {result['best_score']}")
    else:
        agent = _build_agent(agent_cfg)
        scores: List[float] = []
        for env in envs:
            res = eval_fn(agent, env, eval_cfg)
            scores.append(res.score)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"[RESULT] Avg score over {len(envs)} envs: {avg_score}")
