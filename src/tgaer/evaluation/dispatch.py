from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from tgaer.agents.arc_agi3_llm import ArcAgi3LLMAgent
from tgaer.agents.arc_agi3_random import RandomArcAgi3Agent
from tgaer.core.agent_base import Agent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcTransport
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent
from tgaer.evaluation.metrics import EvalResult
from tgaer.evaluation.wandb_logger import build_logger
from tgaer.guards import FutileActionGuard, Guard, RepeatedPlanGuard

Loader = Callable[..., EvalResult]

_ARC_AGI3_AGENTS = {"random": RandomArcAgi3Agent, "llm": ArcAgi3LLMAgent}
_GUARDS = {"futile_action": FutileActionGuard, "repeated_plan": RepeatedPlanGuard}


def build_guards(cfg: dict[str, Any]) -> list[Guard]:
    enabled = cfg.get("guards", {})
    return [cls() for key, cls in _GUARDS.items() if enabled.get(key, True)]


def _build_arc_agi3_agent(cfg: dict[str, Any], seed: int) -> Agent:
    agent_cfg = cfg.get("agent", {})
    kind = agent_cfg.get("kind", "random")
    params = {k: v for k, v in agent_cfg.items() if k != "kind"}
    try:
        return _ARC_AGI3_AGENTS[kind](seed=seed, **params)
    except KeyError:
        raise ValueError(
            f"unknown agent kind {kind!r}; known: {sorted(_ARC_AGI3_AGENTS)}"
        ) from None


def _live_arc_agi3_transport() -> ArcTransport:
    from tgaer.envs.arc_agi3.arc_agi3_client import ArcAgi3Client

    api_key = os.environ.get("ARC_API_KEY")
    if not api_key:
        raise RuntimeError("ARC_API_KEY is not set")
    return ArcAgi3Client(api_key=api_key)


def _run_arc_agi3(
    cfg: dict[str, Any],
    *,
    transport: ArcTransport | None = None,
    agent: Agent | None = None,
) -> EvalResult:
    env_cfg = cfg["env"]
    transport = transport or _live_arc_agi3_transport()
    env = ArcAgi3Environment(
        transport, env_cfg["game_id"], max_actions=env_cfg.get("max_actions", 80)
    )
    eval_cfg = {
        "guards": build_guards(cfg),
        "max_steps": cfg.get("evaluation", {}).get("max_steps", 1000),
    }
    agent = agent or _build_arc_agi3_agent(cfg, cfg.get("seed", 0))
    logger = build_logger(cfg.get("wandb"), run_config=cfg)
    with _scorecard(transport):
        return evaluate_arc_agi3_agent(agent, env, eval_cfg, logger=logger)


@contextmanager
def _scorecard(transport: ArcTransport) -> Iterator[None]:
    """Open a scorecard around the run so a live play is scored, and always
    close it. No-ops for transports without scorecard support (test fakes)."""
    scored = hasattr(transport, "open_scorecard") and hasattr(
        transport, "close_scorecard"
    )
    if scored:
        transport.open_scorecard()
    try:
        yield
    finally:
        if scored:
            transport.close_scorecard()


def _not_wired(kind: str) -> Loader:
    def loader(cfg: dict[str, Any], **_: Any) -> EvalResult:
        raise NotImplementedError(f"eval loader for env.kind={kind!r} is not wired yet")

    return loader


_LOADERS: dict[str, Loader] = {
    "arc_agi3": _run_arc_agi3,
    "orak": _not_wired("orak"),
    "arc": _not_wired("arc"),
}


def available_kinds() -> list[str]:
    return sorted(_LOADERS)


def run_eval(cfg: dict[str, Any], **overrides: Any) -> EvalResult:
    """Dispatch ``cfg`` to the eval loop matching its ``env.kind``.

    ``overrides`` (e.g. ``transport=...``, ``agent=...``) are forwarded to the
    selected loader — the dependency-injection seam that keeps tests off the
    network.
    """
    kind = cfg.get("env", {}).get("kind")
    loader = _LOADERS.get(kind)
    if loader is None:
        raise ValueError(f"unknown env.kind {kind!r}; known: {available_kinds()}")
    return loader(cfg, **overrides)
