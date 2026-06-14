"""Shaped-reward variant of the vendored ``arc_agi_3_env`` — granular reward for RL.

Keeps the upstream env file pristine; adds a dense exploration signal on top of
its binary WIN reward so GRPO/GSPO has a gradient on sparse games. The reward
logic itself lives (and is unit-tested) in ``tgaer.envs.arc_agi3.shaping``; this
module is just the verifiers wiring.

Runtime deps: ``verifiers`` + ``arcengine`` (the vendored env) and ``tgaer``
(for the shaping core) installed in the training venv. Entrypoint
``load_environment`` mirrors the upstream signature, so it drops into the trainer.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from arc_agi_3_env import ArcAgi3Env  # vendored upstream (package re-export)
from arc_agi_3_env import load_environment as _upstream_load_environment

from tgaer.envs.arc_agi3 import shaping


class ShapedArcAgi3Env(ArcAgi3Env):
    """Upstream env + per-step exploration accumulation for shaped reward."""

    async def env_response(self, messages: Any, state: dict[str, Any], **kwargs: Any):
        result = await super().env_response(messages, state, **kwargs)
        observation = state.get("last_observation")
        if observation:
            shaping.observe(state, observation)
        return result


def build_shaped_rubric(
    w_win: float = 1.0, w_novelty: float = 0.2, w_stall: float = 0.1
) -> vf.Rubric:
    return vf.Rubric(
        funcs=[shaping.win_reward, shaping.novelty_reward, shaping.anti_stall_penalty],
        weights=[w_win, w_novelty, w_stall],
    )


def load_environment(
    game_family: str = "simple_maze",
    max_turns: int = 100,
    level_index: int = -1,
    game_id: str | None = None,
    w_win: float = 1.0,
    w_novelty: float = 0.2,
    w_stall: float = 0.1,
    **_: Any,
) -> vf.Environment:
    # Reuse upstream to build the dataset, then swap in our class + shaped rubric.
    base = _upstream_load_environment(
        game_family=game_family, max_turns=max_turns, level_index=level_index, game_id=game_id
    )
    rubric = build_shaped_rubric(w_win=w_win, w_novelty=w_novelty, w_stall=w_stall)
    return ShapedArcAgi3Env(dataset=base.dataset, rubric=rubric, max_turns=max_turns)
