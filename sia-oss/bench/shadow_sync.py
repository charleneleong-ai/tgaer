#!/usr/bin/env python3
"""Can a locally-instantiated game shadow a run action-for-action?

The scored Kaggle run is `OPERATION_MODE=online` against a gateway sidecar with
`ENVIRONMENTS_DIR=` empty, so there is no game object to `deepcopy` — the trick
M0 used to plan for free. The bundled `environment_files` are still in the
kernel, though, so a *shadow* instance replaying the agent's own actions could
serve as the simulator instead.

That only works if the game is reproducible: a fresh instance fed the same
actions must produce byte-identical frames. This plays a trajectory, replays it
into a new instance, and reports the first divergence.

What this cannot test from here: the gateway itself. It shows that the game
*class* is deterministic and seed-independent, not that the gateway serves this
same build. Confirm the version pin (`environment_files/<game>/<hash>`) against
a scored run before relying on it.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import typer
from loguru import logger

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
)
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)
os.environ.setdefault("ARC_TGAER_REPO", str(REPO))

from arcengine import ActionInput, GameAction  # noqa: E402

app = typer.Typer(add_completion=False)

GAMES = ("lp85", "ls20", "sp80", "tu93", "sc25", "ar25", "m0r0", "s5i5")
# ACTION6 needs coordinates; the rest are bare. Probing a stride-3 lattice is
# how `m0_plan` finds buttons, so reuse the same grid for click candidates.
CLICK = 6


def frame_of(fd: object) -> np.ndarray:
    """The latest grid, or an empty array when the engine returns no frame.

    Reset and game-over responses carry an empty `frame` list. That is itself
    state worth comparing — a shadow in sync should go blank on exactly the same
    steps — so it becomes an empty array rather than an error.
    """
    frames = getattr(fd, "frame", None) or []
    if not frames:
        return np.empty((0, 0), dtype=np.int16)
    return np.asarray(frames[-1], dtype=np.int16)


def apply(game: object, action: tuple[int, int, int]) -> object:
    """Perform one recorded action: (action_id, row, col).

    Looks the enum member up by *name*: `GameAction` keys its value map by
    `(int, subclass)` tuples, so `GameAction(4)` raises "not a valid GameAction"
    even though `ACTION4.value` is 4.
    """
    ident, row, col = action
    member = GameAction[f"ACTION{ident}"]
    data = {"x": int(col), "y": int(row)} if ident == CLICK else {}
    return game.perform_action(  # type: ignore[attr-defined]
        ActionInput(id=member, data=data), raw=True
    )


def fresh(game_id: str, seed: int) -> tuple[object, object]:
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    env = arc.make(game_id)
    if env is None:
        raise RuntimeError(f"no environment for {game_id}")
    fd = env.reset()
    return env._game, fd


def random_actions(
    game_id: str, available: list[int], steps: int, rng: random.Random
) -> list[tuple[int, int, int]]:
    """A trajectory of (action_id, row, col), clicks on a stride-3 lattice."""
    out = []
    for _ in range(steps):
        ident = rng.choice(available)
        if ident == CLICK:
            out.append((ident, rng.randrange(0, 64, 3), rng.randrange(0, 64, 3)))
        else:
            out.append((ident, 0, 0))
    return out


@app.command()
def main(
    steps: int = typer.Option(300, help="Actions per trajectory."),
    games: str = typer.Option(",".join(GAMES), help="Comma-separated game ids."),
    seed: int = typer.Option(0),
) -> None:
    """Replay each game's trajectory into a fresh instance and diff the frames."""
    require_starter()
    verdicts: dict[str, str] = {}
    for game_id in games.split(","):
        rng = random.Random(seed)
        try:
            live, fd = fresh(game_id, seed)
        except RuntimeError as exc:
            logger.warning("{}: {}", game_id, exc)
            continue
        available = [int(a) for a in (getattr(fd, "available_actions", None) or [])]
        if not available:
            available = [1, 2, 3, 4, CLICK]
        actions = random_actions(game_id, available, steps, rng)

        recorded = [frame_of(fd)]
        levels = 0
        for action in actions:
            out = apply(live, action)
            recorded.append(frame_of(out))
            levels = max(levels, int(getattr(out, "levels_completed", 0)))

        shadow, shadow_fd = fresh(game_id, seed + 1)  # deliberately a different seed
        replayed = [frame_of(shadow_fd)]
        for action in actions:
            replayed.append(frame_of(apply(shadow, action)))

        diverged = next(
            (
                i
                for i, (a, b) in enumerate(zip(recorded, replayed, strict=True))
                if not np.array_equal(a, b)
            ),
            None,
        )
        if diverged is None:
            verdicts[game_id] = "in sync"
            logger.success(
                "{}: {} actions replayed byte-identical ({} levels cleared)",
                game_id,
                steps,
                levels,
            )
        else:
            verdicts[game_id] = f"diverged at {diverged}"
            logger.error("{}: diverged at step {} of {}", game_id, diverged, steps)

    ok = sum(v == "in sync" for v in verdicts.values())
    logger.info("{}/{} games shadow the live run exactly", ok, len(verdicts))
    if ok != len(verdicts):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
