#!/usr/bin/env python3
"""How many real actions does it take to learn an action's effect well enough to plan?

The scored kernel has no game object to fork, so a forward model has to be
learned from live play — and every observation costs a scored action. That trade
is only worth making if the cost is small and fixed: pay N actions per button
once, then plan for free.

Measures two things per game:

* **Is an action's effect state-independent?** If pressing button k always
  applies the same displacement set, one or two sightings pin it down. If the
  effect depends on the board, a lookup is hopeless and only a real rule will do.
* **How many sightings until the effect stops changing?** The break-even point
  against blind search.

`ForwardModel` in `arc_agi3_kaggle.py` already learns online, but only a single
translation vector per action — enough for a movement game, not for a button
that permutes a dozen sprites.
"""

from __future__ import annotations

import copy
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

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
import m1_codegen as M  # noqa: E402

from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)
os.environ.setdefault("ARC_TGAER_REPO", str(REPO))

app = typer.Typer(add_completion=False)

GAMES = ("lp85", "ls20", "sp80", "tu93", "sc25")


def displacement(before: list, after: list) -> tuple:
    """What an action did, as a sorted set of (object -> new position) moves.

    Keyed on the object's identity (colour and size) plus its displacement, not
    absolute position, so the same rule applied at different places reads the
    same.
    """
    gone = sorted(set(map(tuple, before)) - set(map(tuple, after)))
    fresh = sorted(set(map(tuple, after)) - set(map(tuple, before)))
    moves = []
    for g, f in zip(gone, fresh, strict=False):
        moves.append((g[0], g[3], g[4], f[1] - g[1], f[2] - g[2]))
    return tuple(sorted(moves))


@app.command()
def main(
    games: str = typer.Option(",".join(GAMES)),
    steps: int = typer.Option(400, help="Actions to observe per game."),
    seed: int = typer.Option(0),
) -> None:
    """Report, per game, whether action effects are fixed and how fast they settle."""
    require_starter()
    for game_id in games.split(","):
        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(REPO / "environment_files"),
        )
        env = arc.make(game_id)
        if env is None:
            logger.warning("{}: no environment", game_id)
            continue
        fd = env.reset()
        game = env._game
        base = M.grid(fd)
        bg = M.background(base)
        # Movement games expose ACTION1-5 and no clicks, so probing only the
        # click lattice reports "no actions" on exactly the games whose effects
        # are most likely to be fixed translations.
        simple = [
            int(a)
            for a in (getattr(fd, "available_actions", None) or [])
            if int(a) != M.CLICK_ID
        ]
        cells: list = list(simple) + M.button_cells(game, base)
        if not cells:
            logger.warning("{}: no effective actions found", game_id)
            continue

        rng = random.Random(seed)
        effects: dict[int, list[tuple]] = defaultdict(list)
        node, state = copy.deepcopy(game), M.objects(base, bg)
        for _ in range(steps):
            idx = rng.randrange(len(cells))
            child = copy.deepcopy(node)
            out = M.act(child, cells[idx])
            nxt = M.objects(M.grid(out), bg)
            effects[idx].append(displacement(state, nxt))
            if int(getattr(out, "levels_completed", 0)) > 0 or "GAME_OVER" in str(
                getattr(out, "state", "")
            ):
                node, state = copy.deepcopy(game), M.objects(base, bg)
            else:
                node, state = child, nxt

        distinct = {k: len(set(v)) for k, v in effects.items()}
        sightings = {k: len(v) for k, v in effects.items()}
        fixed = sum(1 for k in distinct if distinct[k] == 1)
        # Sightings before the last previously-unseen effect showed up: the point
        # after which more observation taught nothing new.
        settle = {}
        for k, seq in effects.items():
            seen: set[tuple] = set()
            last_new = 0
            for i, eff in enumerate(seq, 1):
                if eff not in seen:
                    seen.add(eff)
                    last_new = i
            settle[k] = last_new
        logger.info(
            "{}: {} actions | {} with a single fixed effect | distinct effects {} "
            "| sightings {} | settled after {}",
            game_id,
            len(cells),
            fixed,
            [distinct[k] for k in sorted(distinct)],
            [sightings[k] for k in sorted(sightings)],
            [settle[k] for k in sorted(settle)],
        )


if __name__ == "__main__":
    app()
