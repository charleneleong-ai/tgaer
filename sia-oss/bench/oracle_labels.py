#!/usr/bin/env python3
"""Persist the oracle's winning moves as (frame, action) labels.

`m0_suite.py` finds a winning plan and logs only its length, so the one thing
worth keeping — which action wins from which board — is discarded. This reruns
the same uninformed search, then replays each plan against the real game to
record the frame as it stood before every winning action.

The labels are the supervision for a shippable policy prior: the agent online
sees a frame and must rank actions, and these say what the right answer was.
Everything here comes from play (`available_actions`, the rendered grid, the
level counter) — no game internals, so nothing learned from them is privileged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m0_suite as S  # noqa: E402
import m1_codegen as M  # noqa: E402

from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

REPO = S.REPO
OUT = REPO / "sia-oss" / "bench" / "oracle"

app = typer.Typer(add_completion=False)


def encode(action: object) -> dict[str, int]:
    """Actions are an int id or a (row, col) click; JSON needs one shape."""
    if isinstance(action, tuple):
        return {"kind": M.CLICK_ID, "row": int(action[0]), "col": int(action[1])}
    return {"kind": int(action), "row": -1, "col": -1}


def avail_mask(fd: object) -> int:
    """Available action ids as a bitmask; the recall check needs what the agent
    could legally have proposed at this frame."""
    return sum(1 << int(a) for a in (getattr(fd, "available_actions", None) or []))


def label_game(
    game_id: str, budget: int, levels: int, per_game: float
) -> tuple[list[np.ndarray], list[dict[str, int]], list[int], list[int]]:
    """Plan each level, then replay it recording the frame before every action."""
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    env = arc.make(game_id)
    if env is None:
        raise RuntimeError("env-unavailable")
    fd = env.reset()
    game = env._game
    cells = S.actions_for(game, fd, M.grid(fd))
    if not cells:
        raise RuntimeError("no actions found")

    frames: list[np.ndarray] = []
    actions: list[dict[str, int]] = []
    which: list[int] = []
    avail: list[int] = []
    for level in range(levels):
        done = int(getattr(fd, "levels_completed", 0))
        plan, _ = S.plan_level(game, M.grid(fd), done, cells, budget, per_game)
        if plan is None:
            break
        for action in plan:
            frames.append(M.grid(fd))
            actions.append(encode(action))
            which.append(level)
            avail.append(avail_mask(fd))
            fd = M.act(game, action)
    return frames, actions, which, avail


@app.command()
def main(
    games: str = typer.Option(",".join(S.SUITE), help="Comma-separated game ids."),
    budget: int = typer.Option(20000, help="Expansions per level before giving up."),
    levels: int = typer.Option(3, help="Levels to attempt per game."),
    per_game: float = typer.Option(150.0, help="Seconds per level before giving up."),
) -> None:
    """Write one npz of labelled decisions per game the oracle can solve."""
    require_starter()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    solved: list[str] = []
    for game_id in games.split(","):
        try:
            frames, actions, which, avail = label_game(
                game_id, budget, levels, per_game
            )
        except Exception as exc:
            # The games raise on their own account under exhaustive probing;
            # one game's bug must not end the sweep.
            logger.error("{}: {}: {}", game_id, type(exc).__name__, exc)
            continue
        if not frames:
            logger.info("{}: no level reached, nothing to label", game_id)
            continue
        np.savez_compressed(
            OUT / f"{game_id}.npz",
            frames=np.stack(frames).astype(np.int16),
            kind=np.array([a["kind"] for a in actions], dtype=np.int16),
            row=np.array([a["row"] for a in actions], dtype=np.int16),
            col=np.array([a["col"] for a in actions], dtype=np.int16),
            level=np.array(which, dtype=np.int16),
            avail=np.array(avail, dtype=np.int32),
        )
        clicks = sum(1 for a in actions if a["kind"] == M.CLICK_ID)
        total += len(frames)
        solved.append(game_id)
        logger.success(
            "{}: {} labels over {} level(s), {} clicks / {} simple",
            game_id,
            len(frames),
            len(set(which)),
            clicks,
            len(actions) - clicks,
        )
    (OUT / "index.json").write_text(
        json.dumps({"games": solved, "labels": total}, indent=2) + "\n"
    )
    logger.info("{} labels across {} games -> {}", total, len(solved), OUT)


if __name__ == "__main__":
    app()
