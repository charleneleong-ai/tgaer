#!/usr/bin/env python3
"""Phase 1: does a model-written `distance` give a gradient on more than lp85?

M1 asked for three functions and found `simulate` useless (0.1% of moved objects
against 46% for echoing the input) but `distance` good — it matched M0's
*privileged* heuristic and cleared lp85 L1 in 5 real actions. That was one level
of one game.

Without a simulator there is no offline search, so `distance` can only be a
*progress signal*: act, recompute on the new frame, keep or abandon. The cheapest
test of that is whether it decreases along a path already known to win. The
oracle's plans are exactly such paths, and they are shortest paths, so a useful
heuristic should fall on almost every step.

Scored against 0.5, which is what an uninformative function gets. Nothing here
reads a game file: the prompt carries the start board, transitions as diffs and
the objects a solved board has, all from play.
"""

from __future__ import annotations

import copy
import os
import random
import sys
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))

import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src" / "tgaer" / "agents" / "arc_agi3_explorer.py"),
)
import m0_suite as S  # noqa: E402
import m1_codegen as M  # noqa: E402

from tgaer.agents.arc_agi3_kaggle import HTTPChatBackend  # noqa: E402
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)

LABELS = REPO / "sia-oss" / "bench" / "oracle"
app = typer.Typer(add_completion=False)


def walk(
    game: Any, base: np.ndarray, actions: list[Any], limit: int, seed: int
) -> dict:
    """Random walk over *both* action kinds, recording object transitions.

    `m1_codegen.collect` probes clicks only, which suits lp85 and describes
    nothing on the seven roster games whose winning plans are pure simple
    actions.
    """
    bg = M.background(base)  # fixed for the level; see `m1_codegen.objects`
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    wins: list[list[tuple[int, ...]]] = []
    node, state = copy.deepcopy(game), M.objects(base, bg)
    for _ in range(limit):
        idx = rng.randrange(len(actions))
        child = copy.deepcopy(node)
        try:
            out = M.act(child, actions[idx])
        except Exception:
            continue  # games raise on their own account under random probing
        nxt = M.objects(M.grid(out), bg)
        rows.append({"state": state, "action": idx, "next_state": nxt})
        won = int(getattr(out, "levels_completed", 0)) > 0
        if won:
            wins.append(nxt)
        if won or "GAME_OVER" in str(getattr(out, "state", "")):
            node, state = copy.deepcopy(game), M.objects(base, bg)
        else:
            node, state = child, nxt
    return {"transitions": rows, "wins": wins, "bg": bg}


def monotone(fn: Any, frames: np.ndarray, bg: int) -> tuple[int, int, int]:
    """(falls, steps, distinct values) for `distance` along a winning path."""
    values: list[float] = []
    for arr in frames:
        try:
            v = float(fn(M.objects(arr.astype(np.int16), bg)))
        except Exception:
            return 0, 0, 0
        if not np.isfinite(v):
            return 0, 0, 0
        values.append(v)
    falls = sum(1 for a, b in zip(values, values[1:], strict=False) if b < a)
    return falls, len(values) - 1, len(set(values))


@app.command()
def main(
    games: str = typer.Option("", help="Comma-separated subset; default all labelled."),
    samples: int = typer.Option(300, help="Transitions to record per game."),
    train: int = typer.Option(50, help="Transitions shown to the model."),
    base_url: str = typer.Option("http://127.0.0.1:8011/v1"),
    model: str = typer.Option("qwen-27b"),
    attempts: int = typer.Option(2, help="Generations to try per game."),
    temperature: float = typer.Option(0.3),
    max_tokens: int = typer.Option(4000),
    timeout: float = typer.Option(1800.0),
    seed: int = typer.Option(0),
) -> None:
    """Generate a `distance` per game and score its gradient on the oracle path."""
    require_starter()
    ids = games.split(",") if games else sorted(p.stem for p in LABELS.glob("*.npz"))
    backend = HTTPChatBackend(
        base_url=base_url, model=model, seed=seed, timeout=timeout
    )
    good = 0
    logger.info(
        "{:6} {:>8} {:>9} {:>8} {:>7}", "game", "wins", "falls", "rate", "values"
    )
    for game_id in ids:
        arc = arc_agi.Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(REPO / "environment_files"),
        )
        env = arc.make(game_id)
        if env is None:
            continue
        fd = env.reset()
        game = env._game
        try:
            actions = S.actions_for(game, fd, M.grid(fd))
        except Exception as exc:
            logger.error("{}: probing raised {}", game_id, type(exc).__name__)
            continue
        rec = walk(game, M.grid(fd), actions, samples, seed)
        rows, wins, bg = rec["transitions"], rec["wins"], rec["bg"]
        if not rows:
            logger.warning("{}: no transitions recorded", game_id)
            continue

        try:
            prompt, _ = M.fit_prompt(
                rows[:train],
                rows[0]["state"],
                len(actions),
                wins,
                M.TASK_GOAL,
                base_url,
                model,
                reserve=max_tokens + 512,
            )
        except ValueError:
            logger.error("{}: no transition count fits the context", game_id)
            continue

        frames = np.load(LABELS / f"{game_id}.npz")["frames"]
        best = (0, 0, 0)
        for attempt in range(1, attempts + 1):
            try:
                reply = backend.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (urllib.error.HTTPError, Exception) as exc:
                logger.error("{} attempt {}: {}", game_id, attempt, exc)
                continue
            text = reply["choices"][0]["message"].get("content") or ""
            code = M.extract(text)
            ns = M.load(code, ("is_goal", "distance")) if code else None
            if not ns:
                continue
            got = monotone(ns["distance"], frames, bg)
            if got[1] and got[0] / got[1] > (best[0] / best[1] if best[1] else -1):
                best = got
        falls, steps, distinct = best
        rate = falls / steps if steps else 0.0
        # A function with two or three values gives a search no gradient, which
        # is the failure M1 called out by name.
        flag = "" if distinct > 3 else "  <-- degenerate"
        if steps and rate > 0.5 and distinct > 3:
            good += 1
        logger.info(
            "{:6} {:8} {:9} {:7.0%} {:7}{}",
            game_id,
            len(wins),
            f"{falls}/{steps}",
            rate,
            distinct,
            flag,
        )
    logger.info(
        "{}/{} games have a usable gradient (>50% falls, >3 values)", good, len(ids)
    )


if __name__ == "__main__":
    app()
