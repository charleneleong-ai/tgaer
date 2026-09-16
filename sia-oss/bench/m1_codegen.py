#!/usr/bin/env python3
"""M1: ask a model for lp85's world model, then score it on recorded play.

The closed LLM line (`feat/arc-agi3-llm-codegen`) asked for
``policy(frame) -> action`` and had to judge whether a *choice* was good —
subjective, six validator designs, `0 judged` on every game. This asks for three
functions instead, each with an objective test:

    simulate(state, action) -> state   exact next-state prediction on held-out data
    is_goal(state) -> bool             fires on recorded wins, nowhere else
    distance(state) -> float           greedy descent on it must actually reach a goal

M0 established why `distance` is the interesting one: over a *perfect* simulator,
uninformed BFS still failed (40k expansions, depth 6), while adding a
goal-distance heuristic solved lp85 L2 in 48 expansions. The world model was
never the constraint.

State is object-level — each component as (colour, row, col, height, width,
pixels) — which `m1_markov.py` confirmed is Markov for lp85: 119 distinct
(state, action) pairs over 1500 transitions, zero ambiguous.
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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
from tgaer.agents.arc_agi3_grid import components  # noqa: E402
from tgaer.agents.arc_agi3_kaggle import HTTPChatBackend  # noqa: E402
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)
os.environ.setdefault("ARC_TGAER_REPO", str(REPO))

from arcengine import ActionInput, GameAction  # noqa: E402

app = typer.Typer(add_completion=False)

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
SCALE = 3  # lp85 sprites are 3x3 blocks; a stride-3 probe cannot miss a button


PROMPT = """You are given recorded play from a deterministic puzzle game.

The board is described as a list of objects. Each object is a 6-tuple:
    (colour, row, col, height, width, pixels)
`row`/`col` are the top-left corner. Objects never overlap.

There are {n_actions} distinct actions, numbered 0..{max_action}. Each is a
button press that rearranges objects. The mapping is fixed for the whole level.

The board starts in this state:

{start}

Here are {n_train} recorded transitions. Sending every full board would be
mostly repetition, so each line gives the action and only the objects that
actually changed, as `before -> after`. Objects not listed were untouched.

{transitions}

{goal_note}

{task}

Do NOT hardcode the transitions above as a lookup table. The code must work on
boards it has not seen. Keep it short — well under 200 lines.

Use only the standard library. `import numpy as np` is available if you want it.
"""

# Asked for separately. A single request for all three spent its whole 8000-token
# budget on `simulate` twice over and never reached the other two, and `simulate`
# is the part we least need: M0 already has a perfect simulator by deepcopy, and
# showed the heuristic is what actually gates search.
TASK_GOAL = """Write exactly two Python functions, in one ```python block, with
no explanation:

1. `is_goal(state)` - return True only for a solved board.

2. `distance(state)` - return a number that is 0 exactly when `is_goal(state)`
   is True and gets smaller as the board gets closer to solved. It must be
   graded, not a count of unsatisfied conditions: a value that takes only two or
   three distinct values gives a search no gradient to follow and is useless.
   Prefer something like the summed distance from each movable object to where
   it needs to end up."""

TASK_SIMULATE = """Write exactly one Python function, in one ```python block,
with no explanation:

`simulate(state, action)` - `state` is a list of 6-tuples as above, `action` is
an int. Return the next state in the same representation, sorted the same way.
Work out the rule each action applies to the objects and implement that rule."""


def grid(fd: Any) -> np.ndarray:
    return np.asarray(fd.frame[-1], dtype=np.int16)


def click(game: Any, rc: tuple[int, int]) -> Any:
    r, c = rc
    return game.perform_action(
        ActionInput(id=GameAction.ACTION6, data={"x": int(c), "y": int(r)}), raw=True
    )


def background(arr: np.ndarray) -> int:
    """The board's floor colour, taken once and then held fixed."""
    return int(np.bincount(arr.ravel()).argmax())


def objects(arr: np.ndarray, bg: int) -> list[tuple[int, ...]]:
    """The board as non-background components, the representation M1 hands over.

    ``bg`` is passed in rather than recomputed per frame. Taking the modal colour
    of each frame makes the representation non-stationary: on lp85 colour 4 is
    the floor on an unsolved board and so never appears, but a solved board is
    mostly one 41x41 colour-4 region, which shifts the mode and makes colour 4
    materialise as an object. A model given that data inferred — correctly, from
    what it could see — that "a 41x41 colour-4 object exists" means solved, then
    wrote a `distance` referring to colour-4 objects that are absent from every
    unsolved state, so it returned infinity everywhere and had no gradient at all.
    """
    out: list[tuple[int, ...]] = []
    for v in (int(x) for x in np.unique(arr)):
        if v == bg:
            continue
        for comp in components(arr, (v,)):
            rs, cs = comp[:, 0], comp[:, 1]
            out.append(
                (
                    v,
                    int(rs.min()),
                    int(cs.min()),
                    int(rs.max() - rs.min() + 1),
                    int(cs.max() - cs.min() + 1),
                    len(comp),
                )
            )
    return sorted(out)


def button_cells(game: Any, base: np.ndarray) -> list[tuple[int, int]]:
    """One representative click per distinct effect, probed on a stride-3 lattice."""
    seen: dict[bytes, tuple[int, int]] = {}
    for r in range(0, 64, SCALE):
        for c in range(0, 64, SCALE):
            after = grid(click(copy.deepcopy(game), (r, c)))
            if np.array_equal(after, base):
                continue
            seen.setdefault(after.tobytes(), (r, c))
    return list(seen.values())


def collect(game: Any, base: np.ndarray, limit: int, seed: int) -> dict[str, Any]:
    """Random walk the level, recording object-level transitions and any win."""
    cells = button_cells(game, base)
    bg = background(base)  # fixed for the level; see `objects`
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    wins: list[list[tuple[int, ...]]] = []
    node, state = copy.deepcopy(game), objects(base, bg)
    for _ in range(limit):
        idx = rng.randrange(len(cells))
        child = copy.deepcopy(node)
        out = click(child, cells[idx])
        nxt = objects(grid(out), bg)
        rows.append({"state": state, "action": idx, "next_state": nxt})
        won = int(getattr(out, "levels_completed", 0)) > 0
        if won:
            wins.append(nxt)
        if won or "GAME_OVER" in str(getattr(out, "state", "")):
            node, state = copy.deepcopy(game), objects(base, bg)
        else:
            node, state = child, nxt
    return {"transitions": rows, "cells": cells, "wins": wins}


def as_diff(row: dict[str, Any]) -> str:
    """One transition as `action N: before -> after` over changed objects only.

    Full board pairs are ~95% repetition: 60 of them came to 29,769 input tokens
    against a 32,768 context. Only a handful of objects move per press, and the
    diff is also the clearer statement of what an action *does*.
    """
    before = {tuple(o) for o in row["state"]}
    after = {tuple(o) for o in row["next_state"]}
    gone = sorted(before - after)
    fresh = sorted(after - before)
    if not gone and not fresh:
        return f"action {row['action']}: (no change)"
    pairs = ", ".join(
        f"{list(g)} -> {list(f)}" for g, f in zip(gone, fresh, strict=False)
    )
    # Unequal counts mean objects appeared or vanished, not just moved.
    if len(gone) != len(fresh):
        pairs = f"removed {[list(g) for g in gone]}, added {[list(f) for f in fresh]}"
    return f"action {row['action']}: {pairs}"


def build_prompt(
    rows: list[dict[str, Any]],
    start: list[Any],
    n_actions: int,
    wins: list[Any],
    task: str,
) -> str:
    """The request: the start board in full, transitions as diffs, and a win.

    Showing a winning board is not optional. Told only that wins existed, the
    model wrote `# The solved board is the initial board` and hardcoded the
    start state as the target — exactly inverted, because the start was the only
    complete board it had ever been shown.
    """
    if wins:
        example = "\n".join(
            f"  {list(o)}"
            for o in sorted(set(map(tuple, wins[0])) - set(map(tuple, start)))
        )
        goal_note = (
            f"{len(wins)} of the recorded transitions reached the solved board. "
            "Here are the objects that a solved board has which the starting "
            f"board does not:\n\n{example}\n\n"
            "`is_goal` must return True for boards like that and False for the "
            "starting board."
        )
    else:
        goal_note = (
            "None of these transitions reached a solved board, so infer the goal "
            "from the board's structure: some objects are movable and others mark "
            "where they belong. The starting board is NOT solved."
        )
    return PROMPT.format(
        n_actions=n_actions,
        max_action=n_actions - 1,
        n_train=len(rows),
        start="\n".join(f"  {list(o)}" for o in start),
        transitions="\n".join(as_diff(r) for r in rows),
        goal_note=goal_note,
        task=task,
    )


def token_count(base_url: str, model: str, text: str) -> int | None:
    """Exact prompt length from vLLM's own tokenizer, or None if unavailable.

    Worth the round trip: a char/4 rule of thumb is off by ~3x on this content,
    which is dense bracketed integers and tokenizes at roughly 1.2 chars/token.
    Guessing produced three silent HTTP 400s.
    """
    payload = json.dumps({"model": model, "prompt": text}).encode()
    url = base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(json.load(resp)["count"])
    except Exception as exc:
        logger.warning("tokenize unavailable ({}), falling back to chars/2", exc)
        return None


def fit_prompt(
    rows: list[dict[str, Any]],
    start: list[Any],
    n_actions: int,
    wins: list[Any],
    task: str,
    base_url: str,
    model: str,
    reserve: int,
    limit: int = 32768,
) -> tuple[str, int]:
    """The longest prompt that still leaves ``reserve`` tokens for the answer.

    Halves the transition count until it fits rather than trusting an estimate,
    and returns how many transitions survived so the caller can report it.
    """
    n = len(rows)
    while n > 0:
        prompt = build_prompt(rows[:n], start, n_actions, wins, task)
        used = token_count(base_url, model, prompt)
        if used is None:
            used = len(prompt) // 2
        if used + reserve <= limit:
            return prompt, n
        n = n * 2 // 3
    raise ValueError("no transition count fits the context window")


def extract(text: str) -> str | None:
    """The last fenced python block, tolerating a missing closing fence.

    A reply cut off by `max_tokens` has an opening fence and no closing one, and
    a strict pattern throws away code that is very nearly complete — three
    generations were discarded that way before this fell back to "everything
    after the last opening fence".
    """
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return blocks[-1].strip()
    opened = re.split(r"```(?:python)?\s*\n", text)
    return opened[-1].strip() if len(opened) > 1 else None


def load(code: str, required: tuple[str, ...]) -> dict[str, Any] | None:
    """Exec the generated module and return its namespace, or None if unusable."""
    ns: dict[str, Any] = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102
    except Exception as exc:
        logger.error("generated code failed to import: {}", exc)
        return None
    missing = [n for n in required if not callable(ns.get(n))]
    if missing:
        logger.error("generated code is missing {}", missing)
        return None
    return ns


def norm(state: Any) -> list[tuple[int, ...]] | None:
    """Coerce a generated state back to the comparable representation."""
    try:
        return sorted(tuple(int(x) for x in obj) for obj in state)
    except Exception:
        return None


def score_simulate(ns: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = errors = 0
    for row in rows:
        try:
            got = norm(ns["simulate"]([tuple(o) for o in row["state"]], row["action"]))
        except Exception:
            errors += 1
            continue
        if got is not None and got == norm(row["next_state"]):
            exact += 1
    return {
        "n": len(rows),
        "exact": exact,
        "errors": errors,
        "accuracy": exact / max(len(rows), 1),
    }


def score_is_goal(
    ns: dict[str, Any], rows: list[dict[str, Any]], wins: list[Any]
) -> dict[str, Any]:
    def fires(state: Any) -> bool:
        try:
            return bool(ns["is_goal"]([tuple(o) for o in state]))
        except Exception:
            return False

    non_win = [r["next_state"] for r in rows]
    return {
        "true_positive": sum(fires(w) for w in wins),
        "wins": len(wins),
        "false_positive": sum(fires(s) for s in non_win),
        "non_wins": len(non_win),
    }


def score_distance_only(
    ns: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """How graded is the generated `distance`, independent of any simulator.

    M0's finding in one number: a heuristic taking two or three values leaves
    greedy search with no gradient, however correct its zero set is.
    """
    values = []
    for row in rows[:200]:
        try:
            values.append(float(ns["distance"]([tuple(o) for o in row["state"]])))
        except Exception:
            pass
    return {"distinct_values": len(set(values)), "sampled": len(values)}


def score_distance(
    ns: dict[str, Any], rows: list[dict[str, Any]], n_actions: int, budget: int
) -> dict[str, Any]:
    """Does greedy descent on the generated distance, through the generated
    simulate, actually reach the generated goal? The end-to-end question."""
    values = []
    for row in rows[:200]:
        try:
            values.append(float(ns["distance"]([tuple(o) for o in row["state"]])))
        except Exception:
            pass
    distinct = len(set(values))
    start = [tuple(o) for o in rows[0]["state"]]
    state, seen, steps = start, set(), 0
    reached = False
    for _ in range(budget):
        try:
            if ns["is_goal"](state):
                reached = True
                break
        except Exception:
            break
        key = repr(norm(state))
        if key in seen:
            break
        seen.add(key)
        best, best_h = None, None
        for a in range(n_actions):
            try:
                nxt = ns["simulate"](state, a)
                h = float(ns["distance"](nxt))
            except Exception:
                continue
            if best_h is None or h < best_h:
                best, best_h = nxt, h
        if best is None:
            break
        state, steps = best, steps + 1
    return {
        "distinct_values": distinct,
        "sampled": len(values),
        "greedy_steps": steps,
        "greedy_reached_goal": reached,
    }


@app.command()
def main(
    level: int = typer.Option(1, help="lp85 level to model (1-indexed)."),
    samples: int = typer.Option(400, help="Transitions to record."),
    train: int = typer.Option(60, help="Transitions shown to the model."),
    base_url: str = typer.Option("http://127.0.0.1:8011/v1", help="vLLM endpoint."),
    model: str = typer.Option("qwen-27b", help="Served model name."),
    attempts: int = typer.Option(3, help="Generations to try."),
    temperature: float = typer.Option(0.3),
    max_tokens: int = typer.Option(8000),
    timeout: float = typer.Option(
        1800.0,
        help="Seconds per request. The backend defaults to 300, and a full "
        "max_tokens answer at ~25 tok/s on this 27B needs more than that.",
    ),
    dump: Path = typer.Option(None, help="Write the best generation here."),
) -> None:
    """Record lp85 play, ask the model for a world model, and score it."""
    require_starter()
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    env = arc.make("lp85")
    fd = env.reset()
    game = env._game

    # M0's plans are the cheapest way to stand at the start of a later level.
    if level > 1:
        logger.info("advancing to level {}", level)
        from m0_plan import solve  # noqa: PLC0415

        for _ in range(level - 1):
            plan, _, _ = solve(
                game, grid(fd), int(getattr(fd, "levels_completed", 0)), 60000
            )
            if plan is None:
                raise typer.Exit(f"could not reach level {level}")
            for rc in plan:
                fd = click(game, rc)

    logger.info("recording {} transitions on level {}", samples, level)
    data = collect(game, grid(fd), samples, seed=0)
    rows, cells, wins = data["transitions"], data["cells"], data["wins"]
    n_actions = len(cells)
    holdout = rows[train:]
    logger.info(
        "{} actions | {} transitions ({} shown, {} held out) | {} recorded wins",
        n_actions,
        len(rows),
        train,
        len(holdout),
        len(wins),
    )

    backend = HTTPChatBackend(base_url=base_url, model=model, seed=0, timeout=timeout)

    def generate(task: str, required: tuple[str, ...]) -> list[dict[str, Any]]:
        """Ask for one group of functions, `attempts` times, keeping what loads."""
        prompt, shown = fit_prompt(
            rows[:train],
            rows[0]["state"],
            n_actions,
            wins,
            task,
            base_url,
            model,
            reserve=max_tokens + 512,
        )
        if shown < train:
            logger.warning("trimmed {} -> {} transitions to fit", train, shown)
        out: list[dict[str, Any]] = []
        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                reply = backend.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except urllib.error.HTTPError as exc:
                # The body names the real reason (context overflow, bad param);
                # the bare status does not, and three 400s went undiagnosed for it.
                logger.error(
                    "{} attempt {}: HTTP {}: {}",
                    required[0],
                    attempt,
                    exc.code,
                    exc.read().decode()[:300],
                )
                continue
            except Exception as exc:
                logger.error("{} attempt {}: {}", required[0], attempt, exc)
                continue
            choice = reply["choices"][0]
            logger.info(
                "{} attempt {}: finish={} tokens={} in {:.0f}s",
                required[0],
                attempt,
                choice.get("finish_reason"),
                reply.get("usage", {}).get("completion_tokens"),
                time.monotonic() - t0,
            )
            code = extract(choice["message"]["content"])
            if not code:
                logger.error("{} attempt {}: no python block", required[0], attempt)
                continue
            ns = load(code, required)
            if ns is not None:
                out.append({"code": code, "ns": ns})
        return out

    # The semantic half first: M0 showed `distance` is what gates search, and
    # asking for all three at once spent the whole budget on `simulate`.
    goals = generate(TASK_GOAL, ("is_goal", "distance"))
    for i, cand in enumerate(goals, 1):
        goal = score_is_goal(cand["ns"], holdout, wins)
        dist = score_distance_only(cand["ns"], rows)
        cand["scores"] = {"is_goal": goal, "distance": dist}
        logger.info(
            "goal candidate {}: is_goal {}/{} wins, {}/{} false positives | "
            "distance {} distinct values over {} states",
            i,
            goal["true_positive"],
            goal["wins"],
            goal["false_positive"],
            goal["non_wins"],
            dist["distinct_values"],
            dist["sampled"],
        )

    sims = generate(TASK_SIMULATE, ("simulate",))
    for i, cand in enumerate(sims, 1):
        sim = score_simulate(cand["ns"], holdout)
        cand["scores"] = {"simulate": sim}
        logger.info(
            "simulate candidate {}: {}/{} exact ({:.0%}), {} errors",
            i,
            sim["exact"],
            sim["n"],
            sim["accuracy"],
            sim["errors"],
        )

    if not goals and not sims:
        logger.error("no usable generation in {} attempts per task", attempts)
        raise typer.Exit(1)

    # End to end: greedy over the generated distance, through the generated
    # simulate when there is one, else through the real game as M0 did.
    if goals:
        best_goal = max(goals, key=lambda c: c["scores"]["distance"]["distinct_values"])
        best_sim = (
            max(sims, key=lambda c: c["scores"]["simulate"]["accuracy"])
            if sims
            else None
        )
        if best_sim is not None:
            merged = {**best_sim["ns"], **best_goal["ns"]}
            end = score_distance(merged, rows, n_actions, budget=200)
            logger.info(
                "end to end (generated simulate + distance): greedy {} steps, goal={}",
                end["greedy_steps"],
                end["greedy_reached_goal"],
            )
        logger.success(
            "best distance: {} distinct values | is_goal {}/{} wins, {} false positives",
            best_goal["scores"]["distance"]["distinct_values"],
            best_goal["scores"]["is_goal"]["true_positive"],
            best_goal["scores"]["is_goal"]["wins"],
            best_goal["scores"]["is_goal"]["false_positive"],
        )
        if dump:
            dump.write_text(best_goal["code"])
            logger.info("wrote {}", dump)
    if sims:
        logger.success(
            "best simulate: {:.0%} exact on {} held-out transitions",
            max(c["scores"]["simulate"]["accuracy"] for c in sims),
            len(holdout),
        )


if __name__ == "__main__":
    app()
