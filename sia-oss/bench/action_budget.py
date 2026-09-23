#!/usr/bin/env python3
"""Where do the actions go on the games the explorer actually clears?

Roughly 70% of the headroom the oracle exposes is not in the four games that
never clear (+0.73pp) but in the ones that already do, played slowly
(+1.83pp): tu93 takes 1463 actions for three levels where the oracle needs 47,
s5i5 1708 for one where it needs 13. RHAE is quadratic in actions per level, so
even a 3x speed-up on a game already won is worth more than a new game.

Prior accounting only ever covered games that never clear — the finding that 27%
of the budget went into repeating a single move. This accounts for every action
on the games that win: which branch chose it, which level it belonged to, and
whether it returned to a board already seen.
"""

from __future__ import annotations

import sys
from collections import Counter
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

from tgaer.agents.arc_agi3_explorer import (  # noqa: E402
    ExplorerArcAgi3Agent,
    field_box,
    frame_signature,
)
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    load_agent_class,
    play,
    require_starter,
)

LABELS = REPO / "sia-oss" / "bench" / "oracle"
# Games the explorer clears at the shipping budget.
CLEARS = "tu93,s5i5,ar25,sp80,ls20,m0r0,lp85"

app = typer.Typer(add_completion=False)


def oracle_cost(game_id: str) -> tuple[int, int]:
    """(actions, levels) the offline oracle needed; (0, 0) when unlabelled."""
    path = LABELS / f"{game_id}.npz"
    if not path.exists():
        return 0, 0
    d = np.load(path)
    return len(d["kind"]), len(set(d["level"].tolist()))


def account(game_id: str, max_steps: int, seed: int) -> dict[str, Any]:
    """Play one game, recording the branch and board identity of every action.

    `_choose` is split into its four sub-paths, because a high revisit rate does
    not by itself mean cycling: a board with several untested primitives is
    revisited once per primitive, which is correct breadth-first play.

      untested  a primitive never tried at this board
      plan      continuing a route already planned to a frontier
      frontier  starting a new route to the nearest board with something untested
      rotate    nothing untested and no frontier reachable — the stall pathology
    """
    arc = arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    branches: Counter[str] = Counter()
    per_level: Counter[int] = Counter()
    prims: Counter[Any] = Counter()
    # Only actions inside a *completed* level are scored, so the pathology has
    # to be measured per level, not over the run.
    lvl_seen: dict[int, set[Any]] = {}
    lvl_revisits: Counter[int] = Counter()
    lvl_prims: dict[int, set[Any]] = {}
    seen: set[Any] = set()
    revisits = 0
    stalls = [0]
    paths: Counter[str] = Counter()

    original = ExplorerArcAgi3Agent._choose

    def traced_choose(self: Any, sig: Any, prims: list[Any]) -> Any:
        had_plan = bool(self._plan and self._plan[0] in set(prims))
        untested = len(self._graph.untested_at(sig))
        before = self._stalls
        out = original(self, sig, prims)
        if had_plan:
            paths["plan"] += 1
        elif untested:
            paths["untested"] += 1
        elif self._stalls > before:
            paths["rotate"] += 1
        else:
            paths["frontier"] += 1
        return out

    def hook(step: int, observation: Any, env: Any, actor: Any) -> None:
        nonlocal revisits
        trace = getattr(actor, "trace", {}) or {}
        level = int((observation or {}).get("levels_completed", 0))
        branches[trace.get("branch", "?")] += 1
        per_level[level] += 1
        prims[str(trace.get("prim"))] += 1
        lvl_prims.setdefault(level, set()).add(str(trace.get("prim")))
        stalls[0] = int(getattr(actor, "_stalls", 0))
        frame = (observation or {}).get("frame") or []
        if frame:
            arr = np.asarray(frame[-1], dtype=np.int16)
            sig = frame_signature(arr, field_box(arr))
            if sig in seen:
                revisits += 1
            seen.add(sig)
            here = lvl_seen.setdefault(level, set())
            if sig in here:
                lvl_revisits[level] += 1
            here.add(sig)

    ExplorerArcAgi3Agent._choose = traced_choose
    try:
        row = play(
            load_agent_class(None, "explorer"),
            game_id,
            arc,
            None,
            max_steps,
            seed=seed,
            on_step=hook,
        )
    finally:
        ExplorerArcAgi3Agent._choose = original
    total = sum(branches.values())
    return {
        "levels": int(row.get("levels_completed", 0)),
        "actions": total,
        "branches": branches,
        "per_level": per_level,
        "revisit_rate": revisits / max(total, 1),
        "distinct_prims": len(prims),
        "top_prim_share": (prims.most_common(1)[0][1] / total) if total else 0.0,
        "stalls": stalls[0],
        "lvl_revisit": {lv: lvl_revisits[lv] / n for lv, n in per_level.items() if n},
        "lvl_prims": {lv: len(p) for lv, p in lvl_prims.items()},
        "choose_paths": paths,
    }


@app.command()
def main(
    games: str = typer.Option(CLEARS, help="Comma-separated game ids."),
    max_steps: int = typer.Option(6000),
    seed: int = typer.Option(0),
) -> None:
    """Account for every action on the games that win."""
    require_starter()
    for game_id in games.split(","):
        try:
            r = account(game_id, max_steps, seed)
        except Exception as exc:
            logger.error("{}: {}: {}", game_id, type(exc).__name__, exc)
            continue
        oa, ol = oracle_cost(game_id)
        waste = r["actions"] / oa if oa else 0.0
        logger.info(
            "{:6} {} level(s) in {:5} actions | oracle {:4} for {} → {:5.0f}x | "
            "revisited {:4.0%} | {:3} distinct prims, top one {:3.0%} | stalls {}",
            game_id,
            r["levels"],
            r["actions"],
            oa,
            ol,
            waste,
            r["revisit_rate"],
            r["distinct_prims"],
            r["top_prim_share"],
            r["stalls"],
        )
        logger.info("        branches {}", dict(r["branches"].most_common()))
        cp = r["choose_paths"]
        n = sum(cp.values()) or 1
        logger.info(
            "        _choose {}",
            "  ".join(
                f"{k} {cp[k]} ({cp[k] / n:.0%})"
                for k in ("untested", "plan", "frontier", "rotate")
            ),
        )
        for lv in sorted(r["per_level"]):
            tag = "stuck" if lv >= r["levels"] else "     "
            logger.info(
                "        level {} {} {:5} actions, revisited {:4.0%}, "
                "{:3} distinct prims",
                lv,
                tag,
                r["per_level"][lv],
                r["lvl_revisit"].get(lv, 0.0),
                r["lvl_prims"].get(lv, 0),
            )


if __name__ == "__main__":
    app()
