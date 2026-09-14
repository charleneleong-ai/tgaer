"""The per-level W&B table shared by `supervisor.py` and `bench/measure.py`.

Both log a generation's game state under the same ``game_state`` key in the same
W&B project, so they have to agree on the columns — two 12-column tables with
different schemas do not compare in the UI. Keeping the names and the row
builder in one place is what makes a supervisor row and a bench row line up.
"""

from __future__ import annotations

from typing import Any

GAME_STATE_COLUMNS = [
    "game",
    "level",
    "passed",
    "reason",
    "level_score",
    "human_baseline_actions",
    "our_actions_median",
    "ratio_vs_baseline",
    "env_score",
    "cap",
    "weighted_efficiency",
    "levels_completed",
    "total_levels",
]


def game_state_rows(
    results: dict[str, Any], submission: dict[str, Any] | None
) -> list[list[Any]]:
    """One row per level, or a single row for a game that never cleared one.

    Keys are subscripted, not ``.get()``-ed: ``evaluate.py:grade_game`` emits
    every one of them on both its branches, including the error short-circuit,
    so a missing key is a schema change that should fail here rather than
    silently log a column of ``None``.
    """
    levels_by_game: dict[str, list[dict[str, Any]]] = {
        card["game"]: card.get("levels", [])
        for card in (submission or {}).get("scorecards", [])
    }
    rows: list[list[Any]] = []
    for detail in results["details"]:
        env = [
            detail["env_score"],
            detail["cap"],
            detail["weighted_efficiency"],
            detail["levels_completed"],
            detail["total_levels"],
        ]
        head = [detail["game"], detail["passed"], detail["reason"]]
        levels = levels_by_game.get(detail["game"], [])
        if not levels:
            rows.append([head[0], None, *head[1:], detail["level_score"], None, None, None, *env])
            continue
        rows.extend(
            [
                head[0],
                level["level"],
                *head[1:],
                level["level_score"],
                level["human_baseline_actions"],
                level["our_actions_median"],
                level["ratio_vs_baseline"],
                *env,
            ]
            for level in levels
        )
    return rows
