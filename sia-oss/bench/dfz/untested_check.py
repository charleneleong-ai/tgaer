"""Did the suite episodes that cleared a stuck level use (object, action) pairs the original run never tried there?"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import typer
from harness import EvalRun, Workdir, level_key
from replay import RecordedGame

BACKGROUND = 600
ARMS = ("stallsuite", "stallsuite_warm")

Pair = tuple[Any, ...]


class UntestedPairs:
    """(object, action) pairs: a click is keyed by the clicked component's colour and size."""

    @staticmethod
    def component(board: list[list[int]], r: int, c: int) -> tuple[int, int]:
        colour, seen, q = board[r][c], {(r, c)}, deque([(r, c)])
        while q:
            y, x = q.popleft()
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                inside = 0 <= ny < len(board) and 0 <= nx < len(board[0])
                if inside and (ny, nx) not in seen and board[ny][nx] == colour:
                    seen.add((ny, nx))
                    q.append((ny, nx))
        return colour, len(seen)

    @classmethod
    def pair(cls, pre: list[list[int]], event: dict[str, Any]) -> Pair:
        if click := RecordedGame.click(event):
            colour, size = cls.component(pre, click["y"], click["x"])
            return (
                ("click", "background")
                if size >= BACKGROUND
                else ("click", colour, size)
            )
        return ("key", event["action_name"])

    @classmethod
    def on_level(cls, events: list[dict[str, Any]], level: int) -> list[Pair]:
        out, pre = [], None
        for e in events:
            if (
                e["type"] == "action"
                and pre is not None
                and pre["level"] == level
                and e["action_name"] != "RESET"
            ):
                out.append(cls.pair(pre["board"], e))
            if e["type"] in ("initial", "action"):
                pre = e
        return out

    @classmethod
    def episode(
        cls, events: list[dict[str, Any]], level: int, tried: set[Pair]
    ) -> dict[str, Any]:
        used = cls.on_level(events, level)
        cleared = any(
            e["type"] == "action"
            and (
                e["level"] > level
                or e.get("level_completed")
                or e.get("state") == "WIN"
            )
            for e in events
        )
        novel = [p for p in used if p not in tried]
        return {
            "cleared": cleared,
            "actions": len(used),
            "novel_types": len(set(novel)),
            "novel_share": len(novel) / max(1, len(used)),
            "last5_novel": any(p not in tried for p in used[-5:]) if cleared else None,
            "tried": len(tried),
        }

    @classmethod
    def rows(cls, wd: Workdir) -> list[dict[str, Any]]:
        out = []
        for code, level in wd.stuck_levels():
            key = level_key(code, level)
            tried = set(cls.on_level(EvalRun.load(wd.eval25.events(code)), level))
            for arm in ARMS:
                for path in sorted(wd.run(arm).root.glob(f"**/{key}r*_events.jsonl")):
                    out.append(
                        {
                            "level": key,
                            "arm": arm,
                            **cls.episode(EvalRun.load(path), level, tried),
                        }
                    )
        return out


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def main(workdir: Path = typer.Option(..., envvar="DFZ_WORKDIR")) -> None:
    episodes = UntestedPairs.rows(Workdir(workdir))
    won = [r for r in episodes if r["cleared"]]
    lost = [r for r in episodes if not r["cleared"] and r["actions"]]
    print(f"{len(episodes)} episodes; cleared {len(won)}")
    print(
        f"cleared: uses any untested pair {sum(r['novel_types'] > 0 for r in won)}/{len(won)}; "
        f"winning stretch (last 5 actions) uses one {sum(bool(r['last5_novel']) for r in won)}/{len(won)}; "
        f"untested share of actions {mean([r['novel_share'] for r in won]):.0%}"
    )
    print(
        f"not cleared: uses any untested pair {sum(r['novel_types'] > 0 for r in lost)}/{len(lost)}; "
        f"untested share of actions {mean([r['novel_share'] for r in lost]):.0%}"
    )
    for r in won:
        print(
            f"  WON {r['level']} {r['arm']}: {r['actions']} actions, {r['novel_types']} untested pair types "
            f"({r['novel_share']:.0%} of actions), winning stretch novel={r['last5_novel']}, "
            f"original tried {r['tried']} types"
        )


if __name__ == "__main__":
    typer.run(main)
