"""Replay dfranzen's priority scheduler over recorded per-level progress curves; it cannot value unseen turns."""

from __future__ import annotations

import heapq
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import typer
from harness import REPO, EvalRun, Finished, Workdir

LEVEL_PRIOR = {1: 30, 2: 54, 3: 51, 4: 54}
LEVEL_PRIOR_LATE = 96
BUDGET_S = 7254.0  # 532*60*25//110, the eval kernels' per-game share
EVAL25_WALL_S = 7340.0  # eval25 ran 2h02m40s with 10 slots
BASE_SLOTS = 10
SCHEDULER = "ARC3-Inference/inference/agent/priority_scheduler.py"
RHAE = REPO / "sia-oss/tasks/arc-agi3/data/public/evaluate.py"
PARK_S = 600


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GRADER = load_module("sia_evaluate", RHAE)


@dataclass
class Variant:
    name: str
    level_prior: bool = False
    fade_fraction: float = 0.2
    slots: int = BASE_SLOTS
    throughput_gain: float = 1.0


VARIANTS = {
    v.name: v
    for v in (
        Variant("stock"),
        Variant("prior", level_prior=True),
        Variant("prior_fade", level_prior=True, fade_fraction=0.75),
        Variant("prior_fade1", level_prior=True, fade_fraction=1.0),
        Variant("fade_only", fade_fraction=0.75),
        Variant(
            "prior_fade_s12",
            level_prior=True,
            fade_fraction=0.75,
            slots=12,
            throughput_gain=1.1,
        ),
    )
}


@dataclass(frozen=True)
class Curve:
    """A game's fixed progress curve: turns and actions needed per level."""

    gid: str
    baselines: list[int]
    level_turns: list[float]  # inf when never cleared
    level_actions: list[int]
    actions_per_turn: float
    tokens_per_turn: float

    @property
    def n_levels(self) -> int:
        return len(self.baselines)


@dataclass
class Game:
    curve: Curve
    level: int = 0
    turns_on_level: float = 0.0
    turns: int = 0
    last_turn_at: float = 0.0
    cleared_actions: list[int] = field(default_factory=list)

    @property
    def won(self) -> bool:
        return self.level == self.curve.n_levels

    def snapshot(self, ps: ModuleType) -> Any:
        return ps.PrioritySnapshot(
            level=self.level + 1,
            actions=int(self.turns_on_level * self.curve.actions_per_turn),
            tokens=self.turns_on_level * self.curve.tokens_per_turn,
            total_levels=self.curve.n_levels,
        )

    def play_turn(self) -> None:
        self.turns += 1
        self.turns_on_level += 1
        if self.turns_on_level >= self.curve.level_turns[self.level]:
            self.cleared_actions.append(self.curve.level_actions[self.level])
            self.level += 1
            self.turns_on_level = 0.0

    def score(self) -> float:
        levels = {
            i + 1: {"our_actions_median": max(1, a)}
            for i, a in enumerate(self.cleared_actions)
        }
        return (
            100
            * GRADER.environment_score(
                levels, [float(h) for h in self.curve.baselines]
            )["env_score"]
        )


class Curves:
    """Per-game curves from eval25's event logs, extended by levels other runs cleared; parsed once."""

    def __init__(self, base: EvalRun, others: list[EvalRun]) -> None:
        self.finished = base.finished()
        self.others = [o.finished() for o in others]
        self.curves = [
            self.curve(path)
            for path in sorted((base.root / "artifacts").glob("*_events.jsonl"))
        ]

    @staticmethod
    def turn_steps(path: Path) -> list[tuple[int, int]]:
        turns = {
            r["analysis_step"]: (r["level"], r["action_num"])
            for r in EvalRun.load(path)
            if r.get("analysis_step") is not None
        }
        return [turns[k] for k in sorted(turns)]

    def extra_actions(self, gid: str, i: int) -> int | None:
        found = (
            o[gid].level_pairs[i][0]
            for o in self.others
            if gid in o and o[gid].level > i
        )
        return next(found, None)

    def level_curve(
        self, fin: Finished, steps: list[tuple[int, int]], apt: float
    ) -> tuple[list[float], list[int]]:
        pairs = fin.level_pairs
        level_turns: list[float] = []
        level_actions: list[int] = []
        start = 0
        for i in range(len(pairs)):
            if i < fin.level:
                # a won game never reports a level past the last, so its final clear is the last turn
                end = next(
                    (j for j, (lv, _) in enumerate(steps) if lv > i + 1), len(steps) - 1
                )
                level_turns.append(float(end + 1 - start))
                level_actions.append(pairs[i][0])
                start = end + 1
                continue
            extra = self.extra_actions(fin.code, i)
            level_turns.append(extra / apt if extra else float("inf"))
            level_actions.append(extra or 0)
        return level_turns, level_actions

    def curve(self, path: Path) -> Curve:
        fin = self.finished[path.name[:4]]
        steps = self.turn_steps(path)
        apt = max(1.0, steps[-1][1] / len(steps))
        level_turns, level_actions = self.level_curve(fin, steps, apt)
        baselines = [h for _, h in fin.level_pairs]
        return Curve(
            fin.code,
            baselines,
            level_turns,
            level_actions,
            apt,
            fin.tokens / len(steps),
        )

    def games(self) -> list[Game]:
        return [Game(c) for c in self.curves]


@dataclass
class Simulator:
    """Allocates slot time over fixed curves with one scheduler variant."""

    ps: ModuleType
    curves: Curves
    stint: int
    budget: float
    slot_rate: float

    def priority(self, game: Game, variant: Variant, now: float) -> int:
        kwargs: dict[str, Any] = {
            "tail_fraction": max(0.0, self.budget - now)
            / (variant.fade_fraction * self.budget),
            "normalize_score": True,
            "tail_lookup": "remaining",
        }
        if variant.level_prior:
            kwargs["human_actions"] = float(
                LEVEL_PRIOR.get(game.level + 1, LEVEL_PRIOR_LATE)
            )
        return self.ps.priority_value(game.snapshot(self.ps), **kwargs)

    def run(self, variant: Variant) -> dict[str, Any]:
        games = self.curves.games()
        rate = self.slot_rate * variant.throughput_gain * BASE_SLOTS / variant.slots
        waiting = list(range(len(games)))
        running: list[
            tuple[float, int, int]
        ] = []  # (turn end time, game index, turns left in stint)
        now = 0.0

        def admit() -> None:
            while waiting and len(running) < variant.slots:
                best = max(
                    waiting,
                    key=lambda i: (
                        self.priority(games[i], variant, now),
                        -waiting.index(i),
                    ),
                )
                waiting.remove(best)
                heapq.heappush(
                    running,
                    (now + games[best].curve.tokens_per_turn / rate, best, self.stint),
                )

        admit()
        while running:
            now, i, left = heapq.heappop(running)
            if now > self.budget:
                break
            g = games[i]
            g.play_turn()
            g.last_turn_at = now
            if g.won:
                admit()
            elif left > 1:
                heapq.heappush(
                    running, (now + g.curve.tokens_per_turn / rate, i, left - 1)
                )
            else:
                waiting.append(i)
                admit()
        return self.result(games)

    def result(self, games: list[Game]) -> dict[str, Any]:
        idle = {g.curve.gid: self.budget - g.last_turn_at for g in games if not g.won}
        return {
            "score": sum(g.score() for g in games) / len(games),
            "won": sum(g.won for g in games),
            "parked": {gid: round(s / 60) for gid, s in idle.items() if s >= PARK_S},
        }


def main(
    workdir: Path = typer.Option(..., envvar="DFZ_WORKDIR"),
    src: Path = typer.Option(
        ..., envvar="DFZ_SRC", help="Extracted da-fr/arc-agi-3-solution repo."
    ),
    stint: int = 8,
    variants: str = ",".join(VARIANTS),
) -> None:
    wd = Workdir(workdir)
    curves = Curves(wd.eval25, [wd.run("eval25_prior_fade"), wd.run("eval25_prior")])
    slot_rate = float(wd.eval25.summary_field("total tokens")) / (
        BASE_SLOTS * EVAL25_WALL_S
    )
    sim = Simulator(
        load_module("dfz_priority", src / SCHEDULER), curves, stint, BUDGET_S, slot_rate
    )
    for name in variants.split(","):
        r = sim.run(VARIANTS[name])
        print(
            f"{name:16s} score {r['score']:6.2f}  won {r['won']:2d}  parked {r['parked']}"
        )


if __name__ == "__main__":
    typer.run(main)
