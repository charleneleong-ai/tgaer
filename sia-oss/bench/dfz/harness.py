"""Locations, episode ids and run readers shared by the dfranzen stuck-level tools (stdlib only: shipped to kernels)."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
HARNESS_PATHS = ("tufa-arc-agi-framework/src", "ARC3-Inference")
KAGGLE_USER = "charyeezy"
STALL_REPEATS = 2
LEVEL_KEY = re.compile(r"^(\w{4})-L(\d+)$")
EPISODE_ID = re.compile(r"^(\w{4})-L(\d+)r(\d+)$")
FINISHED = re.compile(
    r"\[finished\] (\w{4})-\S+ state=(\S+) level=(\d+)/(\d+) score=([\d.]+) actions=(\d+) tokens=(\d+) per-level=(\S+)"
)
RUNS = {
    "eval25": "dfranzen_eval25_output",
    "eval25_prior": "dfranzen_eval25_prior_output",
    "eval25_prior_fade": "dfranzen_eval25_prior_fade_output",
    "stallsuite": "dfranzen_stallsuite_output",
    "stallsuite_warm": "dfranzen_stallsuite_warm_output",
    "stallsuite_warm_rep2": "dfranzen_stallsuite_warm_rep2_output",
    "stallsuite_warm_winrec": "dfranzen_stallsuite_warm_winrec_output",
}
BASE_KERNEL = "kaggle_dfranzen_eval25"
WARM_DATA = "warmdata"
WINREC_DATA = "warmdata_winrec"


def level_key(code: str, level: int) -> str:
    return f"{code}-L{level}"


def episode_id(code: str, level: int, rep: int) -> str:
    return f"{level_key(code, level)}r{rep}"


def parse_level_key(key: str) -> tuple[str, int] | None:
    return (m.group(1), int(m.group(2))) if (m := LEVEL_KEY.match(key)) else None


def parse_episode_id(eid: str) -> tuple[str, int, int] | None:
    return (
        (m.group(1), int(m.group(2)), int(m.group(3)))
        if (m := EPISODE_ID.match(eid))
        else None
    )


class Harness:
    """dfranzen's extracted da-fr/arc-agi-3-solution repo, made importable."""

    @staticmethod
    def install(src: Path) -> None:
        for sub in HARNESS_PATHS:
            if (path := str(src / sub)) not in sys.path:
                sys.path.insert(0, path)
        os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "offline")

    @staticmethod
    def env_files() -> Path:
        return Path(os.environ.get("DFZ_ENV_FILES") or REPO / "environment_files")


@dataclass(frozen=True)
class Finished:
    """One `[finished]` line of a run log."""

    code: str
    state: str
    level: int
    n_levels: int
    score: float
    actions: int
    tokens: int
    per_level: str

    @property
    def won(self) -> bool:
        return self.state == "won"

    @property
    def level_pairs(self) -> list[tuple[int, int]]:
        return [
            (int(a), int(h))
            for a, h in (p.split("/") for p in self.per_level.split(","))
        ]


class EvalRun:
    """One downloaded dfranzen run output directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def events(self, code: str) -> Path:
        return next((self.root / "artifacts").glob(f"{code}-*_p0_events.jsonl"))

    def requests(self, code: str) -> Path:
        return next(self.root.glob(f"{code}-*_p0_requests.jsonl"))

    @cached_property
    def log_records(self) -> list[dict[str, Any]]:
        return json.loads(
            next(self.root.glob("arc-agi-3-dfranzen-m2-*.log")).read_text()
        )

    @cached_property
    def log_text(self) -> str:
        return "".join(r["data"] for r in self.log_records)

    def finished(self) -> dict[str, Finished]:
        return {
            g: Finished(
                g,
                state,
                int(lv),
                int(n),
                float(score),
                int(actions),
                int(tokens),
                per_level,
            )
            for g, state, lv, n, score, actions, tokens, per_level in FINISHED.findall(
                self.log_text
            )
        }

    def summary(self) -> str:
        return (self.root / "summary.txt").read_text()

    def summary_field(self, label: str) -> str:
        return re.search(rf"{re.escape(label)}:\s+([\d.]+)", self.summary()).group(1)

    @staticmethod
    def load(path: Path) -> list[dict[str, Any]]:
        with path.open() as f:
            return [json.loads(line) for line in f]


class Workdir:
    """The directory holding run outputs, stall verdicts, warm-start data and kernel dirs."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, name: str) -> EvalRun:
        return EvalRun(self.root / RUNS[name])

    @property
    def eval25(self) -> EvalRun:
        return self.run("eval25")

    def kernel_dir(self, name: str) -> Path:
        return self.root / name

    def warm_data(self, winrec: bool = False) -> Path:
        return self.root / (WINREC_DATA if winrec else WARM_DATA)

    def stuck_levels(self) -> list[tuple[str, int]]:
        return [
            (v["game"][:4], int(v["level"]))
            for f in sorted((self.root / "stall").glob("verdicts_*.json"))
            for v in json.loads(f.read_text())
        ]
