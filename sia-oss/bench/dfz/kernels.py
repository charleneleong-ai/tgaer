"""Build the stuck-level suite Kaggle kernels from the base eval25 notebook, plus their warm-start datasets."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from candidates import CANDIDATES
from harness import (
    BASE_KERNEL,
    KAGGLE_USER,
    STALL_REPEATS,
    Harness,
    Workdir,
    parse_level_key,
)
from messages import MARK, Messages
from replay import RecordedGame
from warm_start import WarmStart

HERE = Path(__file__).resolve().parent
DFZ_DIR = "/kaggle/working/dfz"
DATASETS_ROOT = "/kaggle/input/datasets"
BASE_SLUG = "arc-agi-3-dfranzen-m2-eval25"
SUITE_SLUG = "arc-agi-3-dfranzen-m2-stallsuite"
CANDIDATE_SLUG = "arc-agi-3-dfz-warm"
WARM_DATASET = f"{KAGGLE_USER}/arc3-stall-warmstart"
WINREC_DATASET = f"{KAGGLE_USER}/arc3-stall-warmstart-winrec"
FIRST_OPENER = "No previous sequence has been executed yet"
HOOK_ANCHOR = "bm.solver.concurrency ="
GAMES_ANCHOR = "    bm.games = _offline_games(competition_env_files)\n"
SUITE_START = "# --- Stuck-level suite (experiment) ---"
SUITE_GAMES_START = "    # --- Stuck-level suite (experiment): each episode starts on a replayed stuck level ---"
MAX_SLUG = 50
NEW_CELL = {
    "cell_type": "code",
    "metadata": {"jupyter": {"source_hidden": True}, "trusted": True},
    "outputs": [],
    "execution_count": None,
}

WORKDIR = typer.Option(
    ..., envvar="DFZ_WORKDIR", help="Holds run outputs, stall verdicts and kernel dirs."
)
DFZ_SRC = typer.Option(
    ..., envvar="DFZ_SRC", help="Extracted da-fr/arc-agi-3-solution repo."
)

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class Features:
    """Which experiment hooks a suite kernel carries; candidate and winrec both imply warm start."""

    warm: bool = False
    winrec: bool = False
    candidate: str | None = None

    @property
    def is_warm(self) -> bool:
        return self.warm or self.winrec or self.candidate is not None

    @property
    def dataset(self) -> str | None:
        return (
            (WINREC_DATASET if self.winrec else WARM_DATASET) if self.is_warm else None
        )

    @property
    def slug(self) -> str:
        if self.candidate:
            base = f"{CANDIDATE_SLUG}-{self.candidate.replace('_', '-')}"
        else:
            base = SUITE_SLUG + ("-warm" if self.is_warm else "")
        return base + ("-winrec" if self.winrec else "")

    @property
    def dirname(self) -> str:
        base = (
            f"kaggle_gepa_{self.candidate}"
            if self.candidate
            else "kaggle_dfranzen_stallsuite" + ("_warm" if self.is_warm else "")
        )
        return base + ("_winrec" if self.winrec else "")

    @property
    def modules(self) -> list[str]:
        mods = ["harness", "level_start"]
        mods += ["messages", "warm_start"] if self.is_warm else []
        mods += ["win_records"] if self.winrec else []
        mods += ["candidates"] if self.candidate else []
        return mods


class SuiteHooks:
    """The code each feature appends to the customization-hook cell and the games cell."""

    @staticmethod
    def suite_levels(wd: Workdir) -> list[dict[str, Any]]:
        out = []
        for code, level in wd.stuck_levels():
            game = RecordedGame(wd.eval25.events(code))
            out.append(
                {"game": game.game_id, "level": level, "prefix": game.prefix(level)}
            )
        return sorted(out, key=lambda s: s["game"])

    @staticmethod
    def suite(dfz_dir: str) -> str:
        return (
            f"\n\n{SUITE_START}\n"
            f"sys.path.insert(0, {dfz_dir!r})\n"
            "STALL_SUITE = True\n"
            f"STALL_REPEATS = {STALL_REPEATS}\n"
            "if STALL_SUITE and not TRUE_SUBMISSION:\n"
            "    bm.solver.concurrency = 10\n"
            "    bm.solver.max_runtime_s_per_game = 15 * 60\n"
        )

    @staticmethod
    def suite_games(blob: str) -> str:
        return (
            f"\n{SUITE_GAMES_START}\n"
            "    if STALL_SUITE:\n"
            "        import json, taaf.game_api\n"
            "        from harness import episode_id\n"
            "        from level_start import LevelStartGameAPI\n"
            f"        STALL_LEVELS = json.loads({blob!r})\n"
            "        _spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=competition_env_files)\n"
            "        bm.games = [\n"
            "            LevelStartGameAPI(env_name=s['game'], arcade_spec=_spec, external_game_id=episode_id(s['game'][:4], s['level'], r),\n"
            "                              prefix=tuple((a, d) for a, d in s['prefix']))\n"
            "            for r in range(STALL_REPEATS) for s in STALL_LEVELS\n"
            "        ]\n"
            "        print('stall suite active:', len(bm.games), 'episodes')\n"
        )

    @staticmethod
    def warm(dataset_dir: str, n_levels: int) -> str:
        return (
            "\n\n# --- Warm start (experiment): seed each suite episode with the history it had on that level ---\n"
            "if STALL_SUITE and not TRUE_SUBMISSION:\n"
            "    import inference.agent.tool_agent as _tool_agent\n"
            "    from warm_start import WarmStart\n"
            f"    _warm_dir = Path({dataset_dir!r})\n"
            "    _warm = {p.stem: json.loads(p.read_text()) for p in _warm_dir.glob('*-L*.json')}\n"
            f"    if len(_warm) != {n_levels}:\n"
            "        raise RuntimeError(f'warm-start data missing: found {len(_warm)} records under {_warm_dir}')\n"
            "    WarmStart.install(_tool_agent.ToolAgent, _warm)\n"
            "    print('warm start active:', len(_warm), 'levels')\n"
        )

    @staticmethod
    def winrec() -> str:
        return (
            "\n\n# --- Pinned level-win records (experiment) ---\n"
            "if True:\n"
            "    import inference.agent.tool_agent as _tool_agent\n"
            "    from win_records import WinRecords\n"
            "    WinRecords(_tool_agent._render_auto_frame_diff, _tool_agent._AUTO_FRAME_DIFF_MAX_GROUP).install(_tool_agent.ToolAgent)\n"
            "    print('win records active')\n"
        )

    @staticmethod
    def candidate(name: str) -> str:
        return (
            f"\n\n# --- Prompt candidate {name} (GEPA screen; general procedure only) ---\n"
            "if True:\n"
            "    import inference.agent.tool_agent as _tool_agent\n"
            "    from candidates import CANDIDATES, install\n"
            f"    install(_tool_agent, CANDIDATES[{name!r}])\n"
            f"    print('prompt candidate active: {name}', len(CANDIDATES[{name!r}]), 'chars')\n"
        )


class Kernel:
    """The base eval25 notebook plus metadata, rebuilt in one pass with a set of features."""

    def __init__(self, base_dir: Path, base_slug: str = BASE_SLUG) -> None:
        self.notebook = json.loads((base_dir / f"{base_slug}.ipynb").read_text())
        self.meta = json.loads((base_dir / "kernel-metadata.json").read_text())

    @staticmethod
    def check(src: str, name: str) -> None:
        compile(src, name, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    def find(self, needle: str) -> int:
        hits = [
            i
            for i, c in enumerate(self.notebook["cells"])
            if c["cell_type"] == "code" and needle in "".join(c["source"])
        ]
        if len(hits) != 1:
            raise LookupError(
                f"expected one code cell containing {needle!r}, found {len(hits)}"
            )
        return hits[0]

    def append(self, needle: str, transform: Callable[[str], str]) -> None:
        i = self.find(needle)
        src = transform("".join(self.notebook["cells"][i]["source"]))
        self.check(src, f"cell{i}")
        self.notebook["cells"][i]["source"] = src.splitlines(keepends=True)

    def ship(self, modules: list[str], dfz_dir: str, before: str) -> None:
        mkdir = f"import os\nos.makedirs({dfz_dir!r}, exist_ok=True)\n"
        cells = [mkdir]
        for mod in modules:
            body = (HERE / f"{mod}.py").read_text()
            self.check(body, mod)
            cells.append(f"%%writefile {dfz_dir}/{mod}.py\n{body}")
        at = self.find(before)
        self.notebook["cells"][at:at] = [
            dict(NEW_CELL, source=src.splitlines(keepends=True)) for src in cells
        ]

    def write(self, dst: Path, slug: str, dataset: str | None) -> Path:
        assert len(slug) <= MAX_SLUG, (
            f"Kaggle kernel slugs are capped at {MAX_SLUG} chars: {slug}"
        )
        dst.mkdir(exist_ok=True)
        (dst / f"{slug}.ipynb").write_text(json.dumps(self.notebook, indent=1))
        self.meta.update(
            id=f"{KAGGLE_USER}/{slug}", title=slug, code_file=f"{slug}.ipynb"
        )
        if dataset:
            self.meta["dataset_sources"] = [*self.meta["dataset_sources"], dataset]
        (dst / "kernel-metadata.json").write_text(json.dumps(self.meta, indent=2))
        return dst

    @classmethod
    def build(
        cls,
        wd: Workdir,
        features: Features,
        dfz_dir: str = DFZ_DIR,
        datasets_root: str = DATASETS_ROOT,
    ) -> Kernel:
        kernel = cls(wd.kernel_dir(BASE_KERNEL))
        levels = SuiteHooks.suite_levels(wd)
        blob = json.dumps(levels, separators=(",", ":"))
        hook = SuiteHooks.suite(dfz_dir)
        if features.is_warm:
            hook += SuiteHooks.warm(f"{datasets_root}/{features.dataset}", len(levels))
        if features.winrec:
            hook += SuiteHooks.winrec()
        if features.candidate:
            hook += SuiteHooks.candidate(features.candidate)
        kernel.append(HOOK_ANCHOR, lambda src: src + hook)
        games = GAMES_ANCHOR + "    import arc_agi\n" + SuiteHooks.suite_games(blob)
        kernel.append(GAMES_ANCHOR, lambda src: src.replace(GAMES_ANCHOR, games, 1))
        kernel.ship(features.modules, dfz_dir, before=HOOK_ANCHOR)
        return kernel


class WinRecordWarmData:
    """Adds to each warm-start record the win records a win-record run would have carried into that level."""

    def __init__(self, src: Path) -> None:
        Harness.install(src)
        import inference.agent.tool_agent as ta  # both need the harness on sys.path first
        from win_records import WinRecords

        self.hook = WinRecords(
            ta._render_auto_frame_diff, ta._AUTO_FRAME_DIFF_MAX_GROUP
        )

    def extend(self, rec: dict[str, Any], events: Path, level: int) -> int:
        history = self.hook.history_from_events(events)
        wins = {k: r for k in range(1, level) if (r := self.hook.record(history, k))}
        if not wins:
            return 0
        rec["win_records"] = {str(k): v for k, v in wins.items()}
        if level - 1 in wins:
            rec["opener"] = Messages.append_text(
                rec["opener"], Messages.block({level - 1: wins[level - 1]})
            )
        head = rec["history"][0] if rec["history"] else None
        # dfranzen's first-opener wording, not equality with the first request: tn36 differs
        if head is not None and FIRST_OPENER not in Messages.strip(json.dumps(head)):
            rec["history"][0] = Messages.pin(head, wins)
        return len(wins)


@app.command()
def kernel(
    workdir: Path = WORKDIR,
    warm: bool = typer.Option(
        False, help="Seed each episode with its recorded history."
    ),
    winrec: bool = typer.Option(
        False, help="Pin level-win records (uses the winrec warm dataset)."
    ),
    candidate: str | None = typer.Option(
        None, help="Prompt candidate name; implies --warm."
    ),
    out: Path | None = None,
) -> None:
    """Suite kernel with the chosen hooks, built from the base eval25 notebook."""
    if candidate is not None and candidate not in CANDIDATES:
        raise typer.BadParameter(
            f"unknown candidate {candidate!r}; known: {sorted(CANDIDATES)}"
        )
    wd, features = Workdir(workdir), Features(warm, winrec, candidate)
    built = Kernel.build(wd, features)
    print(
        "built",
        features.slug,
        "->",
        built.write(
            out or wd.kernel_dir(features.dirname), features.slug, features.dataset
        ),
    )


@app.command("warm-data")
def warm_data(workdir: Path = WORKDIR, out: Path | None = None) -> None:
    """One warm-start JSON per stuck level, from the unmodified run's request logs."""
    wd = Workdir(workdir)
    out = out or wd.warm_data()
    for code, level in wd.stuck_levels():
        rec = WarmStart.record(
            WarmStart.first_request(
                wd.eval25.requests(code), wd.eval25.events(code), level
            )
        )
        path = out / f"{code}-L{level}.json"
        path.write_text(json.dumps(rec))
        first = rec["history"][0]["role"] if rec["history"] else "-"
        print(
            f"{code}-L{level}: {len(rec['history'])} msgs, opener {rec['opener']['role']}, first {first}, "
            f"kept {sorted(rec['kept'])[:4]}, {path.stat().st_size // 1024} KB"
        )


@app.command("winrec-data")
def winrec_data(
    workdir: Path = WORKDIR, src: Path = DFZ_SRC, out: Path | None = None
) -> None:
    """Warm-start records carrying the win records a win-record run would have had on each stuck level."""
    wd, builder = Workdir(workdir), WinRecordWarmData(src)
    out = out or wd.warm_data(winrec=True)
    out.mkdir(exist_ok=True)
    for path in sorted(wd.warm_data().glob("*-L*.json")):
        rec = json.loads(path.read_text())
        code, level = parse_level_key(path.stem)
        wins = builder.extend(rec, wd.eval25.events(code), level)
        (out / path.name).write_text(json.dumps(rec))
        pinned = bool(wins) and MARK in json.dumps(rec["history"][:1])
        print(f"{path.stem}: {wins} win records, head pinned: {pinned}")
    meta = {
        "title": WINREC_DATASET.split("/")[1],
        "id": WINREC_DATASET,
        "licenses": [{"name": "other"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    app()
