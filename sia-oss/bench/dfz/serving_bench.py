"""Serving-only Kaggle kernels on dfranzen's SGLang stack: replay real ARC analyzer requests, play no games.

`workload` samples consecutive turns from a local run so replays keep real prefix-cache reuse; `kernel` turns the
base eval notebook into a bench whose last cell (from `serving_cells/`) boots the server per config.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

HERE = Path(__file__).resolve().parent
CELLS = HERE / "serving_cells"
KAGGLE_USER = "charyeezy"
WORKLOAD_DATASET = f"{KAGGLE_USER}/arc3-specbench-workload"
LAUNCHER_PATH = "/kaggle/launcher_cell.py"
REQUEST_KEYS = ("messages", "tools", "tool_choice", "chat_template_kwargs")
BASE_CELLS = (2, 4, 6)  # harness patch, environment, precaching
LAUNCHER_CELL = 12
BENCHES = {
    "specbench": "Speculative-decoding settings bench: boots dfranzen's server once per config and replays the same real ARC requests. Measures tok/s and accept length; plays no games.",
    "capture_probe": "Drafter capture probe (stage 1): boots dfranzen's server without speculative decoding and checks what `return_hidden_states` returns on 20 real ARC requests. Plays no games.",
    "drafter_stage3": "Drafter retraining (stage 3): captures the target's hyper-connection states on real ARC turns, retrains the MTP drafter's dense weights on them and exports mtp-dense.safetensors. Plays no games.",
}
BENCH_MODULES = {
    "drafter_stage3": ("mtp_ref.py", "drafter_train.py")
}  # written next to the notebook

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class Workload:
    """Lanes of consecutive turns from one local run, one lane per game."""

    lanes: int = 10
    turns: int = 10
    seed: int = 7

    def sample(self, run_dir: Path) -> list[dict[str, Any]]:
        rng = random.Random(self.seed)
        files = sorted(run_dir.glob("*_requests.jsonl"))
        out = []
        for f in rng.sample(files, min(self.lanes, len(files))):
            rows = [json.loads(line) for line in f.open()]
            lo = len(rows) // 3
            start = rng.randrange(lo, max(lo + 1, len(rows) - self.turns))
            reqs = [
                {k: r[k] for k in REQUEST_KEYS}
                for r in rows[start : start + self.turns]
            ]
            out.append({"game": f.name[:4], "requests": reqs})
        return out


class BenchKernel:
    """A copy of the base eval notebook that keeps the setup cells and ends in one bench cell."""

    @staticmethod
    def code(src: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src,
        }

    @staticmethod
    def notebook(base: dict[str, Any], bench: str) -> dict[str, Any]:
        nb = copy.deepcopy(base)
        cells = base["cells"]
        launcher = "".join(cells[LAUNCHER_CELL]["source"])
        nb["cells"] = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": f"# {bench}\n\n{BENCHES[bench]}",
            },
            *(cells[i] for i in BASE_CELLS),
            BenchKernel.code(f"%%writefile {LAUNCHER_PATH}\n{launcher}"),
            *(
                BenchKernel.code(
                    f"%%writefile /kaggle/working/{m}\n{(HERE / m).read_text()}"
                )
                for m in BENCH_MODULES.get(bench, ())
            ),
            BenchKernel.code((CELLS / f"{bench}.py").read_text()),
        ]
        return nb

    @staticmethod
    def metadata(base: dict[str, Any], slug: str) -> dict[str, Any]:
        meta = dict(
            base, id=f"{KAGGLE_USER}/{slug}", title=slug, code_file=f"{slug}.ipynb"
        )
        meta["dataset_sources"] = [*base["dataset_sources"], WORKLOAD_DATASET]
        return meta


@app.command()
def workload(run_dir: Path, out: Path) -> None:
    lanes = Workload().sample(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "workload.json").write_text(json.dumps(lanes))
    typer.echo({lane["game"]: len(lane["requests"]) for lane in lanes})


@app.command()
def kernel(bench: str, base_dir: Path, out_dir: Path, slug: str) -> None:
    if bench not in BENCHES:
        raise typer.BadParameter(f"bench must be one of {sorted(BENCHES)}")
    base_nb = json.loads(next(base_dir.glob("*.ipynb")).read_text())
    base_meta = json.loads((base_dir / "kernel-metadata.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.ipynb").write_text(
        json.dumps(BenchKernel.notebook(base_nb, bench), indent=1)
    )
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(BenchKernel.metadata(base_meta, slug), indent=2)
    )
    typer.echo(f"wrote {out_dir}")


if __name__ == "__main__":
    app()
