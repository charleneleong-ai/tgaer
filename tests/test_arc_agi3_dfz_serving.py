"""Serving-only bench kernels: workload sampling and notebook assembly."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sia-oss/bench/dfz"))

from serving_bench import (  # noqa: E402
    BENCH_MODULES,
    BENCHES,
    CELLS,
    LAUNCHER_PATH,
    REQUEST_KEYS,
    WORKLOAD_DATASET,
    BenchKernel,
    Workload,
)


def write_run(root: Path, games: int, turns: int) -> None:
    for g in range(games):
        rows = [
            {
                "messages": [{"role": "user", "content": f"g{g} t{t}"}],
                "tools": [],
                "tool_choice": "auto",
                "chat_template_kwargs": {},
                "event": "request",
                "analysis_step": t,
            }
            for t in range(turns)
        ]
        (root / f"g{g:03d}-x_p0_requests.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows)
        )


@pytest.fixture
def base() -> tuple[dict, dict]:
    cells = [
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": f"cell {i}",
        }
        for i in range(17)
    ]
    meta = {
        "id": "u/base",
        "title": "base",
        "code_file": "base.ipynb",
        "dataset_sources": ["d/a"],
    }
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, meta


class TestWorkload:
    def test_lanes_are_consecutive_turns_with_request_keys_only(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path, games=12, turns=30)
        lanes = Workload().sample(tmp_path)
        assert len(lanes) == 10 and len({lane["game"] for lane in lanes}) == 10
        for lane in lanes:
            steps = [
                int(r["messages"][0]["content"].split("t")[-1])
                for r in lane["requests"]
            ]
            assert steps == list(range(steps[0], steps[0] + 10)) and steps[0] >= 10
            assert all(set(r) == set(REQUEST_KEYS) for r in lane["requests"])

    def test_short_runs_and_few_games_do_not_fail(self, tmp_path: Path) -> None:
        write_run(tmp_path, games=3, turns=4)
        lanes = Workload().sample(tmp_path)
        assert len(lanes) == 3 and all(
            1 <= len(lane["requests"]) <= 4 for lane in lanes
        )


class TestBenchKernel:
    @pytest.mark.parametrize("bench", sorted(BENCHES))
    def test_notebook_keeps_setup_cells_and_ends_in_the_bench(
        self, base: tuple[dict, dict], bench: str
    ) -> None:
        nb = BenchKernel.notebook(base[0], bench)
        srcs = ["".join(c["source"]) for c in nb["cells"]]
        assert srcs[1:4] == ["cell 2", "cell 4", "cell 6"]
        assert srcs[4] == f"%%writefile {LAUNCHER_PATH}\ncell 12"
        modules = srcs[5:-1]
        assert [m.split("\n", 1)[0] for m in modules] == [
            f"%%writefile /kaggle/working/{m}" for m in BENCH_MODULES.get(bench, ())
        ]
        assert srcs[-1] == (CELLS / f"{bench}.py").read_text()
        assert base[0]["cells"][1]["source"] == "cell 1"  # base untouched

    def test_metadata_adds_the_workload_dataset(self, base: tuple[dict, dict]) -> None:
        meta = BenchKernel.metadata(base[1], "arc-x")
        assert meta["id"].endswith("/arc-x") and meta["code_file"] == "arc-x.ipynb"
        assert meta["dataset_sources"] == ["d/a", WORKLOAD_DATASET] and base[1][
            "dataset_sources"
        ] == ["d/a"]


@pytest.mark.parametrize("bench", sorted(BENCHES))
def test_cells_parse(bench: str) -> None:
    ast.parse((CELLS / f"{bench}.py").read_text())


def test_cli_adds_configs_dataset_and_jit_export(
    base: tuple[dict, dict], tmp_path: Path
) -> None:
    from serving_bench import app
    from typer.testing import CliRunner

    nb, meta = base
    src = tmp_path / "base"
    src.mkdir()
    (src / "base.ipynb").write_text(json.dumps(nb))
    (src / "kernel-metadata.json").write_text(json.dumps(meta))
    cfg = '[["stock", {}]]'
    args = [
        "kernel",
        "specbench",
        str(src),
        str(tmp_path / "out"),
        "arc-x",
        "--configs",
        cfg,
        "--dataset",
        "u/d",
        "--export-jit",
    ]
    assert CliRunner().invoke(app, args).exit_code == 0
    out = json.loads((tmp_path / "out" / "arc-x.ipynb").read_text())
    srcs = ["".join(c["source"]) for c in out["cells"]]
    assert srcs[-3] == f"%%writefile /kaggle/working/bench_configs.json\n{cfg}"
    assert "jitcache.tar" in srcs[-1] and srcs[-2].startswith("# ruff: noqa: F821")
    assert (
        json.loads((tmp_path / "out" / "kernel-metadata.json").read_text())[
            "dataset_sources"
        ][-1]
        == "u/d"
    )
