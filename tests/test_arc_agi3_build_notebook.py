"""Tests for the Kaggle notebook builder.

The kernel gets its code by having this builder copy module text into cells, so
a mistake here is invisible locally and only shows up as a submission that
scores zero. Three features have already shipped in this project built,
unit-tested and never executed; a staged package that cannot be imported is the
same failure with a bigger blast radius.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tgaer.evaluation import arc_agi3_build_notebook as bnb

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return bnb.build()


@pytest.fixture(scope="module")
def code_cells(notebook: dict) -> list[str]:
    return [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]


class TestPortedModules:
    """The explorer's stack is copied into the kernel rather than reimplemented."""

    def test_every_ported_module_exists_in_the_tree(self) -> None:
        missing = [m for m in bnb.PORTED_MODULES if not (REPO / "src" / m).exists()]
        assert not missing, f"builder would copy files that do not exist: {missing}"

    def test_each_module_gets_its_own_writefile_cell(
        self, code_cells: list[str]
    ) -> None:
        written = {
            c.split("\n", 1)[0].split(maxsplit=1)[1].strip()
            for c in code_cells
            if c.startswith("%%writefile")
        }
        for module in bnb.PORTED_MODULES:
            assert f"{bnb.KERNEL_PKG}/{module}" in written

    def test_module_bodies_are_copied_verbatim(self, code_cells: list[str]) -> None:
        """Copied, not re-implemented: the explorer is under active development
        and a hand-maintained second copy drifts out of date silently."""
        target = f"%%writefile {bnb.KERNEL_PKG}/tgaer/agents/arc_agi3_semantics.py\n"
        cell = next(c for c in code_cells if c.startswith(target))
        assert (
            cell[len(target) :]
            == (REPO / "src/tgaer/agents/arc_agi3_semantics.py").read_text()
        )

    def test_the_staging_cell_runs_before_any_module_is_written(
        self, code_cells: list[str]
    ) -> None:
        """%%writefile will not create parent directories, so the tree and its
        __init__.py files have to exist first."""
        staging = next(i for i, c in enumerate(code_cells) if "[pkg] staged" in c)
        first_module = next(
            i
            for i, c in enumerate(code_cells)
            if c.startswith(f"%%writefile {bnb.KERNEL_PKG}/")
        )
        assert staging < first_module

    def test_no_placeholder_survives_into_the_notebook(
        self, code_cells: list[str]
    ) -> None:
        """Cell bodies are plain strings full of braces, so the package path is
        substituted rather than f-string interpolated; an unsubstituted
        placeholder would be a NameError in the kernel."""
        assert not [c for c in code_cells if "__KERNEL_PKG__" in c]

    def test_the_preflight_imports_what_was_staged(self, code_cells: list[str]) -> None:
        """Writing the files proves nothing; importing them is the check."""
        preflight = [c for c in code_cells if "arc_agi3_explorer import" in c]
        assert preflight, "preflight must import the ported package"
        assert any("import numpy" in c for c in preflight), (
            "numpy is assumed present in the kernel rather than installed, so it "
            "must be asserted — installing pillow here once broke pillow"
        )


class TestStagedPackageImports:
    """The staged tree must actually import, not merely be written."""

    def test_the_explorer_imports_and_acts_from_the_staged_tree(
        self, code_cells: list[str]
    ) -> None:
        np = pytest.importorskip("numpy")
        staging = next(c for c in code_cells if "[pkg] staged" in c)
        with tempfile.TemporaryDirectory() as tmp:
            pkg = str(Path(tmp) / "kernel_pkg")
            exec(compile(staging.replace(bnb.KERNEL_PKG, pkg), "<staging>", "exec"), {})
            for cell in code_cells:
                if not cell.startswith(f"%%writefile {bnb.KERNEL_PKG}/"):
                    continue
                head, _, body = cell.partition("\n")
                dest = Path(
                    head.split(maxsplit=1)[1].strip().replace(bnb.KERNEL_PKG, pkg)
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(body)

            probe = (
                f"import sys; sys.path.insert(0, {pkg!r})\n"
                "import numpy as np\n"
                "from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent\n"
                "b = np.full((10, 10), 3); b[0,:] = b[-1,:] = b[:,0] = b[:,-1] = 4\n"
                "b[2,2] = 12\n"
                "a = ExplorerArcAgi3Agent().act({'frame': [b.tolist()],\n"
                "    'available_actions': [1,2,3,4], 'levels_completed': 0})\n"
                "print(a.id)\n"
            )
            # -E so an inherited PYTHONPATH cannot satisfy the import instead.
            out = subprocess.run(
                [sys.executable, "-E", "-c", probe],
                capture_output=True,
                text=True,
                cwd=tmp,
            )
        assert out.returncode == 0, out.stderr[-800:]
        assert out.stdout.strip().isdigit(), out.stdout
        assert np  # the fixture guard is the reason this test can run at all


class TestNotebookShape:
    def test_the_notebook_is_valid_json_with_cells(self, notebook: dict) -> None:
        assert json.loads(json.dumps(notebook))["cells"]

    def test_the_header_points_at_the_real_source_paths(self, notebook: dict) -> None:
        """The paths moved out of the starter checkout; a stale header sends the
        next reader to a file that no longer exists."""
        header = notebook["cells"][0]["source"]
        assert "src/tgaer/agents/arc_agi3_kaggle.py" in header
        assert "agent/my_agent.py" not in header


class TestAgentSelection:
    """Which agent plays is chosen at build time and must reach both runners."""

    def test_the_explorer_is_registered_so_main_py_can_be_pointed_at_it(
        self, code_cells: list[str]
    ) -> None:
        registry = [c for c in code_cells if "AVAILABLE_AGENTS: dict" in c]
        assert registry, "could not find the cell that writes the registry"
        assert "'explorer': ExplorerAgent" in registry[0]

    def test_the_rerun_subprocess_can_import_the_staged_package(
        self, code_cells: list[str]
    ) -> None:
        """main.py runs in a subprocess, where the in-process sys.path insert
        does not apply; without PYTHONPATH the explorer import fails there and
        every game dies on its first action."""
        run = [c for c in code_cells if "'main.py', '--agent'" in c]
        assert run, "could not find the cell that launches main.py"
        assert "PYTHONPATH" in run[0] and bnb.KERNEL_PKG in run[0]

    def test_both_runners_play_the_agent_the_build_selected(
        self, code_cells: list[str]
    ) -> None:
        """The mock is the only free signal before a submission, so it has to
        exercise the same agent the rerun will play."""
        joined = "\n".join(code_cells)
        assert f"'--agent', '{bnb.KERNEL_AGENT}'" in joined
        assert f"AGENT_CLASS = my_agent.{bnb.KERNEL_AGENT_CLASS}" in joined

    def test_the_placeholder_is_always_substituted(self, code_cells: list[str]) -> None:
        """An unsubstituted placeholder reaches the kernel as a literal and
        fails there — the selector is a build-time constant and nothing in the
        kernel sets it."""
        assert not any(
            "__KERNEL_AGENT__" in c or "__KERNEL_AGENT_CLASS__" in c for c in code_cells
        )

    def test_a_model_free_agent_does_not_start_an_inference_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The explorer calls no model, and the vLLM install plus weight load is
        ~10 minutes billed against a wall-clock-scored run."""
        monkeypatch.setattr(bnb, "KERNEL_AGENT", "explorer")
        monkeypatch.setattr(bnb, "NEEDS_MODEL", False)
        cells = [
            "".join(c["source"])
            for c in bnb.build()["cells"]
            if c["cell_type"] == "code"
        ]
        assert not any("api_server" in c for c in cells)
        assert any("'--agent', 'explorer'" in c for c in cells)

    def test_an_unknown_agent_fails_the_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bnb, "KERNEL_AGENT", "nope")
        with pytest.raises(SystemExit, match="ARC_KERNEL_AGENT"):
            bnb.build()

    @pytest.mark.parametrize("agent", ["myagent", "explorer"])
    def test_every_build_installs_the_competition_sdk(
        self, agent: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """arc-agi used to be installed by the vLLM cell, so selecting the
        model-free agent dropped the SDK with it and every runner died on
        "No module named 'arc_agi'"."""
        monkeypatch.setattr(bnb, "KERNEL_AGENT", agent)
        monkeypatch.setattr(bnb, "KERNEL_AGENT_CLASS", bnb.KERNEL_AGENTS[agent])
        monkeypatch.setattr(bnb, "NEEDS_MODEL", agent != "explorer")
        cells = [
            "".join(c["source"])
            for c in bnb.build()["cells"]
            if c["cell_type"] == "code"
        ]
        assert any('"arc-agi", "python-dotenv"' in c for c in cells)

    def test_the_model_free_mock_covers_the_games_that_separate_the_agents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ls20 and lp85 are the only games any agent clears and they are not in
        the first six alphabetically, so a six-game mock reports 0 for both
        agents and looks like a tie."""
        monkeypatch.setattr(bnb, "KERNEL_AGENT", "explorer")
        monkeypatch.setattr(bnb, "KERNEL_AGENT_CLASS", "ExplorerAgent")
        monkeypatch.setattr(bnb, "NEEDS_MODEL", False)
        monkeypatch.setattr(bnb, "MOCK_THREADS", 25)
        cells = [
            "".join(c["source"])
            for c in bnb.build()["cells"]
            if c["cell_type"] == "code"
        ]
        assert any("'ARC_MOCK_THREADS', '25'" in c for c in cells)

    def test_a_game_that_scored_is_never_hidden_by_the_print_cap(
        self, code_cells: list[str]
    ) -> None:
        """The mock printed the first 8 games alphabetically, which excludes
        ls20 and lp85 — so a run that cleared a level reported all zeroes."""
        mock = [c for c in code_cells if "Mock submission:" in c]
        assert mock, "could not find the mock cell"
        assert "sorted(results.items())[:8]" not in mock[0]
        assert "scored + rest[:8]" in mock[0]

    def test_the_model_free_mock_gets_a_budget_worth_testing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rerun projects ~12.5k actions per game for this agent, so a
        150-action mock measures a fraction of what it will actually get."""
        monkeypatch.setattr(bnb, "KERNEL_AGENT", "explorer")
        monkeypatch.setattr(bnb, "KERNEL_AGENT_CLASS", "ExplorerAgent")
        monkeypatch.setattr(bnb, "NEEDS_MODEL", False)
        monkeypatch.setattr(bnb, "MOCK_ACTIONS", 2400)
        cells = [
            "".join(c["source"])
            for c in bnb.build()["cells"]
            if c["cell_type"] == "code"
        ]
        assert any("'ARC_MOCK_ACTIONS', '2400'" in c for c in cells)
