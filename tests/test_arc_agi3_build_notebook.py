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

    def test_each_module_gets_its_own_writefile_cell(self, code_cells: list[str]) -> None:
        written = {c.split("\n", 1)[0].split(maxsplit=1)[1].strip()
                   for c in code_cells if c.startswith("%%writefile")}
        for module in bnb.PORTED_MODULES:
            assert f"{bnb.KERNEL_PKG}/{module}" in written

    def test_module_bodies_are_copied_verbatim(self, code_cells: list[str]) -> None:
        """Copied, not re-implemented: the explorer is under active development
        and a hand-maintained second copy drifts out of date silently."""
        target = f"%%writefile {bnb.KERNEL_PKG}/tgaer/agents/arc_agi3_semantics.py\n"
        cell = next(c for c in code_cells if c.startswith(target))
        assert cell[len(target):] == (REPO / "src/tgaer/agents/arc_agi3_semantics.py").read_text()

    def test_the_staging_cell_runs_before_any_module_is_written(
        self, code_cells: list[str]
    ) -> None:
        """%%writefile will not create parent directories, so the tree and its
        __init__.py files have to exist first."""
        staging = next(i for i, c in enumerate(code_cells) if "[pkg] staged" in c)
        first_module = next(i for i, c in enumerate(code_cells)
                            if c.startswith(f"%%writefile {bnb.KERNEL_PKG}/"))
        assert staging < first_module

    def test_no_placeholder_survives_into_the_notebook(self, code_cells: list[str]) -> None:
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
                dest = Path(head.split(maxsplit=1)[1].strip().replace(bnb.KERNEL_PKG, pkg))
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
            out = subprocess.run([sys.executable, "-E", "-c", probe],
                                 capture_output=True, text=True, cwd=tmp)
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
