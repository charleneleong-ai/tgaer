# tests/test_arc_agi3_bench_sweep.py
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "sia-oss" / "bench" / "sweep.py"


def _load():
    """Import sweep.py by path: sia-oss/bench is not an importable package."""
    spec = importlib.util.spec_from_file_location("bench_sweep", SWEEP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_sweep"] = module
    spec.loader.exec_module(module)
    return module


sweep = _load()


SOURCE = """\
PROBE_LIMIT = 1


class ExplorerArcAgi3Agent:
    STUCK_WINDOW = 96
    MIN_NOVELTY = 0.15
"""


@pytest.fixture
def explorer(tmp_path: Path) -> Path:
    path = tmp_path / "arc_agi3_explorer.py"
    path.write_text(SOURCE)
    return path


class TestSetConstant:
    """Half the tuned constants sit on the agent class, so indentation is data."""

    def test_a_module_level_constant_is_rewritten(self, explorer: Path) -> None:
        sweep.set_constant(explorer, "PROBE_LIMIT", "3")
        assert "PROBE_LIMIT = 3" in explorer.read_text()

    @pytest.mark.parametrize(
        ("name", "value"), [("STUCK_WINDOW", "48"), ("MIN_NOVELTY", "0.2")]
    )
    def test_a_class_attribute_keeps_its_indentation(
        self, explorer: Path, name: str, value: str
    ) -> None:
        """Rewritten at column zero it would leave the class, changing the file's
        meaning rather than its value — and the agent would still read the old one."""
        sweep.set_constant(explorer, name, value)
        assert f"    {name} = {value}" in explorer.read_text()

    def test_the_rest_of_the_file_is_untouched(self, explorer: Path) -> None:
        sweep.set_constant(explorer, "STUCK_WINDOW", "48")
        text = explorer.read_text()
        assert "PROBE_LIMIT = 1" in text
        assert "    MIN_NOVELTY = 0.15" in text
        assert text.count("class ExplorerArcAgi3Agent:") == 1

    def test_an_unknown_name_is_refused(self, explorer: Path) -> None:
        """Silently sweeping nothing would report the baseline at every value."""
        with pytest.raises(SystemExit):
            sweep.set_constant(explorer, "NO_SUCH_CONSTANT", "1")
