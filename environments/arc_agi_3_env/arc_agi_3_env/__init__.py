from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_impl() -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / "arc_agi_3_env.py"
    spec = importlib.util.spec_from_file_location("_arc_agi_3_env_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load environment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()

load_environment = _impl.load_environment
ArcAgi3Env = _impl.ArcAgi3Env
ArcAgi3Rubric = _impl.ArcAgi3Rubric
reward = _impl.reward
num_actions = _impl.num_actions
timed_out = _impl.timed_out

__all__ = [
    "load_environment",
    "ArcAgi3Env",
    "ArcAgi3Rubric",
    "reward",
    "num_actions",
    "timed_out",
]
