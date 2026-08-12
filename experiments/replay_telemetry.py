"""Offline replay of a captured ls20 trajectory with induction telemetry.

Re-feeds the obs stream from a RecordingAgent capture to a fresh explorer,
asserts the replayed action matches the captured one (faithful reproduction),
and reports where induction dies: when the avatar pins, how large the move
lattice grows, and which act() branch fires each step. Free and re-runnable —
the only paid step is the one-game capture (experiments/capture_ls20.py).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
from tgaer.agents.arc_agi3_recorder import _action_dict


def replay_traces(records: list[dict]) -> list[dict]:
    agent = ExplorerArcAgi3Agent()
    traces: list[dict] = []
    for i, rec in enumerate(records):
        action = agent.act(rec["obs"])
        got, want = _action_dict(action), rec["action"]
        assert got == want, (
            f"step {i}: replay {got} != captured {want} (non-determinism)"
        )
        traces.append(agent.trace)
    return traces


def summarise(traces: list[dict]) -> dict:
    pins = next((t["step"] for t in traces if t["avatar"] is not None), None)
    return {
        "avatar_pins_at": pins,
        "max_lattice": max((t["lattice_size"] for t in traces), default=0),
        "branches": dict(Counter(t["branch"] for t in traces)),
    }


def main(capture: str, out: str) -> None:
    records = [json.loads(ln) for ln in Path(capture).read_text().splitlines() if ln]
    traces = replay_traces(records)
    Path(out).write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    print(f"[replay] {len(traces)} steps -> {out}", flush=True)
    print(f"[replay] {summarise(traces)}", flush=True)


if __name__ == "__main__":
    cap = sys.argv[1] if len(sys.argv) > 1 else "experiments/ls20_capture.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "experiments/ls20_telemetry.jsonl"
    main(cap, dst)
