"""Final-window rotation for dfranzen's priority gate: in the last ENDGAME_FRACTION of the run, a freed slot
goes to the longest-waiting game (FIFO) instead of the highest-priority one, so every parked game gets a turn
before the deadline. Outside the window the gate is unchanged.
"""

from __future__ import annotations

import heapq
import os
import time
from typing import Any

ENDGAME_FRACTION = float(os.environ.get("ARC3_ENDGAME_ROTATION_FRACTION", "0.15"))


class EndgameRotation:
    """Wraps _PriorityGate._pump; the gate's own lock is already held whenever _pump runs."""

    @staticmethod
    def in_window(gate: Any, now: float) -> bool:
        start, deadline = gate._clock_start, gate._clock_deadline
        if start is None or deadline is None or deadline <= start:
            return False
        return deadline - now <= ENDGAME_FRACTION * (deadline - start)

    @staticmethod
    def admit_fifo(gate: Any) -> None:
        woke = False
        while gate._free > 0 and gate._waiting:
            token = min(t for _, t in gate._waiting)
            gate._waiting = [(p, t) for p, t in gate._waiting if t != token]
            gate._snapshots.pop(token, None)
            gate._admitted.add(token)
            gate._free -= 1
            woke = True
        if woke:
            heapq.heapify(gate._waiting)
            gate._cond.notify_all()

    @staticmethod
    def install(gate_cls: type) -> None:
        pump = gate_cls._pump

        def _pump(gate: Any) -> None:
            if EndgameRotation.in_window(gate, time.monotonic()):
                if not getattr(gate, "_rotation_announced", False):
                    gate._rotation_announced = True
                    print(
                        "[endgame-rotation] final window: admitting waiting games first-in-first-out",
                        flush=True,
                    )
                EndgameRotation.admit_fifo(gate)
                return
            pump(gate)

        gate_cls._pump = _pump
