"""Is lp85's object-level state Markov? The gating prerequisite for M1.

A model-written ``simulate(state, action)`` can only work if the state it is
handed actually determines the successor. lp85 has hidden state the frame never
shows — the per-level ``StepCounter`` above all — so this is not a given.

Walks the reachable space with random button presses and looks for a
``(state, action)`` pair that produced two different successors. Compares an
object-level key (colour, top-left, pixel count per component) against the whole
frame, since object-level is what M1 would hand a model.

What this cannot show: it resets on GAME_OVER, so it never probes the
budget-exhaustion boundary — exactly where the hidden counter would bite.
"""

import copy
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sia-oss" / "bench"))
import arc_runner  # noqa: E402

arc_runner._load_local(
    "tgaer.agents.arc_agi3_explorer",
    str(REPO / "src/tgaer/agents/arc_agi3_explorer.py"),
)
from tgaer.agents.arc_agi3_grid import components  # noqa: E402
from tgaer.evaluation.arc_agi3_score_local import (  # noqa: E402
    OperationMode,
    arc_agi,
    require_starter,
)

os.chdir(REPO)
os.environ.setdefault("ARC_TGAER_REPO", str(REPO))
from arcengine import ActionInput, GameAction  # noqa: E402

require_starter()
arc = arc_agi.Arcade(
    operation_mode=OperationMode.OFFLINE,
    environments_dir=str(REPO / "environment_files"),
)
env = arc.make("lp85")
fd = env.reset()
game = env._game


def G(f):
    return np.asarray(f.frame[-1], dtype=np.int16)


def click(g, rc):
    r, c = rc
    return g.perform_action(
        ActionInput(id=GameAction.ACTION6, data={"x": int(c), "y": int(r)}), raw=True
    )


def objs(a):
    bg = np.bincount(a.ravel()).argmax()
    out = []
    for v in (int(x) for x in np.unique(a)):
        if v == bg:
            continue
        for c in components(a, (v,)):
            rs, cs = c[:, 0], c[:, 1]
            out.append((v, int(rs.min()), int(cs.min()), len(c)))
    return tuple(sorted(out))


def buttons(g, base):
    seen = {}
    for r in range(0, 64, 3):
        for c in range(0, 64, 3):
            g2 = G(click(copy.deepcopy(g), (r, c)))
            if np.array_equal(g2, base):
                continue
            seen.setdefault(g2.tobytes(), (r, c))
    return list(seen.values())


base = G(fd)
cells = buttons(game, base)
print(f"buttons: {len(cells)}")

rng = random.Random(0)
trans = defaultdict(set)  # (obj_state, action) -> {next obj_state}
full_trans = defaultdict(set)  # (full_frame, action) -> {next frame}
node = copy.deepcopy(game)
cur_o, cur_f = objs(base), base.tobytes()
N = 1500
for i in range(N):
    rc = rng.choice(cells)
    child = copy.deepcopy(node)
    out = click(child, rc)
    a2 = G(out)
    o2, f2 = objs(a2), a2.tobytes()
    trans[(cur_o, rc)].add(o2)
    full_trans[(cur_f, rc)].add(f2)
    if (
        "GAME_OVER" in str(getattr(out, "state", ""))
        or int(getattr(out, "levels_completed", 0)) > 0
    ):
        node = copy.deepcopy(game)
        a2 = base
        o2, f2 = objs(base), base.tobytes()
    else:
        node = child
    cur_o, cur_f = o2, f2

amb_o = {k: v for k, v in trans.items() if len(v) > 1}
amb_f = {k: v for k, v in full_trans.items() if len(v) > 1}
print(f"\n{N} transitions sampled")
print(
    f"  object-level : {len(trans)} distinct (state, action) keys, {len(amb_o)} ambiguous"
)
print(
    f"  frame-level  : {len(full_trans)} distinct (frame, action) keys, {len(amb_f)} ambiguous"
)
print(f"\nobject state is Markov: {not amb_o}")
print(f"frame  state is Markov: {not amb_f}")
if amb_o:
    k = next(iter(amb_o))
    print(
        f"  example ambiguity: action {k[1]} from one state -> {len(amb_o[k])} different successors"
    )
