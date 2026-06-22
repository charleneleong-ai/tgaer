"""Empirically pin the ARC-AGI-3 *click* mechanic + coordinate convention.

No VL server needed — we send scripted ACTION6 clicks and watch the frame delta.
Answers two design questions for the ``click`` verb:
  1. What does a click DO? direct-click (clicking an object consumes/activates it),
     click-to-move (clicking a tile translates an avatar), or no visible effect.
  2. Is ACTION6's (x, y) ordered (col, row) or (row, col)? We click an isolated
     object at an asymmetric (row, col) under BOTH orderings and see which lands.

Usage:
  PYTHONPATH=src uv run python experiments/probe_click_mechanic.py scan
  PYTHONPATH=src uv run python experiments/probe_click_mechanic.py probe <game_id>
"""

from __future__ import annotations

import os
import sys
from collections import Counter, deque

import numpy as np

from tgaer.envs.arc_agi3.arc_agi3_client import ArcAgi3Client

NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(arr: np.ndarray, value: int) -> list[np.ndarray]:
    mask = arr == value
    seen = np.zeros_like(mask, bool)
    out = []
    for r0, c0 in np.argwhere(mask):
        if seen[r0, c0]:
            continue
        q, comp = deque([(r0, c0)]), []
        seen[r0, c0] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in NBRS:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < arr.shape[0] and 0 <= c2 < arr.shape[1] and mask[r2, c2] and not seen[r2, c2]:
                    seen[r2, c2] = True
                    q.append((r2, c2))
        out.append(np.array(comp))
    return out


def _objects(arr: np.ndarray) -> list[tuple[int, np.ndarray, int]]:
    """Return (colour, centroid, size) for every connected component of every
    non-background colour, smallest first — small compact pieces are the likely
    clickable sprites/markers."""
    bg = Counter(arr.ravel().tolist()).most_common(1)[0][0]
    objs = []
    for v in np.unique(arr):
        if v == bg:
            continue
        for comp in _components(arr, int(v)):
            objs.append((int(v), comp.mean(0), len(comp)))
    return sorted(objs, key=lambda o: o[2])


def _diff(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int, int, int]]:
    return [(int(r), int(c), int(a[r, c]), int(b[r, c])) for r, c in np.argwhere(a != b)]


def _client() -> ArcAgi3Client:
    c = ArcAgi3Client(api_key=os.environ["ARC_API_KEY"])
    c.open_scorecard()
    return c


def scan() -> None:
    import requests

    from tgaer.envs.arc_agi3.arc_agi3_client import BASE_URL

    key = os.environ["ARC_API_KEY"]
    games = [g["game_id"] for g in requests.get(f"{BASE_URL}/api/games", headers={"X-API-Key": key}).json()]
    c = _client()
    for gid in games:
        try:
            f = c.reset(gid)
            has6 = 6 in f.available_actions
            print(f"{gid:18s} actions={f.available_actions} click6={'YES' if has6 else 'no'}", flush=True)
        except Exception as exc:  # noqa: BLE001 - probe must survive a bad game
            print(f"{gid:18s} ERROR {type(exc).__name__}: {exc}", flush=True)


def _click(c: ArcAgi3Client, gid: str, guid: str, x: int, y: int):
    return c.act(gid, guid, 6, x=x, y=y)


def probe(gid: str) -> None:
    c = _client()
    f = c.reset(gid)
    print(f"[probe] {gid} state={f.state} actions={f.available_actions}", flush=True)
    if 6 not in f.available_actions:
        print("[probe] ACTION6 not available — not a click game", flush=True)
        return
    arr0 = np.asarray(f.frame[-1])
    objs = _objects(arr0)
    print(f"[probe] {len(objs)} objects; smallest 6:", flush=True)
    for col, cen, sz in objs[:6]:
        print(f"    colour={col} centroid=(r={cen[0]:.1f},c={cen[1]:.1f}) size={sz}", flush=True)

    if not objs:
        print("[probe] no foreground objects to click", flush=True)
        return

    # Pick the most asymmetric small object so (col,row) vs (row,col) is unambiguous.
    target = max(objs[:8], key=lambda o: abs(o[1][0] - o[1][1]))
    col, cen, _ = target
    r, cc = int(round(cen[0])), int(round(cen[1]))
    print(f"\n[probe] target colour={col} at (row={r}, col={cc})", flush=True)

    guid = f.guid
    # Convention test A: x=col, y=row (standard image convention)
    fa = _click(c, gid, guid, x=cc, y=r)
    arr_a = np.asarray(fa.frame[-1])
    dA = _diff(arr0, arr_a)
    print(
        f"[probe] click(x=col={cc}, y=row={r}): changed_cells={len(dA)} "
        f"state={fa.state} levels={fa.levels_completed} "
        f"target_now={int(arr_a[r, cc])}",
        flush=True,
    )
    _summarize_delta(arr0, arr_a, r, cc)

    # Re-RESET so test B starts from the same frame (clicks are stateful).
    f2 = c.reset(gid)
    arr0b = np.asarray(f2.frame[-1])
    fb = _click(c, gid, f2.guid, x=r, y=cc)  # swapped ordering
    arr_b = np.asarray(fb.frame[-1])
    dB = _diff(arr0b, arr_b)
    print(
        f"[probe] click(x=row={r}, y=col={cc}): changed_cells={len(dB)} "
        f"state={fb.state} levels={fb.levels_completed}",
        flush=True,
    )

    print("\n[probe] VERDICT", flush=True)
    print(f"    convention x=col,y=row affected {len(dA)} cells; x=row,y=col affected {len(dB)} cells", flush=True)
    print("    (the ordering with more/object-local change is the live convention)", flush=True)


def _summarize_delta(before: np.ndarray, after: np.ndarray, r: int, cc: int) -> None:
    d = _diff(before, after)
    if not d:
        print("    -> no change: click had no visible effect on this object", flush=True)
        return
    near = [x for x in d if abs(x[0] - r) <= 3 and abs(x[1] - cc) <= 3]
    print(f"    -> {len(d)} cells changed, {len(near)} within 3 of target", flush=True)
    if after[r, cc] != before[r, cc]:
        print(f"    -> TARGET CELL changed {before[r, cc]}->{after[r, cc]} (direct-click: object reacted)", flush=True)
    elif near:
        print("    -> change is target-local but not on the cell (activate/select)", flush=True)
    else:
        print("    -> change is elsewhere (possible avatar move / global effect)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if cmd == "scan":
        scan()
    elif cmd == "probe":
        probe(sys.argv[2])
    else:
        print(__doc__)
