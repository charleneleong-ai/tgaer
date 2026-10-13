# ruff: noqa: F821  # runs inside the bench notebook, after the base cells define WORKING_DIR, proc, LOG, ...
# Speculative-decoding settings bench: boot SGLang once per config and replay the same 10 lanes of real
# ARC analyzer requests (consecutive turns per game, 10 games at once), recording tok/s and accept length.
import glob
import json
import os
import re
import signal
import socket
import statistics
import threading
import time
import urllib.request
from pathlib import Path

LAUNCHER = Path("/kaggle/launcher_cell.py").read_text()

WORKLOAD = json.loads(
    Path(
        sorted(glob.glob("/kaggle/input/**/workload.json", recursive=True))[0]
    ).read_text()
)
CONFIGS = [
    ("base_s3", {}),
    ("s2", {"SPEC_STEPS": 2}),
    ("s4", {"SPEC_STEPS": 4}),
    ("s5", {"SPEC_STEPS": 5}),
    ("s3_nofrspec", {"FRSPEC": False}),
    ("nospec", {"SPEC": False}),
    ("base_s3_rep", {}),
]
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048)
DECODE = re.compile(
    r"#running-req: (\d+).*?(?:accept len: ([\d.]+).*?)?gen throughput \(token/s\): ([\d.]+)"
)
RESULTS = Path(WORKING_DIR, "specbench_results.jsonl")


def post(body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVED_MODEL_PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def replay() -> dict:
    usage, errors, lock = [], [], threading.Lock()

    def lane(requests: list) -> None:
        for r in requests:
            try:
                out = post({"model": SERVED_MODEL_NAME, **r, **SAMPLING})
                with lock:
                    usage.append(out["usage"])
            except Exception as exc:
                with lock:
                    errors.append(repr(exc)[:200])

    t0 = time.time()
    threads = [
        threading.Thread(target=lane, args=(lane_["requests"],)) for lane_ in WORKLOAD
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    completion = sum(u["completion_tokens"] for u in usage)
    return {
        "wall_s": round(wall, 1),
        "completed": len(usage),
        "errors": errors[:3],
        "n_errors": len(errors),
        "completion_tokens": completion,
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage),
        "tok_per_s": round(completion / wall, 1),
    }


def decode_stats(offset: int) -> dict:
    with open(LOG, "rb") as f:
        f.seek(offset)
        text = f.read().decode(errors="replace")
    rows = [
        (int(n), float(a) if a else None, float(g)) for n, a, g in DECODE.findall(text)
    ]
    busy = [r for r in rows if r[0] >= 8]
    acc = [r[1] for r in rows if r[1] is not None]
    return {
        "decode_lines": len(rows),
        "busy_lines": len(busy),
        "gen_tps_busy_mean": round(statistics.mean(r[2] for r in busy), 1)
        if busy
        else None,
        "accept_len_mean": round(statistics.mean(acc), 3) if acc else None,
    }


def stop_server() -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(120):
        if proc.poll() is not None:
            break
        time.sleep(1)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    while port_open():
        time.sleep(2)
    time.sleep(10)


def port_open() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", int(SERVED_MODEL_PORT))) == 0


def wait_healthy(limit_s: float = 1200) -> bool:
    t0 = time.time()
    while time.time() - t0 < limit_s and proc.poll() is None:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=5
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


for i, (cfg_name, overrides) in enumerate(CONFIGS):  # the launcher exec rebinds `name`
    if i:
        precache_model_thread.start = lambda: None  # the first boot already ran it
    NOTEBOOK_START_TIME = time.time()
    SERVER_STARTUP_TIMEOUT = 20 * 60
    exec(LAUNCHER.replace("\nVENV = ", f"\nCFG.update({overrides!r})\nVENV = ", 1))
    offset = log_offset
    ready = wait_healthy()
    row = {
        "config": cfg_name,
        "overrides": overrides,
        "ready": ready,
        "boot_s": int(time.time() - NOTEBOOK_START_TIME),
    }
    if ready:
        post(
            {
                "model": SERVED_MODEL_NAME,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 16,
            }
        )
        row |= replay() | decode_stats(offset)
    print("[specbench]", json.dumps(row), flush=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    stop_server()
