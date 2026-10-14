# ruff: noqa: F821  # runs inside the probe notebook, after the base cells define WORKING_DIR, proc, ...
# Drafter-retraining stage 1: can SGLang hand back the target's hidden states on our ARC requests?
# Boot the target without speculative decoding and with --enable-return-hidden-states, generate on 20 real
# requests with top-20 logprobs, and record what comes back: shapes of hidden states (2560 mixed vs 10240
# hyper-connection), whether top logprobs align with output tokens, and the bytes per token.
import glob
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

LAUNCHER = Path("/kaggle/launcher_cell.py").read_text()
WORKLOAD = json.loads(
    Path(
        sorted(glob.glob("/kaggle/input/**/workload.json", recursive=True))[0]
    ).read_text()
)
OUT = Path(WORKING_DIR, "capture_probe")
OUT.mkdir(parents=True, exist_ok=True)

patched = LAUNCHER.replace("\nVENV = ", "\nCFG.update({'SPEC': False})\nVENV = ", 1)
assert patched.count('"--enable-metrics",') == 1
patched = patched.replace(
    '"--enable-metrics",', '"--enable-metrics", "--enable-return-hidden-states",'
)
exec(patched)


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVED_MODEL_PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def shape(x) -> list:
    dims = []
    while isinstance(x, list) and x:
        dims.append(len(x))
        x = x[0]
    return dims


report = {"boot_ok": proc.poll() is None}
samples = [lane["requests"][0] for lane in WORKLOAD] + [
    lane["requests"][-1] for lane in WORKLOAD
]
t0 = time.time()
for i, r in enumerate(samples):
    body = {
        "model": SERVED_MODEL_NAME,
        **r,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 512,
        "logprobs": True,
        "top_logprobs": 20,
        "return_hidden_states": True,
    }
    try:
        out = post("/v1/chat/completions", body)
    except Exception as exc:
        report.setdefault("errors", []).append(repr(exc)[:300])
        continue
    ch = out["choices"][0]
    hs = ch.get("hidden_states") or out.get("hidden_states")
    lp = (ch.get("logprobs") or {}).get("content") or []
    row = {
        "i": i,
        "choice_keys": sorted(ch),
        "top_keys": sorted(out),
        "completion_tokens": out["usage"]["completion_tokens"],
        "n_logprob_tokens": len(lp),
        "n_top": len(lp[0]["top_logprobs"]) if lp else 0,
        "hidden_shape": shape(hs) if hs is not None else None,
    }
    if i < 3 and hs is not None:
        np.save(OUT / f"hidden_{i}.npy", np.asarray(hs, dtype=np.float32))
        (OUT / f"logprobs_{i}.json").write_text(json.dumps(lp[:64]))
    print("[capture]", json.dumps(row), flush=True)
    report.setdefault("rows", []).append(row)
# Short text prompts captured in full (prompt + greedy output states and token ids): the stage-2 gate replays
# them through mtp_ref.py and checks it predicts token t+2 as often as SGLang's drafter does.
SHORT = [
    "Explain in three sentences why a breadth-first search finds the shortest path in an unweighted grid.",
    "Write a Python function that rotates a 2D list of ints 90 degrees clockwise, then show it on a 3x3 example.",
    "A game board has a red block that moves one cell per action and stops at walls. List a plan to test which actions move it.",
    "Summarise the rules of tic-tac-toe, then say who wins with perfect play.",
]
for j, prompt in enumerate(SHORT):
    try:
        out = post(
            "/generate",
            {
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 384},
                "return_hidden_states": True,
                "return_logprob": True,
                "top_logprobs_num": 20,
                "logprob_start_len": 0,
            },
        )
    except Exception as exc:
        report.setdefault("errors", []).append(f"generate {j}: {exc!r}"[:300])
        continue
    meta = out["meta_info"]
    hs = meta.get("hidden_states") or []
    row = {
        "j": j,
        "meta_keys": sorted(meta),
        "n_hidden_entries": len(hs),
        "hidden_entry_shapes": [shape(x) for x in hs[:3]],
        "n_input_logprobs": len(meta.get("input_token_logprobs") or []),
        "n_output_logprobs": len(meta.get("output_token_logprobs") or []),
    }
    if hs:
        np.savez_compressed(
            OUT / f"short_{j}.npz",
            prompt_hidden=np.asarray(hs[0], dtype=np.float32),
            output_hidden=np.asarray(hs[1:], dtype=np.float32),
            input_ids=np.asarray(
                [t[1] for t in meta.get("input_token_logprobs") or []], dtype=np.int64
            ),
            output_ids=np.asarray(
                [t[1] for t in meta.get("output_token_logprobs") or []], dtype=np.int64
            ),
        )
        (OUT / f"short_{j}_top.json").write_text(
            json.dumps(meta.get("output_top_logprobs", [])[:64])
        )
    print("[capture]", json.dumps(row), flush=True)
    report.setdefault("short_rows", []).append(row)
report["wall_s"] = round(time.time() - t0, 1)
(OUT / "report.json").write_text(json.dumps(report, indent=1))
print(
    "[capture] done",
    json.dumps({k: v for k, v in report.items() if k != "rows"}),
    flush=True,
)
