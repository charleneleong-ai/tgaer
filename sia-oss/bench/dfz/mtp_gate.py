"""Stage-2 gate: does mtp_ref.py draft like SGLang's drafter on the target states captured by the stage-1 probe?

For each output position t of a captured greedy generation, the drafter sees the target's hyper-connection state at
t and token t+1 and should predict token t+2. Depth-1 acceptance of the stock drafter should sit near the ~0.82-0.86
position-1 acceptance reported for this model family; far below that means the port is wrong.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import typer
from mtp_ref import MTPConfig, MTPDrafter, load_dense, load_experts
from torch import Tensor

app = typer.Typer(add_completion=False)
Drafter = Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]


def depth1_acceptance(
    drafter: Drafter, hidden: Tensor, tokens: Tensor, first_output: int
) -> tuple[int, int]:
    """hidden[t] is the target state after token t; returns (hits, trials) over output positions."""
    n = tokens.shape[0]
    with torch.no_grad():
        logits, _ = drafter(hidden[: n - 1], tokens[1:n], torch.arange(1, n))
    pred = logits.argmax(-1)  # row t predicts token t+2
    # rows whose target t+2 is a generated token
    rows = torch.arange(max(first_output - 2, 0), n - 2)
    return int((pred[rows] == tokens[rows + 2]).sum()), len(rows)


def load_capture(path: Path) -> tuple[Tensor, Tensor, int]:
    z = np.load(path)
    hidden = np.concatenate(
        [
            z["prompt_hidden"],
            z["output_hidden"].reshape(-1, z["prompt_hidden"].shape[-1]),
        ]
    )
    tokens = np.concatenate([z["input_ids"], z["output_ids"]])
    n = min(len(hidden), len(tokens))
    return (
        torch.from_numpy(hidden[:n]).bfloat16(),
        torch.from_numpy(tokens[:n]),
        len(z["input_ids"]),
    )


@app.command()
def main(
    capture_dir: Path, drafter_dir: Path, out: Path = Path("mtp_gate.json")
) -> None:
    cfg = MTPConfig.from_json(drafter_dir / "config.json")
    drafter = MTPDrafter(
        cfg,
        load_dense(drafter_dir / "mtp-dense.safetensors", "cpu"),
        load_experts(drafter_dir / "mtp-routed-experts-int4.safetensors", cfg, "cpu"),
    )
    width = cfg.hidden_size * cfg.hc_count
    rows = []
    for path in sorted(capture_dir.glob("short_*.npz")):
        hidden, tokens, first_output = load_capture(path)
        if hidden.shape[-1] != width:
            typer.echo(
                f"{path.name}: hidden width {hidden.shape[-1]}, need {width}; capture needs the HC state"
            )
            raise typer.Exit(1)
        hits, trials = depth1_acceptance(drafter, hidden, tokens, first_output)
        rows.append({"capture": path.name, "hits": hits, "trials": trials})
        typer.echo(f"{path.name}: {hits}/{trials} = {hits / max(trials, 1):.3f}")
    hits, trials = sum(r["hits"] for r in rows), sum(r["trials"] for r in rows)
    out.write_text(
        json.dumps({"rows": rows, "pooled": hits / max(trials, 1)}, indent=1)
    )
    typer.echo(
        f"pooled depth-1 acceptance {hits}/{trials} = {hits / max(trials, 1):.3f}"
    )


if __name__ == "__main__":
    app()
