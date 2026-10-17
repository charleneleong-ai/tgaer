"""Stage 3: retrain the MTP drafter's dense weights on target states captured from the served model.

Reads the per-forward dumps of the SGLang capture hook (request pool index, token id, MRoPE position and the 10240-wide
hyper-connection state per row), rebuilds each request's token sequence, and trains `mtp_ref.MTPDrafter` with KL to the
target's top-20 next-token distribution at depth 1 (the recipe behind ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE runs
6 chained depths). Reports held-out depth-1 acceptance before and after, and writes a drop-in mtp-dense.safetensors.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mtp_ref import MTPConfig, MTPDrafter, hc_mix, load_dense, load_experts
from torch import Tensor

TOP_K = 20
MIXER = re.compile(
    r"^(?!.*(?:mtp|layers)).*hyper_connection_mixer\.(hc_norm|input_mix_weight_down|input_mix_weight_up)\.weight$"
)


@dataclass
class Segment:
    """One request's contiguous tokens: ids [n], MRoPE positions [3, n], HC states [n, 10240] (bf16)."""

    ids: Tensor
    pos: Tensor
    hc: Tensor

    def __len__(self) -> int:
        return self.ids.shape[0]


class Dumps:
    """Rebuilds per-request token sequences from forward-pass dumps written in execution order."""

    def __init__(self, window: int, min_len: int = 64) -> None:
        self.window, self.min_len = window, min_len
        self.open: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self.last_pos: dict[int, int] = {}
        self.done: list[Segment] = []

    def close(self, req: int) -> None:
        parts = self.open.pop(req, [])
        if not parts:
            return
        ids, pos, hc = (
            np.concatenate([p[i] for p in parts], axis=-1 if i == 1 else 0)
            for i in range(3)
        )
        if len(ids) >= self.min_len:
            keep = slice(-self.window, None)
            self.done.append(
                Segment(
                    torch.from_numpy(ids[keep].astype(np.int64)),
                    torch.from_numpy(pos[:, keep].astype(np.int64)),
                    torch.from_numpy(np.ascontiguousarray(hc[keep])).view(
                        torch.bfloat16
                    ),
                )
            )

    def add(
        self, req: np.ndarray, ids: np.ndarray, pos: np.ndarray, hc: np.ndarray
    ) -> None:
        pos = pos if pos.ndim == 2 else np.broadcast_to(pos, (3, pos.shape[0]))
        for r in np.unique(req):
            rows = np.nonzero(req == r)[0]
            start = int(pos[0, rows[0]])
            if (
                self.last_pos.get(int(r), start - 1) != start - 1
            ):  # pool slot reused by a new request
                self.close(int(r))
            self.open.setdefault(int(r), []).append((ids[rows], pos[:, rows], hc[rows]))
            self.last_pos[int(r)] = int(pos[0, rows[-1]])
            if sum(len(p[0]) for p in self.open[int(r)]) > 2 * self.window:
                ids_, pos_, hc_ = (
                    np.concatenate(
                        [p[i] for p in self.open[int(r)]], axis=-1 if i == 1 else 0
                    )
                    for i in range(3)
                )
                self.open[int(r)] = [
                    (ids_[-self.window :], pos_[:, -self.window :], hc_[-self.window :])
                ]

    @classmethod
    def load(cls, dump_dir: Path, window: int, min_len: int = 64) -> list[Segment]:
        d = cls(window, min_len)
        for f in sorted(dump_dir.glob("*.npz")):
            z = np.load(f)
            d.add(z["req"], z["ids"], z["pos"], z["hc"])
        for r in list(d.open):
            d.close(r)
        return d.done


class TargetHead:
    """The target's own final mixer and LM head: HC state -> next-token logits."""

    def __init__(
        self, cfg: MTPConfig, norm: Tensor, down: Tensor, up: Tensor, lm_head: Tensor
    ) -> None:
        self.cfg, self.norm, self.down, self.up, self.lm_head = (
            cfg,
            norm,
            down,
            up,
            lm_head,
        )

    @torch.no_grad()
    def top_k(
        self, hc: Tensor, k: int = TOP_K, chunk: int = 1024
    ) -> tuple[Tensor, Tensor]:
        """Top-k token ids and their probabilities renormalised over the k."""
        ids, probs = [], []
        for i in range(0, hc.shape[0], chunk):
            mixed, _ = hc_mix(
                hc[i : i + chunk].to(self.down.dtype),
                self.norm,
                self.down,
                self.up,
                self.cfg,
            )
            logits, idx = F.linear(mixed, self.lm_head).float().topk(k, dim=-1)
            ids.append(idx)
            probs.append(logits.softmax(-1))
        return torch.cat(ids), torch.cat(probs)

    @classmethod
    def from_checkpoint(
        cls, model_dir: Path, cfg: MTPConfig, device: str, fallback_lm_head: Tensor
    ) -> TargetHead:
        from safetensors import safe_open

        weight_map = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        found = {MIXER.match(k).group(1): k for k in weight_map if MIXER.match(k)}
        if sorted(found) != ["hc_norm", "input_mix_weight_down", "input_mix_weight_up"]:
            raise RuntimeError(f"target mixer weights not found unquantised: {found}")
        heads = [
            k for k in weight_map if k.endswith("lm_head.weight") and "mtp" not in k
        ]
        if len(heads) == 1:
            found["lm_head"] = heads[0]
        out = {
            "lm_head": fallback_lm_head
        }  # the drafter's copy, if the target's is quantised
        for name, key in found.items():
            with safe_open(
                str(model_dir / weight_map[key]), framework="pt", device=device
            ) as f:
                out[name] = f.get_tensor(key)
        return cls(
            cfg,
            out["hc_norm"],
            out["input_mix_weight_down"],
            out["input_mix_weight_up"],
            out["lm_head"],
        )


def kl_to_target(logits: Tensor, top_ids: Tensor, top_p: Tensor) -> Tensor:
    logq = logits.float().log_softmax(-1).gather(-1, top_ids)
    return (top_p * (top_p.clamp_min(1e-12).log() - logq)).sum(-1).mean()


def step_inputs(seg: Segment, head: TargetHead, device: str) -> tuple[Tensor, ...]:
    """Row r: (state at r, token r+1) at position r+1 -> target distribution of token r+2 (target state at r+1)."""
    hc = seg.hc.to(device)
    top_ids, top_p = head.top_k(hc[1:])
    return (
        hc[:-1],
        seg.ids[1:].to(device),
        seg.pos[:, 1:].to(device),
        top_ids,
        top_p,
        seg.ids[2:].to(device),
    )


@torch.no_grad()
def acceptance(
    drafter: MTPDrafter, segs: list[Segment], head: TargetHead, device: str
) -> float:
    hits = total = 0
    for seg in segs:
        hc, nxt, pos, _, _, target = step_inputs(seg, head, device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits, _ = drafter(hc, nxt, pos)
        hits += int(
            (logits[:-1].argmax(-1) == target).sum()
        )  # the last row's token t+2 was never generated
        total += target.shape[0]
    return hits / max(total, 1)


def train(
    drafter: MTPDrafter,
    train_segs: list[Segment],
    head: TargetHead,
    device: str,
    epochs: int,
    lr: float,
    log: list[dict],
) -> None:
    params = [p for p in drafter.params.values() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps = epochs * len(train_segs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 1.0 - 0.9 * s / max(steps, 1)
    )
    rng = random.Random(0)
    t0 = time.time()
    for epoch in range(epochs):
        order = list(range(len(train_segs)))
        rng.shuffle(order)
        for i in order:
            hc, nxt, pos, top_ids, top_p, _ = step_inputs(train_segs[i], head, device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                logits, _ = drafter(hc, nxt, pos)
            loss = kl_to_target(logits, top_ids, top_p)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            log.append(
                {"epoch": epoch, "loss": loss.item(), "t": round(time.time() - t0, 1)}
            )
            if len(log) % 25 == 0:
                print(
                    f"[train] step {len(log)}/{steps} loss {log[-1]['loss']:.4f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )


def export(original: Path, drafter: MTPDrafter, out: Path) -> list[str]:
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(original))
    trained = drafter.trainable()
    for k, v in trained.items():
        tensors[k] = v.to(tensors[k].dtype).cpu().contiguous()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata={"format": "pt"})
    return sorted(trained)


def main(
    dump_dir: Path,
    drafter_dir: Path,
    model_dir: Path,
    out_dir: Path,
    window: int = 4096,
    epochs: int = 2,
    lr: float = 2e-5,
    holdout: float = 0.1,
    device: str = "cuda",
) -> None:
    cfg = MTPConfig.from_json(drafter_dir / "config.json")
    segs = Dumps.load(dump_dir, window)
    random.Random(1).shuffle(segs)
    n_eval = max(1, int(len(segs) * holdout))
    eval_segs, train_segs = segs[:n_eval], segs[n_eval:]
    report: dict = {
        "segments": len(segs),
        "train_rows": sum(map(len, train_segs)),
        "eval_rows": sum(map(len, eval_segs)),
    }
    print("[train]", json.dumps(report), flush=True)

    dense = {
        k: v.to(device)
        for k, v in load_dense(drafter_dir / "mtp-dense.safetensors", device).items()
    }
    experts = load_experts(
        drafter_dir / "mtp-routed-experts-int4.safetensors", cfg, device
    )
    drafter = MTPDrafter(cfg, dense, experts)
    for p in drafter.params.values():
        if p.requires_grad:
            p.data = (
                p.data.float()
            )  # fp32 master weights; forward runs under bf16 autocast
    head = TargetHead.from_checkpoint(model_dir, cfg, device, dense["lm_head.weight"])

    report["acceptance_before"] = acceptance(drafter, eval_segs, head, device)
    print(
        "[train] held-out depth-1 acceptance before",
        report["acceptance_before"],
        flush=True,
    )
    log: list[dict] = []
    train(drafter, train_segs, head, device, epochs, lr, log)
    report["acceptance_after"] = acceptance(drafter, eval_segs, head, device)
    print(
        "[train] held-out depth-1 acceptance after",
        report["acceptance_after"],
        flush=True,
    )
    report["trained_tensors"] = export(
        drafter_dir / "mtp-dense.safetensors",
        drafter,
        out_dir / "mtp-dense.safetensors",
    )
    report["loss_first_last"] = [log[0]["loss"], log[-1]["loss"]] if log else None
    (out_dir / "train_report.json").write_text(
        json.dumps({**report, "log": log}, indent=1)
    )
    print(
        "[train] done",
        json.dumps({k: v for k, v in report.items() if k != "trained_tensors"}),
        flush=True,
    )


if __name__ == "__main__":
    import typer  # CLI only; the Kaggle run calls main() from SGLang's venv, which lacks typer

    typer.run(main)
