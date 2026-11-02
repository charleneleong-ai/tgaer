"""Plain-PyTorch Qwen3.8-Flash-Next MTP drafter, differentiable, for retraining its dense weights.

Mirrors SGLang Pennyroyal 2.5.3 (`models/qwen4_exp_mtp.py` on the `qwen4_exp.py` decoder layer): input fusion of the
target's 4-stream hyper-connection state with the next token's embedding, one attention decoder layer with gated
output and QSA sparse selection, a 512-expert top-10 MoE plus a gated shared expert, and the final hyper-connection
mixer before the LM head. The routed experts stay frozen at their served INT4 values, as do the shared embedding,
the LM head and the QSA indexer; the other dense MTP weights train.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MTP = "mtp.layers.0."


@dataclass(frozen=True)
class MTPConfig:
    hidden_size: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1e7
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    indexer_n_heads: int = 4
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.indexer_compress_ratio

    @classmethod
    def from_json(cls, path: Path) -> MTPConfig:
        raw = json.loads(path.read_text())
        text = raw.get("text_config", raw)
        kept = {f.name: text[f.name] for f in fields(cls) if f.name in text}
        rope = text.get("rope_parameters", {})
        kept["rope_theta"] = text.get("mtp", {}).get(
            "rope_theta", rope.get("rope_theta", cls.rope_theta)
        )
        kept["mrope_section"] = tuple(rope.get("mrope_section", cls.mrope_section))
        return cls(**kept)


def gemma_rms(
    x: Tensor, weight: Tensor, eps: float, group: int | None = None
) -> Tensor:
    shape = x.shape
    xf = x.float() if group is None else x.float().unflatten(-1, (-1, group))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf.reshape(shape) * (1.0 + weight.float())).to(x.dtype)


def dequant_int4(packed: Tensor, scale: Tensor, group: int = 32) -> Tensor:
    """compressed-tensors pack-quantized, symmetric: 8 nibbles per int32, low nibble first, stored as q + 8."""
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> shifts) & 0xF).flatten(-2).to(torch.int8) - 8
    return (q.unflatten(-1, (-1, group)).to(scale.dtype) * scale.unsqueeze(-1)).flatten(
        -2
    )


def hc_mix(
    h: Tensor, norm_w: Tensor, down_w: Tensor, up_w: Tensor, cfg: MTPConfig
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Gated hyper-connection mix of the 4 streams into one hidden; also returns what combine needs."""
    hn = gemma_rms(h, norm_w, cfg.rms_norm_eps, group=cfg.hidden_size)
    gate = torch.sigmoid(F.linear(F.silu(F.linear(hn, down_w) / cfg.hc_count), up_w))
    streams = (cfg.hc_count, cfg.hidden_size)
    mixed = (gate.unflatten(-1, streams) * hn.unflatten(-1, streams)).mean(-2)
    return mixed, (h, hn)


class Rope:
    """NeoX partial RoPE with Qwen interleaved MRoPE; positions are [3, T] (t, h, w) or [T] for text."""

    def __init__(
        self, rotary_dim: int, theta: float, section: tuple[int, int, int]
    ) -> None:
        half = rotary_dim // 2
        self.rotary_dim = rotary_dim
        self.inv_freq = 1.0 / theta ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim
        )
        pair = torch.arange(half)
        self.axis = torch.zeros(half, dtype=torch.long)
        self.axis[((pair % 3) == 1) & (pair < section[1] * 3)] = 1
        self.axis[((pair % 3) == 2) & (pair < section[2] * 3)] = 2

    def __call__(self, x: Tensor, positions: Tensor) -> Tensor:
        pos = positions if positions.ndim == 2 else positions.unsqueeze(0).expand(3, -1)
        pos_per_pair = pos.double()[self.axis.to(pos.device)].T  # [T, half]
        angle = (pos_per_pair * self.inv_freq.to(pos.device)).float()
        cos, sin = (
            angle.cos().unsqueeze(-2),
            angle.sin().unsqueeze(-2),
        )  # broadcast over heads
        r = self.rotary_dim
        x1, x2 = x[..., : r // 2].float(), x[..., r // 2 : r].float()
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(
            x.dtype
        )
        return torch.cat([rotated, x[..., r:]], dim=-1)


class QSASelect:
    """Which earlier tokens each query attends to: the top-scoring complete groups plus its own partial group."""

    @staticmethod
    def expand(blocks: Tensor, positions: Tensor, ratio: int, budget: int) -> Tensor:
        offsets = torch.arange(ratio, device=blocks.device)
        tokens = (
            (blocks.unsqueeze(-1) * ratio + offsets)
            .masked_fill(blocks.unsqueeze(-1) < 0, -1)
            .flatten(1)[:, :budget]
        )
        tail_start = (positions + 1) // ratio * ratio
        tail = tail_start.unsqueeze(1) + torch.arange(ratio - 1, device=blocks.device)
        tail = tail.masked_fill(tail > positions.unsqueeze(1), -1)
        return torch.cat([tokens, tail], dim=1)

    @staticmethod
    def mask(
        q: Tensor,
        k_tokens: Tensor,
        k_norm_w: Tensor,
        rope: Rope,
        positions: Tensor,
        cfg: MTPConfig,
    ) -> Tensor:
        """[T, T] boolean: query row may attend key column. q is [T, H, d] after norm+rope, k_tokens raw [T, d]."""
        n, ratio = q.shape[0], cfg.indexer_compress_ratio
        groups = n // ratio
        idx = torch.arange(n, device=q.device)
        if groups == 0:
            return idx.unsqueeze(0) <= idx.unsqueeze(1)
        pooled = k_tokens[: groups * ratio].unflatten(0, (groups, ratio)).mean(1)
        first = (
            positions[..., : groups * ratio : ratio]
            if positions.ndim == 2
            else positions[: groups * ratio : ratio]
        )
        k_blk = rope(gemma_rms(pooled, k_norm_w, cfg.rms_norm_eps).unsqueeze(1), first)[
            :, 0
        ]
        scores = torch.relu(torch.einsum("thd,gd->tgh", q.float(), k_blk.float())).sum(
            -1
        ) / math.sqrt(q.shape[-1])
        visible = (idx + 1) // ratio
        scores = scores.masked_fill(
            torch.arange(groups, device=q.device).unsqueeze(0) >= visible.unsqueeze(1),
            -math.inf,
        )
        top = scores.topk(min(cfg.block_topk, groups), dim=1)
        blocks = top.indices.masked_fill(torch.isinf(top.values), -1)
        pad = cfg.block_topk - blocks.shape[1]
        if pad > 0:
            blocks = F.pad(blocks, (0, pad), value=-1)
        tokens = QSASelect.expand(blocks, idx, ratio, cfg.indexer_budget)
        allowed = torch.zeros(n, n + 1, dtype=torch.bool, device=q.device)
        allowed.scatter_(1, tokens.masked_fill(tokens < 0, n), True)
        return allowed[:, :n] & (idx.unsqueeze(0) <= idx.unsqueeze(1))


class MTPDrafter(nn.Module):
    """One draft step over a whole sequence: (target HC state at t, token t+1) -> logits for token t+2, HC state."""

    def __init__(
        self,
        cfg: MTPConfig,
        dense: dict[str, Tensor],
        experts: list[list[Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # shared with the target, or (the QSA indexer) only choose tokens, so no gradient reaches them
        frozen = {"lm_head.weight", "model.language_model.embed_tokens.weight"}
        self.params = nn.ParameterDict(
            {
                k.replace(".", "__"): nn.Parameter(
                    v, requires_grad=k not in frozen and ".indexer." not in k
                )
                for k, v in dense.items()
            }
        )
        # experts[e] = [gate, up, down], dequantised and frozen
        self.experts = experts
        self.rope = Rope(cfg.rotary_dim, cfg.rope_theta, cfg.mrope_section)

    def w(self, name: str) -> Tensor:
        return self.params[name.replace(".", "__")]

    def norm(self, x: Tensor, name: str, group: int | None = None) -> Tensor:
        return gemma_rms(x, self.w(name), self.cfg.rms_norm_eps, group)

    def hc_mix(self, h: Tensor, prefix: str) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return hc_mix(
            h,
            self.w(f"{prefix}hc_norm.weight"),
            self.w(f"{prefix}input_mix_weight_down.weight"),
            self.w(f"{prefix}input_mix_weight_up.weight"),
            self.cfg,
        )

    def hc_combine(
        self, out: Tensor, res: tuple[Tensor, Tensor], prefix: str
    ) -> Tensor:
        h, hn = res
        inject = 2 * torch.sigmoid(
            F.linear(hn, self.w(f"{prefix}block_inject_weight.weight"))
            / self.cfg.hc_count
        )
        return (
            h.unflatten(-1, (self.cfg.hc_count, -1))
            + out.unsqueeze(-2) * inject.unsqueeze(-1)
        ).flatten(-2)

    def fuse(self, hc_hidden: Tensor, next_tokens: Tensor) -> Tensor:
        c = self.cfg
        emb = F.embedding(
            next_tokens, self.w("model.language_model.embed_tokens.weight")
        )
        emb = F.linear(
            self.norm(emb, "mtp.pre_fc_norm_embedding.weight"),
            self.w("mtp.fc_embedding.weight"),
        )
        h = self.norm(hc_hidden, "mtp.pre_fc_norm_hidden.weight").unflatten(
            -1, (c.hc_count, c.hidden_size)
        )
        return (
            emb.unsqueeze(-2) + F.linear(h, self.w("mtp.fc_hidden.weight"))
        ).flatten(-2)

    def attention(self, x: Tensor, positions: Tensor, sparse: bool) -> Tensor:
        c, a = self.cfg, MTP + "self_attn."
        n = x.shape[0]
        qg = F.linear(x, self.w(a + "q_proj.weight")).view(
            n, c.num_attention_heads, 2 * c.head_dim
        )
        q, gate = qg.chunk(2, dim=-1)
        k = F.linear(x, self.w(a + "k_proj.weight")).view(
            n, c.num_key_value_heads, c.head_dim
        )
        v = F.linear(x, self.w(a + "v_proj.weight")).view(
            n, c.num_key_value_heads, c.head_dim
        )
        q = self.rope(
            gemma_rms(q, self.w(a + "q_norm.weight"), c.rms_norm_eps), positions
        )
        k = self.rope(
            gemma_rms(k, self.w(a + "k_norm.weight"), c.rms_norm_eps), positions
        )
        if sparse:
            iqk = F.linear(x, self.w(a + "indexer.index_qk_proj.weight"))
            iq = iqk[:, : c.indexer_n_heads * c.indexer_head_dim].view(
                n, c.indexer_n_heads, c.indexer_head_dim
            )
            iq = self.rope(
                gemma_rms(iq, self.w(a + "indexer.q_layernorm.weight"), c.rms_norm_eps),
                positions,
            )
            ik = iqk[:, c.indexer_n_heads * c.indexer_head_dim :]
            allowed = QSASelect.mask(
                iq,
                ik,
                self.w(a + "indexer.k_layernorm.weight"),
                self.rope,
                positions,
                c,
            )
        else:
            allowed = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        rep = c.num_attention_heads // c.num_key_value_heads
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.repeat_interleave(rep, dim=1).transpose(0, 1),
            v.repeat_interleave(rep, dim=1).transpose(0, 1),
            attn_mask=allowed,
            scale=c.head_dim**-0.5,
        ).transpose(0, 1)
        out = out * torch.sigmoid(gate)
        return F.linear(out.reshape(n, -1), self.w(a + "o_proj.weight"))

    def moe(self, x: Tensor) -> Tensor:
        c, m = self.cfg, MTP + "mlp."
        probs = F.linear(x, self.w(m + "gate.weight")).float().softmax(-1)
        top_p, top_i = probs.topk(c.num_experts_per_tok, dim=-1)
        top_p = (top_p / top_p.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(x)
        if self.experts is not None:
            for e in top_i.unique().tolist():
                rows, slot = (top_i == e).nonzero(as_tuple=True)
                g, u, d = self.experts[e]
                y = F.linear(F.silu(F.linear(x[rows], g)) * F.linear(x[rows], u), d)
                out = out.index_add(0, rows, y * top_p[rows, slot].unsqueeze(-1))
        s = m + "shared_expert."
        shared = F.linear(
            F.silu(F.linear(x, self.w(s + "gate_proj.weight")))
            * F.linear(x, self.w(s + "up_proj.weight")),
            self.w(s + "down_proj.weight"),
        )
        return out + shared * torch.sigmoid(
            F.linear(x, self.w(m + "shared_expert_gate.weight"))
        )

    def forward(
        self,
        hc_hidden: Tensor,
        next_tokens: Tensor,
        positions: Tensor,
        sparse: bool = True,
    ) -> tuple[Tensor, Tensor]:
        h = self.fuse(hc_hidden, next_tokens)
        x, res = self.hc_mix(h, MTP + "attn_hyper_connection.")
        h = self.hc_combine(
            self.attention(x, positions, sparse), res, MTP + "attn_hyper_connection."
        )
        x, res = self.hc_mix(h, MTP + "mlp_hyper_connection.")
        h = self.hc_combine(self.moe(x), res, MTP + "mlp_hyper_connection.")
        mixed, _ = self.hc_mix(h, "mtp.hyper_connection_mixer.")
        return F.linear(mixed, self.w("lm_head.weight")), h

    def trainable(self) -> dict[str, Tensor]:
        return {
            k.replace("__", "."): p.detach()
            for k, p in self.params.items()
            if p.requires_grad
        }


def load_experts(
    path: Path, cfg: MTPConfig, device: str | torch.device
) -> list[list[Tensor]]:
    from safetensors import safe_open

    out = []
    with safe_open(str(path), framework="pt", device=str(device)) as f:
        for e in range(cfg.num_experts):
            mats = []
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"{MTP}mlp.experts.{e}.{proj}."
                mats.append(
                    dequant_int4(
                        f.get_tensor(base + "weight_packed"),
                        f.get_tensor(base + "weight_scale"),
                    )
                )
            out.append(mats)
    return out


def load_dense(path: Path, device: str | torch.device) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path), device=str(device))
