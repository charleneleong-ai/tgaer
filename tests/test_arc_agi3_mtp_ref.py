"""Plain-PyTorch MTP drafter: INT4 unpacking, MRoPE, QSA selection, causality and what trains."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sia-oss/bench/dfz"))

from mtp_ref import (  # noqa: E402
    MTPConfig,
    MTPDrafter,
    QSASelect,
    Rope,
    dequant_int4,
)

TINY = MTPConfig(
    hidden_size=32,
    hc_count=4,
    hc_lowrank=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    partial_rotary_factor=0.5,
    mrope_section=(2, 1, 1),
    num_experts=6,
    num_experts_per_tok=2,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    indexer_n_heads=2,
    indexer_head_dim=8,
    indexer_budget=8,
    indexer_compress_ratio=4,
    vocab_size=50,
)


def dense_shapes(c: MTPConfig) -> dict[str, tuple[int, ...]]:
    h, hc, a = c.hidden_size, c.hidden_size * c.hc_count, "mtp.layers.0.self_attn."
    shapes = {
        "lm_head.weight": (c.vocab_size, h),
        "model.language_model.embed_tokens.weight": (c.vocab_size, h),
        "mtp.fc_embedding.weight": (h, h),
        "mtp.fc_hidden.weight": (h, h),
        "mtp.pre_fc_norm_embedding.weight": (h,),
        "mtp.pre_fc_norm_hidden.weight": (hc,),
        a + "q_proj.weight": (2 * c.num_attention_heads * c.head_dim, h),
        a + "k_proj.weight": (c.num_key_value_heads * c.head_dim, h),
        a + "v_proj.weight": (c.num_key_value_heads * c.head_dim, h),
        a + "o_proj.weight": (h, c.num_attention_heads * c.head_dim),
        a + "q_norm.weight": (c.head_dim,),
        a + "k_norm.weight": (c.head_dim,),
        a + "indexer.index_qk_proj.weight": (
            (c.indexer_n_heads + 1) * c.indexer_head_dim,
            h,
        ),
        a + "indexer.q_layernorm.weight": (c.indexer_head_dim,),
        a + "indexer.k_layernorm.weight": (c.indexer_head_dim,),
        "mtp.layers.0.mlp.gate.weight": (c.num_experts, h),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (
            c.shared_expert_intermediate_size,
            h,
        ),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": (
            c.shared_expert_intermediate_size,
            h,
        ),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": (
            h,
            c.shared_expert_intermediate_size,
        ),
        "mtp.layers.0.mlp.shared_expert_gate.weight": (1, h),
    }
    for p in (
        "mtp.layers.0.attn_hyper_connection.",
        "mtp.layers.0.mlp_hyper_connection.",
        "mtp.hyper_connection_mixer.",
    ):
        shapes |= {
            p + "hc_norm.weight": (hc,),
            p + "input_mix_weight_down.weight": (c.hc_lowrank, hc),
            p + "input_mix_weight_up.weight": (hc, c.hc_lowrank),
        }
        if "mixer" not in p:
            shapes[p + "block_inject_weight.weight"] = (c.hc_count, hc)
    return shapes


@pytest.fixture
def drafter() -> MTPDrafter:
    g = torch.Generator().manual_seed(0)
    dense = {
        k: torch.randn(s, generator=g) * 0.2 for k, s in dense_shapes(TINY).items()
    }
    m = TINY.moe_intermediate_size
    experts = [
        [
            torch.randn(m, TINY.hidden_size, generator=g) * 0.2,
            torch.randn(m, TINY.hidden_size, generator=g) * 0.2,
            torch.randn(TINY.hidden_size, m, generator=g) * 0.2,
        ]
        for _ in range(TINY.num_experts)
    ]
    return MTPDrafter(TINY, dense, experts)


def inputs(n: int, seed: int = 1) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, TINY.hidden_size * TINY.hc_count, generator=g),
        torch.randint(0, TINY.vocab_size, (n,), generator=g),
        torch.arange(n),
    )


def test_int4_unpacks_low_nibble_first_with_offset_eight() -> None:
    q = torch.arange(-8, 8).repeat(4)  # 64 values, two groups of 32
    nibbles = (q + 8).view(-1, 8)
    packed = sum((nibbles[:, i] << (4 * i)) for i in range(8)).to(torch.int64)
    packed = (
        torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32).view(1, -1)
    )
    scale = torch.tensor([[0.5, 2.0]])
    expected = torch.cat([q[:32] * 0.5, q[32:] * 2.0]).view(1, -1)
    assert torch.equal(dequant_int4(packed, scale), expected)


class TestRope:
    rope = Rope(rotary_dim=8, theta=1e4, section=(2, 1, 1))

    def test_text_positions_equal_identical_three_axis_positions(self) -> None:
        x, pos = torch.randn(5, 2, 16), torch.arange(5)
        assert torch.allclose(self.rope(x, pos), self.rope(x, pos.expand(3, -1)))

    def test_rotation_keeps_norm_and_leaves_the_tail(self) -> None:
        x, pos = (
            torch.randn(5, 2, 16),
            torch.tensor([[0, 3, 7, 9, 2], [1, 1, 1, 4, 4], [2, 5, 5, 5, 6]]),
        )
        y = self.rope(x, pos)
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
        assert torch.equal(y[..., 8:], x[..., 8:])


class TestQSASelect:
    def test_dense_when_groups_fit_the_budget(self) -> None:
        n = 8  # 2 complete groups <= block_topk 2
        q, k = torch.randn(n, 2, 8), torch.randn(n, 8)
        allowed = QSASelect.mask(
            q, k, torch.zeros(8), Rope(4, 1e4, (2, 1, 1)), torch.arange(n), TINY
        )
        assert torch.equal(allowed, torch.ones(n, n, dtype=torch.bool).tril())

    def test_long_rows_keep_budget_plus_their_own_partial_group(self) -> None:
        n = 40
        q, k = torch.randn(n, 2, 8), torch.randn(n, 8)
        allowed = QSASelect.mask(
            q, k, torch.zeros(8), Rope(4, 1e4, (2, 1, 1)), torch.arange(n), TINY
        )
        assert not allowed.triu(1).any()
        assert (
            allowed.sum(1) <= TINY.indexer_budget + TINY.indexer_compress_ratio - 1
        ).all()
        for p in (21, 38):  # partial-group tokens are always visible to their own row
            start = (p + 1) // 4 * 4
            assert allowed[p, start : p + 1].all()


class TestDrafter:
    @pytest.mark.parametrize("sparse", [True, False])
    def test_causal(self, drafter: MTPDrafter, sparse: bool) -> None:
        h, t, pos = inputs(24)
        logits, hc = drafter(h, t, pos, sparse=sparse)
        h2, t2 = h.clone(), t.clone()
        h2[15:], t2[15:] = torch.randn_like(h2[15:]), (t2[15:] + 1) % TINY.vocab_size
        logits2, hc2 = drafter(h2, t2, pos, sparse=sparse)
        assert logits.shape == (24, TINY.vocab_size) and hc.shape == h.shape
        assert torch.allclose(logits[:15], logits2[:15], atol=1e-5) and torch.allclose(
            hc[:15], hc2[:15], atol=1e-5
        )
        assert not torch.allclose(logits[15:], logits2[15:])

    def test_only_dense_mtp_weights_train(self, drafter: MTPDrafter) -> None:
        h, t, pos = inputs(12)
        logits, _ = drafter(h, t, pos)
        logits.logsumexp(-1).sum().backward()
        grads = {k.replace("__", "."): p.grad for k, p in drafter.params.items()}
        frozen = {k for k in grads if not k.startswith("mtp.") or ".indexer." in k}
        assert frozen == {
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
        } | {k for k in grads if ".indexer." in k}
        assert all(grads[k] is None for k in frozen)
        assert all(
            g is not None and g.abs().sum() > 0
            for k, g in grads.items()
            if k not in frozen
        )
        assert set(drafter.trainable()) == set(grads) - frozen


def test_config_reads_text_config_and_mtp_rope(tmp_path: Path) -> None:
    raw = {
        "text_config": {
            "hidden_size": 2560,
            "num_experts": 512,
            "indexer_budget": 2048,
            "mtp": {"rope_theta": 5e6},
            "rope_parameters": {"mrope_section": [11, 11, 10], "rope_theta": 1e7},
            "unrelated": 1,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    c = MTPConfig.from_json(tmp_path / "config.json")
    assert (c.rope_theta, c.mrope_section, c.rotary_dim, c.block_topk) == (
        5e6,
        (11, 11, 10),
        64,
        512,
    )


def test_gate_counts_hits_on_generated_tokens_only() -> None:
    pytest.importorskip("numpy")
    from mtp_gate import depth1_acceptance

    vocab = 7

    def plus_one(
        hidden, next_tokens, positions
    ):  # drafter that predicts token + 1 from token t+1
        return torch.nn.functional.one_hot(
            (next_tokens + 1) % vocab, vocab
        ).float(), hidden

    tokens = torch.tensor([3, 4, 5, 6, 0, 1, 1, 2])  # one break: 1 -> 1
    hits, trials = depth1_acceptance(
        plus_one, torch.zeros(8, 4), tokens, first_output=4
    )
    assert (hits, trials) == (
        3,
        4,
    )  # predicts tokens 4..7 from rows 2..5; the 1 -> 1 step misses
