"""Stage-3 drafter training: rebuilding sequences from capture dumps, the KL step, and the export."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "sia-oss/bench/dfz"), str(REPO / "tests")]

from drafter_train import (  # noqa: E402
    Dumps,
    Segment,
    TargetHead,
    acceptance,
    export,
    kl_to_target,
    train,
)
from mtp_ref import MTPDrafter  # noqa: E402
from test_arc_agi3_mtp_ref import TINY, dense_shapes  # noqa: E402

WIDTH = TINY.hidden_size * TINY.hc_count


def dump(
    path: Path, i: int, req: list[int], ids: list[int], pos: list[int], dec: int = 0
) -> None:
    hc = torch.randn(len(ids), WIDTH, dtype=torch.bfloat16).view(torch.int16).numpy()
    np.savez(
        path / f"{i:09d}.npz",
        req=np.asarray(req, np.int32),
        ids=np.asarray(ids, np.int32),
        pos=np.asarray(pos, np.int32),
        hc=hc,
        dec=np.full(len(ids), dec, np.int8),
    )


@pytest.fixture
def parts() -> tuple[MTPDrafter, TargetHead]:
    g = torch.Generator().manual_seed(0)
    dense = {
        k: torch.randn(s, generator=g) * 0.2 for k, s in dense_shapes(TINY).items()
    }
    m = TINY.moe_intermediate_size
    experts = [
        [
            torch.randn(m, TINY.hidden_size) * 0.2,
            torch.randn(m, TINY.hidden_size) * 0.2,
            torch.randn(TINY.hidden_size, m) * 0.2,
        ]
        for _ in range(TINY.num_experts)
    ]
    head = TargetHead(
        TINY,
        torch.zeros(WIDTH),
        torch.randn(TINY.hc_lowrank, WIDTH, generator=g) * 0.2,
        torch.randn(WIDTH, TINY.hc_lowrank, generator=g) * 0.2,
        torch.randn(TINY.vocab_size, TINY.hidden_size, generator=g),
    )
    return MTPDrafter(TINY, dense, experts), head


def segments(n: int, length: int) -> list[Segment]:
    g = torch.Generator().manual_seed(3)
    return [
        Segment(
            torch.randint(0, TINY.vocab_size, (length,), generator=g),
            torch.arange(length).expand(3, -1).clone(),
            torch.randn(length, WIDTH, generator=g).bfloat16(),
            torch.arange(length) >= length // 2,
        )
        for _ in range(n)
    ]


class TestDumps:
    def test_rebuilds_interleaved_chunked_and_reused_slots(
        self, tmp_path: Path
    ) -> None:
        # slot 5 prefills request A in two chunks, slot 7 runs B; both decode together; slot 5 is then reused by C
        dump(
            tmp_path,
            0,
            [5] * 70 + [7] * 66,
            list(range(70)) + list(range(100, 166)),
            list(range(70)) + list(range(66)),
        )
        dump(tmp_path, 1, [5] * 10, list(range(70, 80)), list(range(70, 80)))
        for step in range(3):
            dump(
                tmp_path,
                2 + step,
                [5, 7],
                [80 + step, 166 + step],
                [80 + step, 66 + step],
            )
        dump(tmp_path, 9, [5] * 65, list(range(500, 565)), list(range(65)))
        segs = sorted(Dumps.load(tmp_path, window=4096), key=lambda s: int(s.ids[0]))
        assert [len(s) for s in segs] == [83, 69, 65]
        assert torch.equal(segs[0].ids, torch.arange(83)) and torch.equal(
            segs[1].ids, torch.arange(100, 169)
        )
        assert segs[2].pos.shape == (3, 65) and segs[2].hc.dtype == torch.bfloat16

    def test_window_keeps_the_tail_and_short_requests_drop(
        self, tmp_path: Path
    ) -> None:
        dump(
            tmp_path,
            0,
            [1] * 300 + [2] * 10,
            list(range(300)) + list(range(10)),
            list(range(300)) + list(range(10)),
        )
        segs = Dumps.load(tmp_path, window=128)
        assert len(segs) == 1 and torch.equal(segs[0].ids, torch.arange(172, 300))


def test_kl_is_zero_when_drafter_matches_target_and_positive_otherwise() -> None:
    target = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    top_p, top_ids = target.softmax(-1).topk(4)
    assert float(
        kl_to_target(target.log_softmax(-1) + 5.0, top_ids, top_p)
    ) == pytest.approx(0.0, abs=1e-6)
    assert float(kl_to_target(torch.zeros(1, 4), top_ids, top_p)) > 0.05


def test_training_lowers_loss_and_export_touches_trained_tensors_only(
    parts: tuple, tmp_path: Path
) -> None:
    from safetensors.torch import load_file, save_file

    drafter, head = parts
    for p in drafter.params.values():
        if p.requires_grad:
            p.data = p.data.float()
    segs = segments(4, 40)
    log: list[dict] = []
    train(drafter, segs, head, "cpu", epochs=12, lr=3e-3, log=log, depths=2)
    first, last = (
        np.mean([r["loss"] for r in log[:4]]),
        np.mean([r["loss"] for r in log[-4:]]),
    )
    assert last < first * 0.9
    assert all(
        0.0 <= a <= 1.0 for a in acceptance(drafter, segs[:1], head, "cpu", depths=2)
    )

    original = {k: torch.randn(s).bfloat16() for k, s in dense_shapes(TINY).items()}
    save_file(original, str(tmp_path / "orig.safetensors"))
    trained = export(
        tmp_path / "orig.safetensors",
        drafter,
        tmp_path / "out" / "mtp-dense.safetensors",
    )
    saved = load_file(str(tmp_path / "out" / "mtp-dense.safetensors"))
    assert set(saved) == set(original) and all(
        v.dtype == torch.bfloat16 for v in saved.values()
    )
    assert all(torch.equal(saved[k], original[k]) for k in original if k not in trained)
    assert all(not torch.equal(saved[k], original[k]) for k in trained) and not any(
        ".indexer." in k for k in trained
    )


def plus_one(hidden, next_tokens, positions):
    """Stub drafter that guesses token + 1 and passes its input state through."""
    return torch.nn.functional.one_hot(
        (next_tokens + 1) % TINY.vocab_size, TINY.vocab_size
    ).float(), hidden


def test_chain_aligns_tokens_targets_and_generated_masks(parts: tuple) -> None:
    from drafter_train import chain

    _, head = parts
    seg = segments(1, 12)[0]
    seg.ids = torch.arange(12)
    out = chain(plus_one, seg, head, "cpu", depths=3)
    assert [o[0].shape[0] for o in out] == [10, 9, 8]
    for d, (logits, top_ids, _, target, scored) in enumerate(out, start=1):
        assert torch.equal(
            target, torch.arange(d + 1, 12)
        )  # row r predicts token r+d+1
        assert torch.equal(logits.argmax(-1), target)  # input token r+d, so +1 is right
        assert torch.equal(scored, seg.gen[d + 1 :])
        assert torch.equal(
            top_ids, head.top_k(seg.hc[1:])[0][d - 1 : d - 1 + len(target)]
        )


def test_acceptance_is_reported_per_depth_on_generated_tokens_only(
    parts: tuple,
) -> None:
    from drafter_train import acceptance

    _, head = parts
    seg = segments(1, 20)[0]
    seg.ids = torch.arange(20)
    seg.ids[15] = 0  # drafts of token 15 (expects 15) and token 16 (fed 0) miss
    acc = acceptance(plus_one, [seg], head, "cpu", depths=2)
    assert acc == pytest.approx(
        [8 / 10, 8 / 10]
    )  # generated tokens 10-19 at both depths


def test_decode_flags_survive_the_rebuild(tmp_path: Path) -> None:
    dump(tmp_path, 0, [3] * 64, list(range(64)), list(range(64)))
    for step in range(4):
        dump(tmp_path, 1 + step, [3], [64 + step], [64 + step], dec=1)
    (seg,) = Dumps.load(tmp_path, window=4096)
    assert seg.gen.tolist() == [False] * 64 + [True] * 4
