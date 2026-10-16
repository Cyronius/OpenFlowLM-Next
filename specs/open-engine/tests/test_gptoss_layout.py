# Traces: OPEN-GPTOSS-EXPERT-LAYOUT (canonical spec: specs/open-engine/spec.md)
"""GPT-OSS's expert layout (open_kernels/model/gptoss_layout.py) on synthetic weights; the real-GGUF slice is manual."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import gptoss_layout as G
from mxfp4 import KVALUES, e8m0
from recipes import qwen36moe as M

RNG = np.random.default_rng(20261016)


def _bf16(n):
    return (RNG.standard_normal(n).astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)


def _projection(rows, K):
    codes = RNG.integers(0, 16, (rows, K), dtype=np.uint8)
    exps = RNG.integers(110, 130, (rows, K // 32), dtype=np.uint8)
    return codes, exps, _bf16(rows)


def _values(codes, exps):
    return KVALUES[codes] * np.repeat(e8m0(exps), 32, axis=1)


def test_2880_is_ten_tiles_and_a_tile_pair_is_the_q4_1_weight_element():
    """288 columns: 256 would pad K, 320 would not fit the 5120-byte half of the shared w element."""
    assert 2880 == 10 * G.TILE_COLS
    assert G.CALL_BYTES == M.CALL_BYTES == 2 * M.CHUNK
    assert 2880 % 256 and 2880 % 320 == 0
    assert G.TILE_ROWS * 320 // 2 + G.TILE_ROWS * 10 > G.TILE_BYTES
    assert G.BIAS_OFF % 64 == 0 and G.NIB_PLANE % 64 == 0 and G.EXP_OFF % 32 == 0


def test_a_tile_round_trips_codes_exponents_and_bias():
    codes, exps, bias = _projection(32, G.TILE_COLS)
    c2, e2, b2 = G.unpack_tile(G.pack_tile(codes, exps, bias))
    assert np.array_equal(c2, codes) and np.array_equal(e2, exps) and np.array_equal(b2, bias)
    _, _, b0 = G.unpack_tile(G.pack_tile(codes, exps, None))
    assert not b0.any()


def test_the_nibbles_follow_the_q4_kernels_index_law():
    """One code at (19, 200) lands at nibble k * 16 + r % 16 of plane r // 16, and nowhere else."""
    codes = np.zeros((32, G.TILE_COLS), np.uint8)
    codes[19, 200] = 0xB
    t = G.pack_tile(codes, np.zeros((32, 9), np.uint8))
    p = 200 * 16 + 19 % 16
    byte = G.NIB_PLANE * (19 // 16) + p // 2
    assert t[byte] == (0xB << 4 if p % 2 else 0xB)
    assert np.count_nonzero(t) == 1


def test_a_bias_that_is_not_a_bf16_value_is_refused():
    codes, exps, _ = _projection(32, G.TILE_COLS)
    with pytest.raises(ValueError, match="bf16"):
        G.pack_tile(codes, exps, np.full(32, 1.0 + 2 ** -12, np.float32))


def test_a_projection_packs_in_band_order_and_reads_back_exactly():
    rows, K = 128, 576
    codes, exps, bias = _projection(rows, K)
    bands = G.pack_projection(codes, exps, bias)
    assert bands.shape == (rows // 64, 2 * K // G.TILE_COLS, G.TILE_BYTES)
    w, b = G.dequant_projection(bands, rows, K)
    assert np.array_equal(w, _values(codes, exps)) and np.array_equal(b, bias)
    # the bias rides only in k-tile 0 of each band half
    assert all(G.unpack_tile(bands[0, i])[2].any() == (i < 2) for i in range(bands.shape[1]))


@pytest.mark.parametrize("rows, K", [(96, 288), (64, 400)])
def test_a_projection_that_is_not_whole_bands_and_tiles_is_refused(rows, K):
    """Flooring would drop the rows or columns past the last whole band or tile without a word."""
    codes, exps, bias = _projection(rows, K)
    with pytest.raises(ValueError, match="whole"):
        G.pack_projection(codes, exps, bias)
    with pytest.raises(ValueError, match="whole"):
        G.dequant_projection(np.zeros(rows * K, np.uint8), rows, K)


def test_an_expert_block_interleaves_gate_and_up_by_band_then_down():
    rows = K = 576
    g, u, d = (_projection(rows, K) for _ in range(3))
    blk = G.pack_expert(g, u, d)
    L = G.layout(K, rows, experts=4, topk=2)
    assert blk.size == L.expert_bytes
    band = L.band_bytes_up
    gp, up = G.pack_projection(*g).reshape(-1), G.pack_projection(*u).reshape(-1)
    assert np.array_equal(blk[:band], gp[:band]) and np.array_equal(blk[band:2 * band], up[:band])
    assert np.array_equal(blk[L.down_off:], G.pack_projection(*d).reshape(-1))


def test_45_bands_go_six_six_six_six_six_five_five_five():
    assert G.deal(2880, 8) == (6, 6, 6, 6, 6, 5, 5, 5)
    assert G.deal(2880, 8, pad=True) == (6,) * 8
    assert G.deal(4096, 8) == (8,) * 8
    with pytest.raises(ValueError, match="64-row bands"):
        G.deal(2900, 8)


def test_the_chosen_layout_survives_the_three_checks_qwen36moe_refuses_it_on():
    """Core assignment, core scratch and element counts pass here; qwen36moe still refuses the same model."""
    L = G.layout()
    r = G.check(L)
    assert sum(L.up_bands) * 64 == 2880 == sum(L.down_bands) * 64
    assert L.l1 == r["l1"] == 47056 and L.l1 <= G.L1_BUDGET
    assert (L.xm_elems, L.h_elems) == (2, 3) and L.xm_elems <= G.X_DEPTH
    assert L.expert_bytes == 3 * 45 * 10 * G.CALL_BYTES
    # with o_proj's 4096-wide table on the same core it still fits
    assert G.check(G.layout(table_k=4096))["l1"] == 49792
    # and qwen36moe's own layout still refuses the same model, by name
    spec = dataclasses.replace(_spec(), hidden=3072, moe_intermediate=3072)
    with pytest.raises(M.OpRangeError, match="stripes per projection"):
        M.common(spec)


def _spec():
    from recipes.spec import ModelSpec
    from test_gptoss import HF_GPTOSS_20B
    return ModelSpec.from_hf_config(HF_GPTOSS_20B)


def test_every_slot_reads_its_whole_expert_block_once_and_one_stride_patches_it():
    L = G.layout()
    seq = G.host_sequence(L)
    for s in range(L.topk):
        n = sum(t[5] for t in seq if t[3] == "pool" and t[6] == s)
        assert n == L.expert_bytes
    s0 = [t[4] for t in seq if t[3] == "pool" and t[6] == 0]
    s3 = [t[4] - 3 * L.expert_bytes for t in seq if t[3] == "pool" and t[6] == 3]
    assert s0 == s3


def test_the_checks_catch_a_broken_sequence(monkeypatch):
    """Each check fails on the mistake it exists for."""
    L = G.layout()
    good = G.host_sequence

    def shifted(L, A=G.Act()):            # slot 1's down fills one band late
        seq = good(L, A)
        return [t if not (t[3] == "pool" and t[6] == 1 and t[4] >= L.expert_bytes + L.down_off)
                else t[:4] + (t[4] + L.band_bytes_down,) + t[5:] for t in seq]

    monkeypatch.setattr(G, "host_sequence", shifted)
    with pytest.raises(AssertionError, match="slot 1"):
        G.check(L)

    def short(L, A=G.Act()):              # core 7's slot-3 gate|up fill one band short
        seq = good(L, A)
        i = next(i for i, t in enumerate(seq) if t[3] == "pool" and t[2] == 7 and t[6] == 3
                 and t[4] < 3 * L.expert_bytes + L.down_off)
        t = seq[i]
        seq[i] = t[:5] + (t[5] - 2 * L.band_bytes_up,) + t[6:]
        return seq

    monkeypatch.setattr(G, "host_sequence", short)
    with pytest.raises(AssertionError, match="core 7 w"):
        G.check(L)
    monkeypatch.setattr(G, "host_sequence", good)
    with pytest.raises(AssertionError, match="core scratch"):
        G.check(G.layout(table_k=12288))
    with pytest.raises(ValueError, match="x elements"):
        G.check(G.layout(hidden=4608, ff=2880))


def _bar(seq):
    return next(i for i, t in enumerate(seq) if t[0] == "finish")


def _no_barrier(seq, L, A):
    return seq[:_bar(seq)] + seq[_bar(seq) + 1:]


def _h_fills_before_the_barrier(seq, L, A):
    b = _bar(seq)
    return seq[:b] + [t for t in seq[b:] if t[1] == "x"] + [t for t in seq[b:] if t[1] != "x"]


def _slot_0_reads_slot_1s_h(seq, L, A):
    return [t[:4] + (A.h + A.h_stride,) + t[5:] if t[1] == "x" and t[6] == 0 else t for t in seq]


def _outputs_all_at_row_0(seq, L, A):
    return [t[:4] + (0,) + t[5:] if t[3] == "xres" else t for t in seq]


def _outputs_of_cores_0_and_1_swapped(seq, L, A):
    off = {t[2]: t[4] for t in seq if t[3] == "xres"}
    return [t[:4] + (off[1 - t[2]],) + t[5:] if t[3] == "xres" and t[2] < 2 else t for t in seq]


def _residual_headers_all_read_core_0s_rows(seq, L, A):
    return [t[:4] + (A.res,) + t[5:] if t[1] == "w" and t[3] == "act" and t[4] != A.rout else t for t in seq]


def _slots_1_and_2_gate_up_issued_swapped(seq, L, A):
    at = {s: [i for i, t in enumerate(seq) if t[3] == "pool" and t[6] == s
              and t[4] < s * L.expert_bytes + L.down_off] for s in (1, 2)}
    out = list(seq)
    for i, j in zip(at[1], at[2]):
        out[i], out[j] = seq[j], seq[i]
    return out


@pytest.mark.parametrize("edit, message", [
    (_no_barrier, "barrier"),
    (_h_fills_before_the_barrier, "barrier"),
    (_slot_0_reads_slot_1s_h, "each slot's own h"),
    (_outputs_all_at_row_0, "core 1: its drains"),
    (_outputs_of_cores_0_and_1_swapped, "core 0: its drains"),
    (_residual_headers_all_read_core_0s_rows, "core 1: its header fills"),
    (_slots_1_and_2_gate_up_issued_swapped, "core 0: its w fills are not in its program's order"),
])
def test_the_checks_catch_a_misplaced_barrier_or_act_offset(monkeypatch, edit, message):
    """Each sequence moves every byte the balances count and still computes a wrong answer."""
    good = G.host_sequence
    monkeypatch.setattr(G, "host_sequence", lambda L, A=G.Act(): edit(good(L, A), L, A))
    with pytest.raises(AssertionError, match=message):
        G.check(G.layout())


def test_more_cores_deal_and_fit_but_their_streams_need_a_memory_tile_split():
    """One shim stream per core runs out of the shim's 16 channels past 15 cores; the split is not modeled."""
    for n, bands, l1 in ((16, (3,) * 13 + (2,) * 3, 44752), (24, (2,) * 21 + (1,) * 3, 43984)):
        L = G.layout(n_cores=n)
        assert L.up_bands == L.down_bands == bands and L.l1 == l1
        with pytest.raises(AssertionError, match=f"{n + 1} fill and {n} drain streams"):
            G.check(L)


def test_the_layouts_arithmetic_matches_the_fp64_reference():
    """Within bf16 rounding of replica_gptoss, and a band dealt to nobody is not."""
    from replica_gptoss import expert_ffn
    H = F = 576
    E, k = 4, 2
    L = G.layout(H, F, experts=E, topk=k)
    W = {p: {e: _values(*_projection(F if p != "down" else H, H if p != "down" else F)[:2]) / 64
             for e in range(E)} for p in ("gate", "up", "down")}
    B = {p: {e: _bf16(H) * 0.1 for e in range(E)} for p in ("gate", "up", "down")}
    xm = RNG.standard_normal(H).astype(np.float32)
    res = RNG.standard_normal(H).astype(np.float32)
    ids, wts = [3, 1], [0.6, 0.4]
    ref = res.astype(np.float64)
    for e, w in zip(ids, wts):
        ref = ref + w * expert_ffn(xm, W["gate"][e], B["gate"][e], W["up"][e], B["up"][e],
                                   W["down"][e], B["down"][e])
    got = G.emulate_block(L, xm, res, W["gate"], B["gate"], W["up"], B["up"], W["down"], B["down"], ids, wts)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref - res)
    assert rel < 5e-3, rel
    assert L.down_bands == (2, 1, 1, 1, 1, 1, 1, 1)
    bad = dataclasses.replace(L, down_bands=(1,) * 8)                      # the ninth band dealt to nobody
    got2 = G.emulate_block(bad, xm, res, W["gate"], B["gate"], W["up"], B["up"], W["down"], B["down"], ids, wts)
    assert np.linalg.norm(got2 - ref) / np.linalg.norm(ref - res) > 0.1


def test_the_bytes_the_options_stream():
    """The tile pad costs 5120/4896 over the GGUF's bytes; q4_1 costs 5/4.25 per value plus the K pad; int8 2.4x."""
    ideal = G.expert_stream_bytes("mxfp4_ideal")
    tiles = G.expert_stream_bytes("mxfp4")
    q41 = G.expert_stream_bytes("q4_1")
    assert ideal == 4 * 24 * 3 * 2880 * 2880 * 17 // 32
    assert tiles * 4896 == ideal * 5120
    assert q41 == pytest.approx(ideal * (5 / 4.25) * (3072 / 2880), rel=1e-9)
    assert G.expert_stream_bytes("q4_1", pad_rows=True) == pytest.approx(q41 * 3072 / 2880, rel=1e-9)
    # gemv_q8.h's half-tile is 16 rows x 256 K in 5120 B, not the container's 8704 B per 32 rows
    q8 = G.expert_stream_bytes("q8")
    assert q8 == 4 * 24 * 3 * 2880 * 3072 * 5 // 4 == 3_185_049_600
    assert q8 * 5 == tiles * 12
