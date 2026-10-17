# Traces: OPEN-FAMILY-HUNYUAN (canonical spec: specs/open-engine/spec.md)
"""gguf_src.py must pack as pools.cpp does: a slice builds both its pools and its reference from it, so only this sees drift."""
from __future__ import annotations

import struct

import numpy as np
import pytest

import gguf_pool as GP
from gguf_src import GgufSource


def _s(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors):
    """GGUF v3 with one string KV and (name, dims with ne0 first, ggml type, raw blocks) tensors."""
    hdr = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    hdr += _s(b"general.architecture") + struct.pack("<I", 8) + _s(b"hunyuan-dense")
    offs, rel = [], 0
    for *_, raw in tensors:
        offs.append(rel)
        rel = (rel + len(raw) + 31) // 32 * 32
    for (name, dims, ty, _), off in zip(tensors, offs):
        hdr += _s(name.encode()) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        hdr += struct.pack("<IQ", ty, off)
    data = bytearray(hdr.ljust((len(hdr) + 31) // 32 * 32, b"\0"))
    base = len(data)
    for (*_, raw), off in zip(tensors, offs):
        data += b"\0" * (base + off - len(data)) + raw
    path.write_bytes(bytes(data))


@pytest.fixture
def tied(tmp_path):
    """A Q4_1 and a Q4_0 projection, and a 40-row Q8_0 token_embd that is also the head (no output.weight)."""
    rng = np.random.default_rng(7)
    q41 = GP.random_q4_blocks(128, 512, rng, gguf_type="Q4_1")
    q40 = GP.random_q4_blocks(128, 512, rng, gguf_type="Q4_0")
    emb = GP.random_q8_0_blocks(40, 256, rng)
    p = tmp_path / "tied.gguf"
    write_gguf(p, [("blk.0.attn_q.weight", [512, 128], 3, q41.tobytes()),
                   ("blk.0.attn_k.weight", [512, 128], 2, q40.tobytes()),
                   ("token_embd.weight", [256, 40], 8, emb.tobytes())])
    return GgufSource(p), q41, q40, emb


def test_q4_1_and_q4_0_pack_byte_for_byte_as_gguf_pool(tied):
    src, q41, q40, _ = tied
    nch = 128 * 512 // 8192
    got = src.std_perm_gguf("model.layers.0.self_attn.q_proj.weight", nch, 512)
    assert np.array_equal(got, GP.pack_q4_pool(q41, 2, "Q4_1"))
    got = src.std_perm_gguf("model.layers.0.self_attn.k_proj.weight", nch, 512)
    assert np.array_equal(got, GP.pack_q4_pool(q40, 2, "Q4_0"))


def test_tied_q8_0_head_requantizes_and_pads_to_a_whole_band(tied, tmp_path):
    src, _, _, emb = tied
    pool = src.std_perm_gguf("lm_head.weight", 2, 256)
    zero_rows = tmp_path / "zero-rows.gguf"
    write_gguf(zero_rows, [("token_embd.weight", [256, 64], 8,
                            np.concatenate([emb, np.zeros((24, 8, 34), np.uint8)]).tobytes())])
    assert np.array_equal(pool, GgufSource(zero_rows).std_perm_gguf("lm_head.weight", 2, 256))
    w = GP.dequant_pool(pool, 64, 256, 2, "Q4_1")
    assert not w[40:].any()
    d, m, q = src.q4_1_blocks("lm_head.weight")
    assert np.array_equal(w[:40], (q.reshape(40, -1, 32) * d[..., None] + m[..., None]).reshape(40, 256))
    assert np.all(np.abs(w[:40] - GP.dequant_q8_0(emb)) <= 0.52 * np.repeat(d, 32, axis=1) + 1e-6)
    assert np.array_equal(d, d.astype(np.float16).astype(np.float32))
    assert np.array_equal(m, m.astype(np.float16).astype(np.float32))


@pytest.mark.parametrize("nch,in_dim", [(1, 256), (4, 256), (4, 512)])
def test_a_plan_that_does_not_fit_the_file_is_refused(tied, nch, in_dim):
    """32 rows (narrower than the file), 128 rows (a band or more past it), and the wrong width."""
    with pytest.raises(ValueError, match="does not fill"):
        tied[0].std_perm_gguf("lm_head.weight", nch, in_dim)


def test_embedding_rows_are_the_exact_q8_0_values(tied):
    src, _, _, emb = tied
    assert np.array_equal(src.embed(5, 256), GP.dequant_q8_0(emb)[5])
    with pytest.raises(ValueError, match="outside the embedding"):
        src.embed(40, 256)
