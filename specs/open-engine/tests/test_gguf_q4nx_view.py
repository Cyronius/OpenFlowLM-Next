"""OPEN-GGUF-Q4NX-VIEW: a qwen35 GGUF served as the q4nx container its kernel set's plan reads.

The synthetic GGUF here is the one src/open_qwen36/gguf_pack_test.cpp writes (same LCG, same
tensors); both views must hash to VIEW_FNV1A, which pins the C++ view to this NumPy one byte
for byte. The other tests check what the conversions mean, independently of either view.
"""
# Traces: OPEN-GGUF-Q4NX-VIEW, OPEN-CONVERT-QWEN35-VHEADS
from __future__ import annotations

import struct

import numpy as np
import pytest

from gguf_q4nx import GgufAsQ4nx, GgufReader, file_head, requant_q8_blocks
from q4nx import bf16_to_f32
from recipes import pack

VIEW_FNV1A = 0xE255B1CF09155467

GEOMETRY = {"qwen35.block_count": 4, "qwen35.embedding_length": 256, "qwen35.feed_forward_length": 256,
            "qwen35.attention.head_count": 4, "qwen35.attention.head_count_kv": 2,
            "qwen35.attention.key_length": 64, "qwen35.attention.value_length": 64,
            "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.state_size": 64, "qwen35.ssm.group_count": 2,
            "qwen35.ssm.time_step_rank": 4, "qwen35.ssm.inner_size": 256,
            "qwen35.full_attention_interval": 4, "qwen35.rope.dimension_count": 16}

TENSORS = [
    ("token_embd.weight", 3, [256, 100]), ("output.weight", 14, [256, 100]), ("output_norm.weight", 0, [256]),
    ("blk.0.attn_norm.weight", 0, [256]), ("blk.0.post_attention_norm.weight", 0, [256]),
    ("blk.0.attn_qkv.weight", 3, [256, 512]), ("blk.0.attn_gate.weight", 2, [256, 256]),
    ("blk.0.ssm_out.weight", 8, [256, 256]), ("blk.0.ssm_alpha.weight", 3, [256, 4]),
    ("blk.0.ssm_beta.weight", 1, [256, 4]), ("blk.0.ssm_a", 0, [4]), ("blk.0.ssm_dt.bias", 0, [4]),
    ("blk.0.ssm_conv1d.weight", 0, [4, 512]), ("blk.0.ssm_norm.weight", 0, [64]),
    ("blk.0.ffn_up.weight", 14, [256, 256]), ("blk.0.ffn_gate.weight", 12, [256, 256]),
    ("blk.0.ffn_down.weight", 3, [256, 256]),
    ("blk.3.attn_norm.weight", 0, [256]), ("blk.3.post_attention_norm.weight", 0, [256]),
    ("blk.3.attn_q.weight", 3, [256, 512]), ("blk.3.attn_k.weight", 8, [256, 128]),
    ("blk.3.attn_v.weight", 2, [256, 128]), ("blk.3.attn_output.weight", 3, [256, 256]),
    ("blk.3.attn_q_norm.weight", 0, [64]), ("blk.3.attn_k_norm.weight", 0, [64]),
    ("blk.3.ffn_up.weight", 3, [256, 256]), ("blk.3.ffn_gate.weight", 3, [256, 256]),
    ("blk.3.ffn_down.weight", 3, [256, 256]),
]

VIEW_NAMES = [
    "lm_head.weight", "model.norm.weight",
    "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.linear_attn.qkv_proj.weight", "model.layers.0.self_attn.gate_proj.weight",
    "model.layers.0.linear_attn.ssm_out_proj.weight", "model.layers.0.linear_attn.ssm_alpha_proj.bf16.weight",
    "model.layers.0.linear_attn.ssm_beta_proj.bf16.weight", "model.layers.0.linear_attn.ssm_a",
    "model.layers.0.linear_attn.ssm_dt.bias", "model.layers.0.linear_attn.ssm_conv1d.weight",
    "model.layers.0.linear_attn.ssm_norm.weight", "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight",
    "model.layers.3.self_attn.q_proj.weight", "model.layers.3.self_attn.k_proj.weight",
    "model.layers.3.self_attn.v_proj.weight", "model.layers.3.self_attn.o_proj.weight",
    "model.layers.3.self_attn.q_norm.weight", "model.layers.3.self_attn.k_norm.weight",
]

M64 = (1 << 64) - 1


def fnv1a(h: int, data: bytes) -> int:
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & M64
    return h


class Lcg:
    def __init__(self, seed):
        self.x = seed

    def next(self):
        self.x = (self.x * 6364136223846793005 + 1442695040888963407) & M64
        return self.x >> 33


def syn_bytes(name: str, ty: int, dims) -> bytes:
    """gguf_pack_test.cpp `syn_tensor`: random bytes, then every fp16 field patched finite."""
    nv = int(np.prod(dims))
    g = Lcg(fnv1a(0xCBF29CE484222325, name.encode()))
    if ty == 0:
        return np.array([np.float32(int(g.next() & 0xFFFF)) * np.float32(1 / 32768) - np.float32(1)
                         for _ in range(nv)], np.float32).tobytes()
    if ty == 1:
        out = []
        for _ in range(nv):
            sign = (g.next() & 1) << 15
            out.append(sign | 0x3000 | (g.next() & 0x3FF))
        return np.array(out, np.uint16).tobytes()
    bpb = {2: 18, 3: 20, 8: 34, 12: 144, 14: 210}[ty]
    vpb = 256 if ty in (12, 14) else 32
    nb = nv // vpb
    b = bytearray(g.next() & 0xFF for _ in range(nb * bpb))
    for i in range(nb):
        o = i * bpb
        if ty in (2, 3, 8):
            struct.pack_into("<H", b, o, 0x2000 | (g.next() & 0x3FF))
        if ty == 3:
            struct.pack_into("<H", b, o + 2, 0xA000 | (g.next() & 0x3FF))
        if ty == 12:
            struct.pack_into("<H", b, o, 0x1800 | (g.next() & 0x3FF))
            struct.pack_into("<H", b, o + 2, 0x1800 | (g.next() & 0x3FF))
        if ty == 14:
            struct.pack_into("<H", b, o + 208, 0x1800 | (g.next() & 0x3FF))
    return bytes(b)


def syn_gguf() -> bytes:
    """gguf_pack_test.cpp `write_syn_qwen35`, in memory."""
    def s(x: str) -> bytes:
        e = x.encode()
        return struct.pack("<Q", len(e)) + e
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(TENSORS), len(GEOMETRY) + 1))
    out += s("general.architecture") + struct.pack("<I", 8) + s("qwen35")
    for k, v in GEOMETRY.items():
        out += s(k) + struct.pack("<II", 4, v)
    data, rel = [], 0
    for name, ty, dims in TENSORS:
        payload = syn_bytes(name, ty, dims)
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", ty, rel)
        data.append((rel, payload))
        rel = (rel + len(payload) + 31) // 32 * 32
    out += b"\0" * ((32 - len(out) % 32) % 32)
    base = len(out)
    for rel, payload in data:
        out += b"\0" * (base + rel - len(out))
        out += payload
    return bytes(out)


@pytest.fixture(scope="module")
def view():
    return GgufAsQ4nx(GgufReader(syn_gguf()))


def view_hash(v) -> int:
    h_all = 0xCBF29CE484222325
    for n in VIEW_NAMES:
        dtype, shape, data = v._get(n)
        h = fnv1a(0xCBF29CE484222325, dtype.encode())
        for d in shape:
            h = fnv1a(h, struct.pack("<Q", d))
        h = fnv1a(h, np.asarray(data, np.uint8).tobytes())
        h_all = fnv1a(h_all, struct.pack("<Q", h))
    return h_all


def test_view_bytes_match_the_cpp_view(view):
    assert view_hash(view) == VIEW_FNV1A


def tiled_to_grouped(t, num_k, num_v, unit):
    """llama.cpp's rule, written as a reshape: '(q g p) ... -> (g q p) ...' with q = num_v / num_k."""
    grp = num_v // num_k
    t = np.asarray(t)
    return t.reshape(grp, num_k, unit, *t.shape[1:]).swapaxes(0, 1).reshape(t.shape)


def bf16_scaled_rows(g: GgufReader, name: str) -> np.ndarray:
    """A Q4_0 / Q4_1 tensor's values with d / m narrowed to bf16: what a 5120-byte chunk holds."""
    t = g.type_of(name)
    rows, cols = g.shape(name)
    raw = g.raw(name).reshape(rows, cols // 32, 20 if t == "Q4_1" else 18)
    d = bf16_to_f32(pack._bf16_rne(np.ascontiguousarray(raw[..., 0:2]).view(np.float16)[..., 0].astype(np.float32)))
    if t == "Q4_1":
        m = bf16_to_f32(pack._bf16_rne(np.ascontiguousarray(raw[..., 2:4]).view(np.float16)[..., 0].astype(np.float32)))
        qs = raw[..., 4:20]
    else:
        m = -8 * d
        qs = raw[..., 2:18]
    q = np.concatenate([qs & 0xF, qs >> 4], -1).astype(np.float32)
    return (d[..., None] * q + m[..., None]).reshape(rows, cols)


def test_value_heads_come_back_grouped(view):
    g = view.g
    qkv = view.matmul_w("model.layers.0.linear_attn.qkv_proj.weight", 512, 256)
    src = bf16_scaled_rows(g, "blk.0.attn_qkv.weight")
    np.testing.assert_array_equal(qkv[:256], src[:256])                       # q | k keep their order
    np.testing.assert_array_equal(qkv[256:], tiled_to_grouped(src[256:], 2, 4, 64))
    z = view.matmul_w("model.layers.0.self_attn.gate_proj.weight", 256, 256)
    np.testing.assert_array_equal(z, tiled_to_grouped(bf16_scaled_rows(g, "blk.0.attn_gate.weight"), 2, 4, 64))
    a = view.f32("model.layers.0.linear_attn.ssm_a")
    np.testing.assert_array_equal(a, tiled_to_grouped(g.rows_f32("blk.0.ssm_a")[0], 2, 4, 1))
    alpha = view.bf16("model.layers.0.linear_attn.ssm_alpha_proj.bf16.weight")
    want = bf16_to_f32(pack._bf16_rne(tiled_to_grouped(g.rows_f32("blk.0.ssm_alpha.weight"), 2, 4, 1)))
    np.testing.assert_array_equal(alpha, want)


def test_out_projection_columns_come_back_grouped(view):
    g = view.g
    view.native_q8 = lambda n: True
    w = view.matmul_w("model.layers.0.linear_attn.ssm_out_proj.weight", 256, 256)   # Q8_0: read at q8
    view.native_q8 = lambda n: False
    src = g.rows_f32("blk.0.ssm_out.weight")
    raw = g.raw("blk.0.ssm_out.weight").reshape(256, 8, 34)
    d = bf16_to_f32(pack._bf16_rne(np.ascontiguousarray(raw[..., 0:2]).view(np.float16)[..., 0].astype(np.float32)))
    q = np.ascontiguousarray(raw[..., 2:34]).view(np.int8).astype(np.float32)
    exact = (d[..., None] * q).reshape(256, 256)
    np.testing.assert_array_equal(w, tiled_to_grouped(exact.T, 2, 4, 64).T)
    assert np.abs(exact - src).max() <= np.abs(src).max() * 2 ** -8                # only the scale narrowed


def test_q_proj_splits_into_q_then_gate(view):
    src = bf16_scaled_rows(view.g, "blk.3.attn_q.weight").reshape(4, 2, 64, 256)    # [head, q|gate, dim, k]
    w = view.matmul_w("model.layers.3.self_attn.q_proj.weight", 512, 256)
    np.testing.assert_array_equal(w[:256], src[:, 0].reshape(256, 256))
    np.testing.assert_array_equal(w[256:], src[:, 1].reshape(256, 256))


def test_conv1d_back_to_taps_by_channels(view):
    g = view.g
    conv = view.bf16("model.layers.0.linear_attn.ssm_conv1d.weight")
    src = g.rows_f32("blk.0.ssm_conv1d.weight")                                       # [512 channels, 4 taps]
    want = np.concatenate([src[:256], tiled_to_grouped(src[256:], 2, 4, 64)]).T
    np.testing.assert_array_equal(conv, bf16_to_f32(pack._bf16_rne(want)))


def test_requantized_tensor_within_half_a_step(view):
    """A Q6_K / Q4_K tensor is re-quantized by requant_q4_1's law: every value within d/2."""
    for name, src in (("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
                      ("model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight")):
        exact = view.g.rows_f32(src)
        got = view.matmul_w(name, 256, 256)
        chunks = np.frombuffer(view.raw(name), np.uint8).reshape(-1, 5120)
        d = bf16_to_f32(np.ascontiguousarray(chunks[:, :512]).view(np.uint16)).max()
        assert np.abs(got - exact).max() <= d / 2 * (1 + 1e-6)


def test_head_q8_law(view):
    """The q8 head from a Q6_K source: rows padded to the spec's 256-row vocab, every value within s/2."""
    exact = view.g.rows_f32("output.weight")                                            # [100, 256]
    chunks = np.frombuffer(view.raw("lm_head.weight"), np.uint8).reshape(-1, 8704)
    assert chunks.shape[0] == 256 // 32
    s_u, c = requant_q8_blocks(exact.reshape(100, 8, 32))
    s = bf16_to_f32(s_u)
    err = np.abs(c.astype(np.float32) * s[..., None] - exact.reshape(100, 8, 32))
    assert (err <= s[..., None] / 2 * (1 + 1e-6)).all()
    w = view.dq_tile(view.raw("lm_head.weight"), 256, 256, 8704, requant=False)
    np.testing.assert_array_equal(w[:100], (c.astype(np.float32) * s[..., None]).reshape(100, 256))
    assert not w[100:].any()


def test_file_head_is_the_tiling_rule():
    for num_k, num_v in ((16, 16), (16, 32), (16, 48), (2, 6)):
        heads = np.arange(num_v)
        tiled = tiled_to_grouped(heads, num_k, num_v, 1)          # grouped position -> the file head it holds
        np.testing.assert_array_equal(file_head(heads, num_k, num_v), tiled)
