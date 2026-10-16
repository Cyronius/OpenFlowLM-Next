"""A GGUF weight file read the way the open engine reads it, for make_decode.py and the
fp64 replica: the counterpart of q4nx.py for a model installed as `model.gguf`.

The engine (src/open_qwen36/gguf_file.cpp + pools.cpp `std_perm_gguf`) packs each
matmul tensor into 6144-byte f32-scale q4 chunks: Q4_0 / Q4_1 blocks exactly, Q8_0
re-quantised per 32-value block to q4_1 (fp16 d and m from the block's min and max,
codes rounded half away from zero). `std_perm_gguf` below reproduces those bytes, and
`matmul_w` reads the same values back with the scales narrowed to bf16, which is what
the kernel multiplies by -- so a slice comparison measures the kernels, not the packing.

Names are the manifest's HF-style ones; the map is GgufFile::gguf_name's. A tied model
has no `output.weight`, and its head is `token_embd.weight`, as in the engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gguf_pool import read_gguf  # noqa: E402

F32, F16, Q4_0, Q4_1, Q8_0 = 0, 1, 2, 3, 8
BLOCK_BYTES = {Q4_0: 18, Q4_1: 20, Q8_0: 34}
CH = 6144


def gguf_name(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if not name.startswith("model.layers."):
        return name
    n = "blk." + name[len("model.layers."):]
    for a, b in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                 ("self_attn.o_proj", "attn_output"), ("self_attn.q_norm", "attn_q_norm"),
                 ("self_attn.k_norm", "attn_k_norm"), ("input_layernorm", "attn_norm"),
                 ("post_attention_layernorm", "ffn_norm"), ("mlp.gate_proj", "ffn_gate"),
                 ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down")):
        n = n.replace("." + a, "." + b, 1)
    return n


def f32_to_bf16(x) -> np.ndarray:
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_round(x) -> np.ndarray:
    return (f32_to_bf16(x).astype(np.uint32) << 16).view(np.float32)


def f32_to_fp16(x) -> np.ndarray:
    """weight_file.hpp f32_to_fp16: round to nearest even, flushing what would be subnormal to zero."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    sign = (u >> 16) & 0x8000
    exp = ((u >> 23) & 0xFF) - 127 + 15
    man = (u & 0x7FFFFF) + 0x0FFF + (((u & 0x7FFFFF) >> 13) & 1)
    carry = (man & 0x800000) != 0
    exp = np.where(carry, exp + 1, exp)
    man = np.where(carry, 0, man)
    h = sign | (exp << 10) | ((man >> 13) & 0x3FF)
    h = np.where(exp >= 31, sign | 0x7C00, h)
    h = np.where(((u >> 23) & 0xFF) - 127 + 15 <= 0, sign, h)
    return h.astype(np.uint16)


def requant_q4_1(v: np.ndarray):
    """pools.cpp's Q8_0 / K-quant law on f32 blocks [..., 32] -> (fp16 d, fp16 m, codes uint8)."""
    v = np.asarray(v, np.float32)
    lo, hi = v.min(-1), v.max(-1)
    ud = f32_to_fp16((hi - lo) / np.float32(15))
    um = f32_to_fp16(lo)
    df, mf = ud.view(np.float16).astype(np.float32), um.view(np.float16).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(df > 0, np.float32(1) / df, np.float32(0)).astype(np.float32)
    t = ((v - mf[..., None]) * inv[..., None]).astype(np.float32)
    q = np.clip(np.floor(t.astype(np.float64) + 0.5), 0, 15).astype(np.uint8)   # lround, then the clamp
    return ud, um, q


class GgufSource:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta, self.tensors, self.base = read_gguf(str(self.path))
        self.mm = np.memmap(self.path, np.uint8, mode="r")
        self.hidden = None
        self.native_q8 = lambda n: False

    # ---- tensor lookup
    def _info(self, name: str) -> tuple[str, dict]:
        g = gguf_name(name)
        if g not in self.tensors and g == "output.weight":
            g = "token_embd.weight"            # tied head
        if g not in self.tensors:
            raise KeyError(f"no tensor {name!r} ({g}) in {self.path.name}")
        return g, self.tensors[g]

    def _bytes(self, t: dict, nbytes: int) -> np.ndarray:
        o = self.base + t["offset"]
        return np.asarray(self.mm[o:o + nbytes])

    def _f32(self, name: str) -> np.ndarray:
        _, t = self._info(name)
        n = int(np.prod(t["dims"]))
        if t["gguf_type"] == F32:
            return self._bytes(t, 4 * n).view(np.float32).copy()
        if t["gguf_type"] == F16:
            return self._bytes(t, 2 * n).view(np.float16).astype(np.float32)
        raise ValueError(f"{name} is GGUF type {t['gguf_type']}, not a small f32 / f16 tensor")

    def blocks(self, name: str):
        """(type, cols, rows, uint8 [rows, cols/32, block bytes]) of a block-quantised matrix."""
        _, t = self._info(name)
        ty = t["gguf_type"]
        if ty not in BLOCK_BYTES:
            raise ValueError(f"{name}: GGUF type {ty} is not Q4_0 / Q4_1 / Q8_0 (this reader's set)")
        cols, rows = t["dims"][0], (t["dims"][1] if len(t["dims"]) > 1 else 1)
        b = BLOCK_BYTES[ty]
        raw = self._bytes(t, rows * cols // 32 * b).reshape(rows, cols // 32, b)
        return ty, cols, rows, raw

    def q4_1_blocks(self, name: str, r0: int = 0, r1: int | None = None):
        """Rows [r0, r1) as the pool holds them: (d f32 [n, nb], m f32 [n, nb], codes uint8 [n, cols])."""
        ty, cols, rows, raw = self.blocks(name)
        raw = raw[r0:rows if r1 is None else r1]
        n, nb = raw.shape[0], raw.shape[1]
        if ty == Q8_0:
            d = raw[..., 0:2].copy().view(np.float16).reshape(n, nb).astype(np.float32)
            v = d[..., None] * raw[..., 2:].copy().view(np.int8).astype(np.float32)
            ud, um, q = requant_q4_1(v)
            return (ud.view(np.float16).astype(np.float32), um.view(np.float16).astype(np.float32),
                    q.reshape(n, cols))
        d = raw[..., 0:2].copy().view(np.float16).reshape(n, nb).astype(np.float32)
        if ty == Q4_1:
            m, qs = raw[..., 2:4].copy().view(np.float16).reshape(n, nb).astype(np.float32), raw[..., 4:20]
        else:
            m, qs = np.float32(-8) * d, raw[..., 2:18]
        q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(n, cols)
        return d, m, q

    # ---- what the packer and the replica ask a container for
    def raw(self, name: str) -> np.ndarray:
        """A small weight as the consts blob holds it: bf16, rounded to nearest even (pools.cpp pack_norm)."""
        return f32_to_bf16(self._f32(name)).view(np.uint8)

    def bf16(self, name: str) -> np.ndarray:
        return bf16_round(self._f32(name))

    def embed(self, tok: int, hid: int) -> np.ndarray:
        ty, cols, rows, raw = self.blocks("model.embed_tokens.weight")
        if cols != hid or not 0 <= tok < rows:
            raise ValueError(f"token {tok} / width {hid} outside the embedding [{rows}, {cols}]")
        if ty != Q8_0:
            d, m, q = self.q4_1_blocks("model.embed_tokens.weight", tok, tok + 1)
            return (q.reshape(-1, 32) * d.reshape(-1, 1) + m.reshape(-1, 1)).reshape(-1).astype(np.float32)
        b = raw[tok]
        d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
        return (d * b[:, 2:].copy().view(np.int8).astype(np.float32)).reshape(-1)

    def std_perm_gguf(self, name: str, nch: int, in_dim: int) -> np.ndarray:
        """pools.cpp `std_perm_gguf`: the pool chunks in band order, rows past the file's zero."""
        from recipes.pack import std_perm
        _, cols, rows, _ = self.blocks(name)
        out_rows = nch // (in_dim // 256) * 32
        if cols != in_dim or rows > out_rows or out_rows - rows >= 64:
            raise ValueError(f"{name}: [{rows}, {cols}] does not fill a {out_rows} x {in_dim} plan")
        d, m, q = self.q4_1_blocks(name)
        pad = out_rows - rows
        d, m = np.pad(d, ((0, pad), (0, 0))), np.pad(m, ((0, pad), (0, 0)))
        q = np.pad(q, ((0, pad), (0, 0)))
        R, ncol = out_rows // 32, in_dim // 256
        dp = d.reshape(R, 32, ncol, 8).transpose(0, 2, 3, 1).reshape(R * ncol, 256)   # j = kb * 32 + r
        mp = m.reshape(R, 32, ncol, 8).transpose(0, 2, 3, 1).reshape(R * ncol, 256)
        qq = q.reshape(R, 2, 16, ncol, 256).transpose(0, 3, 1, 4, 2).reshape(R * ncol, 8192)
        codes = (qq[:, 0::2] | (qq[:, 1::2] << 4)).astype(np.uint8)
        chunks = np.concatenate([dp.astype(np.float32).view(np.uint8), mp.astype(np.float32).view(np.uint8),
                                 codes], axis=1)
        return chunks[std_perm(nch, in_dim)].reshape(-1)

    def matmul_w(self, name: str, rows: int, cols: int) -> np.ndarray:
        """The weights the NPU multiplies by: pool codes with the scales narrowed to bf16 (gemv_q4.h load_scale)."""
        d, m, q = self.q4_1_blocks(name)
        if q.shape != (rows, cols):
            raise ValueError(f"{name}: {q.shape} is not [{rows}, {cols}]")
        dd, mm = bf16_round(d), bf16_round(m)
        return (q.reshape(rows, -1, 32) * dd[..., None] + mm[..., None]).reshape(rows, cols).astype(np.float32)

    def lm_logits(self, hn, block: int = 8192) -> np.ndarray:
        """logits over the padded head rows (rows past the vocabulary are zero, as in the pool)."""
        _, cols, rows, _ = self.blocks("lm_head.weight")
        hn = np.asarray(hn, np.float64)
        out = np.zeros(rows + (-rows) % 64, np.float64)
        for r0 in range(0, rows, block):
            r1 = min(rows, r0 + block)
            d, m, q = self.q4_1_blocks("lm_head.weight", r0, r1)
            w = (q.reshape(r1 - r0, -1, 32) * bf16_round(d)[..., None] + bf16_round(m)[..., None])
            out[r0:r1] = w.reshape(r1 - r0, cols).astype(np.float64) @ hn
        return out
