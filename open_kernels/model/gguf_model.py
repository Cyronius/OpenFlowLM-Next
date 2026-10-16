"""A GGUF weight file read the way src/open_qwen36 packs it, for the fp64 replicas (2026-10-16).

replica_dense.dense_decode asks its weight object for `bf16(name)` and `matmul_w(name, rows,
cols)` with HF-style names; q4nx.Q4NX answers from a container, this answers from a GGUF:

  * names map as GgufFile::resolve does (llama.cpp's blk.N.* names; a GGUF that carries
    post_attention_norm is a sandwich-norm model, where HF's post_attention_layernorm is
    post_attention_norm and pre_feedforward_layernorm is ffn_norm);
  * small weights round to bf16 (RNE), as pools.cpp pack_norm puts them in `consts`;
  * Q4_0 / Q4_1 matrices dequantize with their fp16 scales narrowed to bf16 (RNE), which is
    what the f32-scale kernels compute with (gemv_q4.h load_scale);
  * a float matrix (Gemma 3's tied F16 embedding as the head) is re-quantized by pools.cpp's
    std_perm_gguf law first: per 32-block, d = fp16((max - min) / 15), m = fp16(min), codes
    lround((v - m) / d) clipped to [0, 15], fp16 below 2^-14 flushed to zero;
  * the embedding row is the stored row times the manifest's scale (Gemma: sqrt(hidden)).
"""
from __future__ import annotations

import numpy as np

from gguf_pool import read_gguf

_LAYER_MAP = (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
              ("self_attn.o_proj", "attn_output"), ("self_attn.q_norm", "attn_q_norm"),
              ("self_attn.k_norm", "attn_k_norm"), ("input_layernorm", "attn_norm"),
              ("post_attention_layernorm", "ffn_norm"), ("pre_feedforward_layernorm", "ffn_norm"),
              ("post_feedforward_layernorm", "post_ffw_norm"), ("mlp.gate_proj", "ffn_gate"),
              ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down"))


def bf16_rne(x) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16).astype(np.uint32).view(np.float32)


def fp16_flush(x) -> np.ndarray:
    """weight_file.hpp f32_to_fp16: round to nearest even, |x| < 2^-14 to (signed) zero."""
    x = np.asarray(x, np.float32)
    h = x.astype(np.float16).astype(np.float32)
    return np.where(np.abs(x) < 2.0 ** -14, np.float32(0.0) * np.sign(x), h).astype(np.float32)


class GGUFModel:
    def __init__(self, path: str, embed_scale: float = 1.0):
        self.path = path
        self.md, self.t, self.base = read_gguf(path)
        self.mm = np.memmap(path, np.uint8, "r")
        self.embed_scale = embed_scale
        self.native_q8 = lambda n: False
        self.hidden = None
        self._w: dict[str, np.ndarray] = {}      # dequantized matrices, kept: a decode reads each per token

    def name(self, hf: str) -> str:
        if hf == "model.embed_tokens.weight":
            return "token_embd.weight"
        if hf == "model.norm.weight":
            return "output_norm.weight"
        if hf == "lm_head.weight":
            return "output.weight" if "output.weight" in self.t else "token_embd.weight"
        if not hf.startswith("model.layers."):
            return hf
        layer, rest = hf[len("model.layers."):].split(".", 1)
        for a, b in _LAYER_MAP:
            if rest.startswith(a + "."):
                g = f"blk.{layer}.{b}.{rest[len(a) + 1:]}"
                if a == "post_attention_layernorm" and f"blk.{layer}.post_attention_norm.weight" in self.t:
                    g = f"blk.{layer}.post_attention_norm.{rest[len(a) + 1:]}"
                return g
        return f"blk.{layer}.{rest}"

    def _bytes(self, g: str, nbytes: int) -> np.ndarray:
        off = self.base + self.t[g]["offset"]
        return np.asarray(self.mm[off:off + nbytes])

    def f32(self, hf: str) -> np.ndarray:
        g = self.name(hf)
        info = self.t[g]
        n = int(np.prod(info["dims"]))
        if info["type"] == "F32":
            return self._bytes(g, 4 * n).view(np.float32).copy()
        if info["type"] == "F16":
            return self._bytes(g, 2 * n).view(np.float16).astype(np.float32)
        raise ValueError(f"{g} is {info['type']}, not a float tensor")

    def bf16(self, hf: str) -> np.ndarray:
        return bf16_rne(self.f32(hf))

    def _float_rows(self, g: str, r0: int, r1: int) -> np.ndarray:
        info = self.t[g]
        cols = info["dims"][0]
        es = 4 if info["type"] == "F32" else 2
        off = self.base + info["offset"] + r0 * cols * es
        raw = np.asarray(self.mm[off:off + (r1 - r0) * cols * es])
        v = raw.view(np.float32) if es == 4 else raw.view(np.float16).astype(np.float32)
        return v.reshape(r1 - r0, cols)

    def _scales_codes(self, g: str, r0: int, r1: int, narrow: bool = True):
        """(d, m, codes) of rows [r0, r1) as the pool holds them: bf16-narrowed f32 scales
        (`narrow=False`: the exact fp16 ones, as the host's embedding row reads them)."""
        nar = bf16_rne if narrow else (lambda a: a)
        info = self.t[g]
        cols = info["dims"][0]
        nb = cols // 32
        if info["type"] in ("Q4_0", "Q4_1"):
            blk = 18 if info["type"] == "Q4_0" else 20
            off = self.base + info["offset"] + r0 * nb * blk
            raw = np.asarray(self.mm[off:off + (r1 - r0) * nb * blk]).reshape(r1 - r0, nb, blk)
            d = raw[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
            if blk == 18:
                m, qs = np.float32(-8.0) * d, raw[..., 2:18]
            else:
                m, qs = raw[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float32), raw[..., 4:20]
            q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).astype(np.float32)
            return nar(d), nar(m), q
        if info["type"] in ("F16", "F32"):
            v = self._float_rows(g, r0, r1).reshape(r1 - r0, nb, 32)
            lo, hi = v.min(-1), v.max(-1)
            df = fp16_flush((hi - lo) / np.float32(15.0))
            mf = fp16_flush(lo)
            inv = np.where(df > 0, np.float32(1.0) / np.where(df > 0, df, 1), np.float32(0.0)).astype(np.float32)
            t = ((v - mf[..., None]) * inv[..., None]).astype(np.float32)
            q = np.clip(np.floor(t.astype(np.float64) + 0.5), 0, 15).astype(np.float32)   # std::lround, exactly
            return nar(df), nar(mf), q
        raise ValueError(f"{g} is {info['type']}; the pack ops take Q4_0 / Q4_1 / F16 / F32 here")

    def matmul_w(self, hf: str, rows: int, cols: int) -> np.ndarray:
        g = self.name(hf)
        dims = self.t[g]["dims"]
        if (dims[1], dims[0]) != (rows, cols):
            raise ValueError(f"{g} is [{dims[1]}, {dims[0]}], the replica asked for [{rows}, {cols}]")
        if g not in self._w:
            d, m, q = self._scales_codes(g, 0, rows)
            self._w[g] = (q * d[..., None] + m[..., None]).reshape(rows, cols).astype(np.float32)
        return self._w[g]

    def head_logits(self, hn, block: int = 16384) -> np.ndarray:
        """logits = W_head @ hn in fp64, from the head the engine packs."""
        g = self.name("lm_head.weight")
        rows, cols = self.t[g]["dims"][1], self.t[g]["dims"][0]
        if "head" not in self._w:
            parts = []
            for r0 in range(0, rows, block):
                d, m, q = self._scales_codes(g, r0, min(rows, r0 + block))
                parts.append((q * d[..., None] + m[..., None]).reshape(-1, cols).astype(np.float32))
            self._w["head"] = np.concatenate(parts)
        x = np.asarray(hn, np.float64)[:cols]
        w = self._w["head"]
        return np.concatenate([w[r0:r0 + block].astype(np.float64) @ x for r0 in range(0, rows, block)])

    def embed(self, tok: int, hidden: int) -> np.ndarray:
        g = self.name("model.embed_tokens.weight")
        cols = self.t[g]["dims"][0]
        if cols != hidden:
            raise ValueError(f"the replica runs at the model's own width {cols}, not {hidden}")
        if self.t[g]["type"] in ("F16", "F32"):
            row = self._float_rows(g, tok, tok + 1)[0]
        else:
            d, m, q = self._scales_codes(g, tok, tok + 1, narrow=False)
            row = (q * d[..., None] + m[..., None]).reshape(-1)
        return (row.astype(np.float64) * self.embed_scale).astype(np.float32)
