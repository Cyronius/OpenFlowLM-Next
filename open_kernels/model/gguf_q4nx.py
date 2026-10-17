"""A GGUF read as the `.q4nx` container a qwen35 kernel set's plan was written for.

The NumPy twin of src/open_qwen36/gguf_view.cpp (OPEN-GGUF-Q4NX-VIEW): the same names,
the same undone llama.cpp conversions, the same q4nx-form bytes. `GgufAsQ4nx` has the
interface recipes/pack.py and the fp64 replicas read a `Q4NX` through, so the Python
packer, the slice harness and replica_qwen35 run on a GGUF unchanged.

    python open_kernels/model/gguf_q4nx.py <model.gguf>     # lists what the view serves
"""
from __future__ import annotations

import mmap
import re
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from q4nx import CHUNK_Q4, CHUNK_Q8, Q4NX, bf16_to_f32  # noqa: E402

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
         10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
# bytes per block and values per block, for the types this reader dequantizes
BLOCK = {"F32": (4, 1), "F16": (2, 1), "BF16": (2, 1), "Q4_0": (18, 32), "Q4_1": (20, 32),
         "Q8_0": (34, 32), "Q4_K": (144, 256), "Q6_K": (210, 256)}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def bf16_rne(x) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_ceil(x) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return np.where(u >> 31, u >> 16, (u + 0xFFFF) >> 16).astype(np.uint16)


class GgufReader:
    """Header, tensor index and exact f32 rows of a GGUF (v2 / v3), from a path or bytes."""

    def __init__(self, src):
        if isinstance(src, (bytes, bytearray, memoryview)):
            self.buf = memoryview(src)
        else:
            f = open(src, "rb")
            self.buf = memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        b = self.buf
        if bytes(b[:4]) != b"GGUF":
            raise ValueError("not a GGUF file")
        ver, nt, nkv = struct.unpack_from("<IQQ", b, 4)
        if ver < 2:
            raise ValueError(f"GGUF v{ver} unsupported (need v2+)")
        p = 24

        def rd_str():
            nonlocal p
            (n,) = struct.unpack_from("<Q", b, p)
            s = bytes(b[p + 8:p + 8 + n]).decode("utf-8", "replace")
            p += 8 + n
            return s

        def rd_val(t):
            nonlocal p
            if t == 8:
                return rd_str()
            if t == 9:
                et, n = struct.unpack_from("<IQ", b, p)
                p += 12
                return [rd_val(et) for _ in range(n)]
            fmt = _SCALAR[t]
            (v,) = struct.unpack_from(fmt, b, p)
            p += struct.calcsize(fmt)
            return v

        self.kv = {}
        for _ in range(nkv):
            k = rd_str()
            (t,) = struct.unpack_from("<I", b, p)
            p += 4
            self.kv[k] = rd_val(t)
        self.tensors = {}
        for _ in range(nt):
            name = rd_str()
            (nd,) = struct.unpack_from("<I", b, p)
            dims = list(struct.unpack_from(f"<{nd}Q", b, p + 4))
            t, off = struct.unpack_from("<IQ", b, p + 4 + 8 * nd)
            p += 4 + 8 * nd + 12
            self.tensors[name] = (TYPES.get(t, f"type{t}"), dims, off)
        align = int(self.kv.get("general.alignment", 32))
        self.base = p + (align - p % align) % align

    def has(self, name):
        return name in self.tensors

    def type_of(self, name):
        return self.tensors[name][0]

    def shape(self, name):
        """Row-major [rows, cols] (GGUF stores dims fastest-first)."""
        dims = self.tensors[name][1]
        return (dims[1] if len(dims) >= 2 else 1), dims[0]

    def raw(self, name) -> np.ndarray:
        t, dims, off = self.tensors[name]
        bpb, vpb = BLOCK[t]
        n = int(np.prod(dims)) // vpb * bpb
        return np.frombuffer(self.buf, np.uint8, n, self.base + off)

    def rows_f32(self, name, r0=0, r1=None) -> np.ndarray:
        """Rows [r0, r1) as f32, by the same arithmetic as GgufFile::embed_row."""
        t = self.type_of(name)
        rows, cols = self.shape(name)
        r1 = rows if r1 is None else r1
        bpb, vpb = BLOCK.get(t, (0, 0))
        if not bpb:
            raise ValueError(f"{name} is {t}; this reader dequantizes {sorted(BLOCK)}")
        row_bytes = cols // vpb * bpb
        raw = self.raw(name)[r0 * row_bytes:r1 * row_bytes].reshape(r1 - r0, cols // vpb, bpb)
        n = r1 - r0
        if t == "F32":
            return raw.reshape(n, -1).view(np.float32).copy()
        if t == "F16":
            return raw.reshape(n, -1).view(np.float16).astype(np.float32)
        if t == "BF16":
            return bf16_to_f32(raw.reshape(n, -1).view(np.uint16))
        if t in ("Q4_0", "Q4_1"):
            d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16)[..., 0].astype(np.float32)
            if t == "Q4_1":
                m = np.ascontiguousarray(raw[..., 2:4]).view(np.float16)[..., 0].astype(np.float32)
                qs = raw[..., 4:20]
                q = np.concatenate([qs & 0xF, qs >> 4], -1).astype(np.float32)
                return (d[..., None] * q + m[..., None]).reshape(n, cols).astype(np.float32)
            qs = raw[..., 2:18]
            q = (np.concatenate([qs & 0xF, qs >> 4], -1).astype(np.int32) - 8).astype(np.float32)
            return (d[..., None] * q).reshape(n, cols).astype(np.float32)
        if t == "Q8_0":
            d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16)[..., 0].astype(np.float32)
            q = np.ascontiguousarray(raw[..., 2:34]).view(np.int8).astype(np.float32)
            return (d[..., None] * q).reshape(n, cols).astype(np.float32)
        if t == "Q6_K":
            return _dq_q6_k(raw).reshape(n, cols)
        if t == "Q4_K":
            return _dq_q4_k(raw).reshape(n, cols)
        raise ValueError(t)


def _dq_q6_k(raw) -> np.ndarray:
    """[..., 210] super-blocks -> [..., 256] f32, gguf_file.cpp's loop order (d * sc, then * q)."""
    ql, qh = raw[..., 0:128].astype(np.int32), raw[..., 128:192].astype(np.int32)
    sc = np.ascontiguousarray(raw[..., 192:208]).view(np.int8).astype(np.float32)
    d = np.ascontiguousarray(raw[..., 208:210]).view(np.float16)[..., 0].astype(np.float32)
    out = np.empty(raw.shape[:-1] + (256,), np.float32)
    for half in range(2):
        L, H, S = ql[..., 64 * half:64 * half + 64], qh[..., 32 * half:32 * half + 32], sc[..., 8 * half:8 * half + 8]
        l = np.arange(32)
        q1 = ((L[..., l] & 0xF) | (((H[..., l] >> 0) & 3) << 4)) - 32
        q2 = ((L[..., l + 32] & 0xF) | (((H[..., l] >> 2) & 3) << 4)) - 32
        q3 = ((L[..., l] >> 4) | (((H[..., l] >> 4) & 3) << 4)) - 32
        q4 = ((L[..., l + 32] >> 4) | (((H[..., l] >> 6) & 3) << 4)) - 32
        isx = l // 16
        for k, q in enumerate((q1, q2, q3, q4)):
            ds = (d[..., None] * S[..., isx + 2 * k]).astype(np.float32)
            out[..., 128 * half + 32 * k + l] = (ds * q.astype(np.float32)).astype(np.float32)
    return out


def _scale_min_k4(s):
    """llama.cpp's 6-bit scales and mins of a Q4_K super-block: [..., 12] -> ([..., 8], [..., 8])."""
    sc = np.empty(s.shape[:-1] + (8,), np.int32)
    mn = np.empty_like(sc)
    for j in range(8):
        if j < 4:
            sc[..., j], mn[..., j] = s[..., j] & 63, s[..., j + 4] & 63
        else:
            sc[..., j] = (s[..., j + 4] & 0xF) | ((s[..., j - 4] >> 6) << 4)
            mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    return sc, mn


def _dq_q4_k(raw) -> np.ndarray:
    """[..., 144] super-blocks -> [..., 256] f32, as gguf_file.cpp (d1 = d * sc; d1 * q - m1)."""
    d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16)[..., 0].astype(np.float32)
    dmin = np.ascontiguousarray(raw[..., 2:4]).view(np.float16)[..., 0].astype(np.float32)
    qs = raw[..., 16:144].astype(np.int32)
    sc, mn = _scale_min_k4(raw[..., 4:16].astype(np.int32))
    out = np.empty(raw.shape[:-1] + (256,), np.float32)
    for g in range(4):
        q = qs[..., 32 * g:32 * g + 32]
        for h, nib in enumerate((q & 0xF, q >> 4)):
            j = 2 * g + h
            d1 = (d * sc[..., j].astype(np.float32)).astype(np.float32)
            m1 = (dmin * mn[..., j].astype(np.float32)).astype(np.float32)
            y = (d1[..., None] * nib.astype(np.float32)).astype(np.float32) - m1[..., None]
            out[..., 64 * g + 32 * h:64 * g + 32 * h + 32] = y
    return out


def file_head(h, num_k, num_v):
    """Grouped head h is file head (h % grp) * num_k + h // grp (OPEN-CONVERT-QWEN35-VHEADS)."""
    grp = num_v // num_k
    return (h % grp) * num_k + h // grp


def _code_index():
    r = np.arange(32)[:, None]
    k = np.arange(256)[None, :]
    return (r // 16) * 4096 + k * 16 + (r % 16)                        # [32, 256]


_CODE_AT = _code_index()


def requant_q8_blocks(v) -> tuple[np.ndarray, np.ndarray]:
    """gguf_view.cpp requant_q8_block: s = max|v| / 127 toward +inf in bf16, code = floor(v / s + 1/2)."""
    v = np.asarray(v, np.float32)
    amax = np.abs(v).max(-1)
    s_u = bf16_ceil((amax / np.float32(127)).astype(np.float32))
    s = bf16_to_f32(s_u).astype(np.float32)
    inv = np.where(s > 0, np.float32(1) / np.where(s > 0, s, np.float32(1)), np.float32(0)).astype(np.float32)
    t = (v * inv[..., None]).astype(np.float32)
    c = np.clip(np.floor((t + np.float32(0.5)).astype(np.float32)), -127, 127).astype(np.int8)
    return s_u, c


def to_q4nx_chunks(g: GgufReader, name, rows, cols, row_of=None, blk_of=None, head=False) -> tuple[np.ndarray, int]:
    """One GGUF matmul tensor -> q4nx chunks in the file raster, gguf_view.cpp `to_q4nx_chunks`."""
    from recipes.pack import requant_values_q4_1
    t = g.type_of(name)
    src_rows, src_cols = g.shape(name)
    if src_cols != cols:
        raise ValueError(f"{name} is [{src_rows}, {src_cols}], the plan reads {cols} columns")
    if not head and rows != src_rows:
        raise ValueError(f"{name} has {src_rows} rows, the plan reads {rows}")
    if rows % 32 or cols % 256:
        raise ValueError(f"{name}: [{rows}, {cols}] is not whole 32 x 256 chunks")
    nb, ncol, nrb = cols // 32, cols // 256, rows // 32
    rmap = np.arange(rows) if row_of is None else np.asarray(row_of)
    bmap = np.arange(nb) if blk_of is None else np.asarray(blk_of)
    live = rmap < src_rows
    q4 = not head and t in ("Q4_0", "Q4_1")
    q4k = not head and t == "Q4_K"
    q8 = t == "Q8_0"
    ch = CHUNK_Q8 if (head or q8) else CHUNK_Q4
    out = np.zeros((nrb, ncol, ch), np.uint8)
    for r0 in range(0, rows, 4096):
        r1 = min(r0 + 4096, rows)
        rr = rmap[r0:r1]
        ok = live[r0:r1]
        nrw = r1 - r0
        if q4k:
            # a Q4_K sub-block is q4_1's per-32 scale and min already: only their products round
            sb = g.raw(name).reshape(src_rows, cols // 256, 144)[np.where(ok, rr, 0)]
            sb = np.where(ok[:, None, None], sb, 0).astype(np.uint8)
            dd = np.ascontiguousarray(sb[..., 0:2]).view(np.float16)[..., 0].astype(np.float32)
            dm = np.ascontiguousarray(sb[..., 2:4]).view(np.float16)[..., 0].astype(np.float32)
            sc, mn = _scale_min_k4(sb[..., 4:16].astype(np.int32))                   # [n, nsb, 8]
            d1 = (dd[..., None] * sc.astype(np.float32)).astype(np.float32).reshape(nrw, nb)[:, bmap]
            m1 = (dm[..., None] * mn.astype(np.float32)).astype(np.float32).reshape(nrw, nb)[:, bmap]
            qs = sb[..., 16:144].reshape(nrw, cols // 256, 4, 32)
            qk = np.stack([qs & 0xF, qs >> 4], 3).reshape(nrw, nb, 32)[:, bmap]        # sub-block 2g + h
        elif q4 or q8:
            bpb = BLOCK[t][0]
            blocks = g.raw(name).reshape(src_rows, nb, bpb)[np.where(ok, rr, 0)][:, bmap]   # [n, nb, bpb]
            blocks = np.where(ok[:, None, None], blocks, 0).astype(np.uint8)
            d = np.ascontiguousarray(blocks[..., 0:2]).view(np.float16)[..., 0].astype(np.float32)
        else:
            vals = np.zeros((nrw, cols), np.float32)
            idx = np.nonzero(ok)[0]
            if idx.size:
                srcv = np.concatenate([g.rows_f32(name, int(s), int(s) + 1) for s in rr[idx]]) \
                    if row_of is not None else g.rows_f32(name, r0, r0 + idx.size)
                vals[idx] = srcv
            vals = vals.reshape(nrw, nb, 32)[:, bmap]                                   # [n, nb, 32]
        # per row and 256-column tile: d / m planes at kb*32 + r, codes at code_at(r, k)
        for i in range(0, nrw, 32):
            rb = (r0 + i) // 32
            rl = slice(i, i + 32)
            for kt in range(ncol):
                c = out[rb, kt]
                bs = slice(8 * kt, 8 * kt + 8)
                # a meta plane is [8 kb][32 r], so a [32 r, 8 kb] array goes in transposed
                if q8:
                    codes = np.ascontiguousarray(blocks[rl, bs, 2:34]).view(np.int8)  # [32 r, 8 kb, 32]
                    c[:512].view(np.uint16)[:] = bf16_rne(d[rl, bs]).T.reshape(-1)
                    c[512:][_CODE_AT.reshape(-1)] = codes.reshape(32, 256).view(np.uint8).reshape(-1)
                elif head:
                    s, codes = requant_q8_blocks(vals[rl, bs])                     # [32, 8], [32, 8, 32]
                    c[:512].view(np.uint16)[:] = s.T.reshape(-1)
                    c[512:][_CODE_AT.reshape(-1)] = codes.reshape(32, 256).view(np.uint8).reshape(-1)
                elif q4:
                    bl = blocks[rl, bs]
                    if t == "Q4_1":
                        m = np.ascontiguousarray(bl[..., 2:4]).view(np.float16)[..., 0].astype(np.float32)
                        qs = bl[..., 4:20]
                    else:
                        m = (np.float32(-8) * d[rl, bs]).astype(np.float32)
                        qs = bl[..., 2:18]
                    q = np.concatenate([qs & 0xF, qs >> 4], -1).reshape(32, 256)
                    c[:512].view(np.uint16)[:] = bf16_rne(d[rl, bs]).T.reshape(-1)
                    c[512:1024].view(np.uint16)[:] = bf16_rne(m).T.reshape(-1)
                    flat = np.zeros(8192, np.uint8)
                    flat[_CODE_AT.reshape(-1)] = q.reshape(-1)
                    c[1024:] = flat[0::2] | (flat[1::2] << 4)
                elif q4k:
                    c[:512].view(np.uint16)[:] = bf16_rne(d1[rl, bs]).T.reshape(-1)
                    c[512:1024].view(np.uint16)[:] = bf16_rne(-m1[rl, bs]).T.reshape(-1)
                    flat = np.zeros(8192, np.uint8)
                    flat[_CODE_AT.reshape(-1)] = qk[rl, bs].reshape(-1)
                    c[1024:] = flat[0::2] | (flat[1::2] << 4)
                else:
                    v = vals[rl, bs]                                                   # [32 r, 8 kb, 32]
                    c[:] = requant_values_q4_1(v[None])[0]
    return out.reshape(-1, ch), ch


_RULES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.gate_proj.weight": "attn_gate.weight",
    "linear_attn.qkv_proj.weight": "attn_qkv.weight",
    "linear_attn.ssm_out_proj.weight": "ssm_out.weight",
    "linear_attn.ssm_alpha_proj.bf16.weight": "ssm_alpha.weight",
    "linear_attn.ssm_beta_proj.bf16.weight": "ssm_beta.weight",
    "linear_attn.ssm_a": "ssm_a",
    "linear_attn.ssm_dt.bias": "ssm_dt.bias",
    "linear_attn.ssm_conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.ssm_norm.weight": "ssm_norm.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


class GgufAsQ4nx(Q4NX):
    """`Q4NX`'s interface over a qwen35-arch GGUF (gguf_view.cpp GgufQ4nxView)."""

    def __init__(self, src):
        self.g = g = src if isinstance(src, GgufReader) else GgufReader(src)
        self.path = str(src) if not isinstance(src, (GgufReader, bytes, bytearray, memoryview)) else "<gguf>"
        a = g.kv["general.architecture"]
        if a != "qwen35":
            raise ValueError(f"architecture {a} has no q4nx view (have qwen35)")

        def k(n):
            if f"{a}.{n}" not in g.kv:
                raise KeyError(f"GGUF lacks {a}.{n}")
            return int(g.kv[f"{a}.{n}"])

        self.geo = dict(hidden=k("embedding_length"), heads=k("attention.head_count"),
                        kv_heads=k("attention.head_count_kv"), head_dim=k("attention.key_length"),
                        lin_k_heads=k("ssm.group_count"), lin_v_heads=k("ssm.time_step_rank"),
                        lin_k_dim=k("ssm.state_size"), conv=k("ssm.conv_kernel"))
        self.geo["lin_v_dim"] = k("ssm.inner_size") // self.geo["lin_v_heads"]
        self.geo["blocks"] = k("block_count") - int(g.kv.get(f"{a}.nextn_predict_layers", 0))
        self.chunk_bytes = CHUNK_Q4
        self.requant_q8 = True
        self.native_q8 = lambda name: False
        self._cache = {}
        rows, cols = g.shape("token_embd.weight")
        self.header = {}
        self.tensors = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [rows, cols]}}

    def source_name(self, name):
        if name == "model.embed_tokens.weight":
            return "token_embd.weight"
        if name == "model.norm.weight":
            return "output_norm.weight"
        if name == "lm_head.weight":
            return "output.weight" if self.g.has("output.weight") else "token_embd.weight"
        m = _LAYER.match(name)
        if not m or int(m.group(1)) >= self.geo["blocks"] or m.group(2) not in _RULES:
            return None
        return f"blk.{m.group(1)}.{_RULES[m.group(2)]}"

    def has(self, name):
        s = self.source_name(name)
        return s is not None and self.g.has(s)

    def _get(self, name):
        if name not in self._cache:
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[name] = self._build(name)
        return self._cache[name]

    def _build(self, name):
        src = self.source_name(name)
        if src is None or not self.g.has(src):
            raise KeyError(f"no tensor {name!r} (GGUF {src})")
        G, g = self.geo, self.g
        vh, kh, dv = G["lin_v_heads"], G["lin_k_heads"], G["lin_v_dim"]
        qk = 2 * kh * G["lin_k_dim"]
        hid, hd = G["hidden"], G["head_dim"]
        qw, kvw, vw = G["heads"] * hd, G["kv_heads"] * hd, vh * dv

        def value_rows(first, unit):
            h = np.arange(vh)
            src_h = file_head(h, kh, vh)
            v = first + (src_h[:, None] * unit + np.arange(unit)[None, :]).reshape(-1)
            return np.concatenate([np.arange(first), v])

        def chunks(rows, cols, row_of=None, blk_of=None, head=False):
            b, ch = to_q4nx_chunks(g, src, rows, cols, row_of, blk_of, head)
            return ("I8", [b.shape[0], ch], b.reshape(-1))

        def small(v, shape, f32=False):
            v = np.asarray(v, np.float32).reshape(-1)
            data = v.view(np.uint8) if f32 else bf16_rne(v).view(np.uint8)
            return ("F32" if f32 else "BF16", list(shape), data.copy())

        def vec(n_rows, n_cols):
            r, c = g.shape(src)
            if (r, c) != (n_rows, n_cols):
                raise ValueError(f"{src} is [{r}, {c}], the plan reads [{n_rows}, {n_cols}]")
            return g.rows_f32(src)

        if name == "lm_head.weight":
            rows = g.shape(src)[0]
            return chunks(-(-rows // 256) * 256, hid, head=True)     # recipes/spec.py's padded vocab
        if name == "model.norm.weight":
            return small(vec(1, hid), [hid])
        sfx = _LAYER.match(name).group(2)
        layer = int(_LAYER.match(name).group(1))
        ff = g.shape(f"blk.{layer}.ffn_up.weight")[0] if g.has(f"blk.{layer}.ffn_up.weight") else 0
        if sfx == "self_attn.q_proj.weight":
            r = np.arange(2 * qw)
            part, rr = r // qw, r % qw
            return chunks(2 * qw, hid, row_of=(rr // hd) * 2 * hd + part * hd + rr % hd)
        if sfx in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            return chunks(kvw, hid)
        if sfx == "self_attn.o_proj.weight":
            return chunks(hid, qw)
        if sfx == "self_attn.gate_proj.weight":
            return chunks(vw, hid, row_of=value_rows(0, dv))
        if sfx == "linear_attn.qkv_proj.weight":
            return chunks(qk + vw, hid, row_of=value_rows(qk, dv))
        if sfx == "linear_attn.ssm_out_proj.weight":
            if dv % 32:
                raise ValueError(f"{name}: the value head dim is not whole 32-column blocks")
            return chunks(hid, vw, blk_of=value_rows(0, dv // 32))
        if sfx in ("mlp.up_proj.weight", "mlp.gate_proj.weight"):
            return chunks(ff, hid)
        if sfx == "mlp.down_proj.weight":
            return chunks(hid, ff)
        if sfx in ("linear_attn.ssm_alpha_proj.bf16.weight", "linear_attn.ssm_beta_proj.bf16.weight"):
            v = vec(vh, hid)
            return small(v[file_head(np.arange(vh), kh, vh)], [vh, hid])
        if sfx in ("linear_attn.ssm_a", "linear_attn.ssm_dt.bias"):
            v = vec(1, vh).reshape(-1)
            return small(v[file_head(np.arange(vh), kh, vh)], [vh], f32=True)
        if sfx == "linear_attn.ssm_conv1d.weight":
            nch = qk + vw
            v = vec(nch, G["conv"])
            return small(v[value_rows(qk, dv)].T, [G["conv"], nch])
        if sfx == "linear_attn.ssm_norm.weight":
            return small(vec(1, dv), [dv])
        if sfx in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            return small(vec(1, hd), [hd])
        return small(vec(1, hid), [hid])

    # ---- the Q4NX interface recipes/pack.py and the replicas read
    def chunk_bytes_of(self, name):
        dtype, shape, _ = self._get(name)
        return shape[-1] if dtype == "I8" else 0

    def raw(self, name):
        if name == "model.embed_tokens.weight":
            raise KeyError("the embedding is read by row (embed), not whole")
        return self._get(name)[2]

    def bf16(self, name):
        dtype, shape, data = self._get(name)
        if dtype != "BF16":
            raise ValueError(f"{name} is {dtype}, not BF16")
        return bf16_to_f32(data.view(np.uint16).reshape(shape))

    def f32(self, name):
        dtype, shape, data = self._get(name)
        if dtype != "F32":
            raise ValueError(f"{name} is {dtype}, not F32")
        return data.view(np.float32).reshape(shape)

    def embed(self, token, hidden=None):
        rows, cols = self.g.shape("token_embd.weight")
        if not 0 <= token < rows:
            raise IndexError(f"token {token} is out of range for a {rows}-row embedding table")
        if hidden is not None and hidden != cols:
            raise ValueError(f"hidden={hidden} does not match the embedding's row width {cols}")
        return self.g.rows_f32("token_embd.weight", token, token + 1)[0].astype(np.float64)

    def quant_format_of(self, name):
        return None


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf")
    a = ap.parse_args(argv)
    v = GgufAsQ4nx(a.gguf)
    print(f"{a.gguf}: {v.geo}")
    for n in ("lm_head.weight", "model.norm.weight", "model.layers.0.linear_attn.qkv_proj.weight",
              "model.layers.3.self_attn.q_proj.weight"):
        if v.has(n):
            dt, shape, _ = v._get(n)
            print(f"  {n:48s} <- {v.source_name(n):28s} {dt} {shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
