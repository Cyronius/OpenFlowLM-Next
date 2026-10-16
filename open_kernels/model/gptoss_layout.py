"""GPT-OSS's expert layout on the main cores as host arithmetic (2026-10-16; plans/gptoss-expert-layout.md).

    python -I open_kernels/model/gptoss_layout.py                      # geometry and stream check
    python -I open_kernels/model/gptoss_layout.py --gguf <MXFP4 .gguf> [--layer L --token T]  # + a one-layer slice

Traces: OPEN-GPTOSS-EXPERT-LAYOUT (specs/open-engine/spec.md).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from mxfp4 import KVALUES, e8m0  # noqa: E402

TILE_BYTES = 5120
TILE_ROWS = 32
TILE_BLOCKS = 9
TILE_COLS = 32 * TILE_BLOCKS
NIB_PLANE = 16 * TILE_COLS // 2
EXP_OFF = 2 * NIB_PLANE                  # E8M0 bytes, index kb * 32 + r
BIAS_OFF = 4928                          # 64-byte aligned; 32 bf16 biases in k-tile 0 only
assert EXP_OFF + TILE_ROWS * TILE_BLOCKS <= BIAS_OFF and BIAS_OFF + 64 <= TILE_BYTES

# recipes/qwen36moe.py's numbers, restated so this module needs no recipe import
BAND_ROWS = 64
ELEM = 4096                               # one x-stream element
CALL_BYTES = 2 * TILE_BYTES               # one w element: both row halves of one k-tile
Y_BYTES = BAND_ROWS * 4
L1_BUDGET = 60 * 1024
STACK = 0x1800
X_DEPTH = 2
PIPELINE_DEPTH = 3
SHIM_CHANNELS = 16                        # NPU2: eight shim tiles, two DMA channels each way


def tab_bytes(K: int) -> int:
    """gemv_tab.h's table: int16 x | int32 shift per block | bf16 block sums hi, lo."""
    return 2 * K + K // 8 + K // 8


def deal(rows: int, n_cores: int, pad: bool = False) -> tuple[int, ...]:
    """64-row bands per core, as even as possible, the larger shares first."""
    if rows % BAND_ROWS:
        raise ValueError(f"{rows} rows is not a whole number of {BAND_ROWS}-row bands")
    bands = rows // BAND_ROWS
    if pad:
        bands = -(-bands // n_cores) * n_cores
    q, r = divmod(bands, n_cores)
    return tuple(q + (c < r) for c in range(n_cores))


@dataclass(frozen=True)
class Layout:
    hidden: int
    ff: int
    experts: int
    topk: int
    n_cores: int
    pad_rows: bool
    up_bands: tuple
    down_bands: tuple
    up_band_elems: int
    down_band_elems: int
    expert_bytes: int
    down_off: int
    xm_elems: int
    h_elems: int
    hdr_elems: int
    ms_floats: int
    table_k: int
    l1: int

    @property
    def up_row0(self) -> tuple:
        return tuple(BAND_ROWS * sum(self.up_bands[:c]) for c in range(self.n_cores))

    @property
    def down_row0(self) -> tuple:
        return tuple(BAND_ROWS * sum(self.down_bands[:c]) for c in range(self.n_cores))

    @property
    def band_bytes_up(self) -> int:
        return self.up_band_elems * CALL_BYTES

    @property
    def band_bytes_down(self) -> int:
        return self.down_band_elems * CALL_BYTES

    def layer_pool_bytes(self) -> int:
        return self.experts * self.expert_bytes

    def stream_bytes_per_layer(self) -> int:
        return self.topk * self.expert_bytes


def layout(hidden: int = 2880, ff: int = 2880, experts: int = 32, topk: int = 4, n_cores: int = 8,
           pad_rows: bool = False, table_k: int | None = None) -> Layout:
    """`table_k`: the widest K the core prepares (4096 when it also runs o_proj)."""
    for what, k in (("hidden", hidden), ("ff", ff)):
        if k % TILE_COLS:
            raise ValueError(f"{what} {k} is not a whole number of {TILE_COLS}-column MXFP4 tiles")
    if 8 + topk > 32:
        raise ValueError(f"top-k {topk} does not fit the 32-float routing record")
    up, down = deal(ff, n_cores, pad_rows), deal(hidden, n_cores, pad_rows)
    up_elems = hidden // TILE_COLS        # a 64-row band is two tiles per k-tile: one element each
    down_elems = ff // TILE_COLS
    gu = 2 * sum(up) * up_elems * CALL_BYTES
    expert = gu + sum(down) * down_elems * CALL_BYTES
    rmax = BAND_ROWS * max(max(up), max(down))
    # ms: router record 32 | residual rows | accumulator rows | gate 64 | up 64 | down rows
    ms = 32 + 3 * rmax + 2 * BAND_ROWS
    tk = table_k or max(hidden, ff)
    l1 = tab_bytes(tk) + 4 * ms + 2 * CALL_BYTES + 2 * ELEM + 2 * Y_BYTES + STACK
    return Layout(hidden=hidden, ff=ff, experts=experts, topk=topk, n_cores=n_cores, pad_rows=pad_rows,
                  up_bands=up, down_bands=down, up_band_elems=up_elems, down_band_elems=down_elems,
                  expert_bytes=expert, down_off=gu, xm_elems=-(-2 * hidden // ELEM),
                  h_elems=-(-4 * ff // ELEM), hdr_elems=2, ms_floats=ms, table_k=tk, l1=l1)


def core_program(L: Layout, c: int) -> list[tuple]:
    """Core c's fifo operations and kernel calls for one token, in order."""
    ev: list[tuple] = []
    if L.xm_elems > X_DEPTH:
        raise ValueError(f"xm is {L.xm_elems} x elements, more than the fifo's {X_DEPTH}")
    ev += [("acq", "x", L.xm_elems)] + [("call", "prep", L.hidden, i) for i in range(L.xm_elems)]
    ev += [("rel", "x", L.xm_elems)]                 # the GEMVs read only the table
    for m in range(L.hdr_elems):
        ev += [("acq", "w", 1), ("call", "hdr", m), ("rel", "w", 1)]
    for s in range(L.topk):
        for j in range(L.up_bands[c]):
            for dst in ("g", "u"):
                for g in range(L.up_band_elems):
                    ev += [("acq", "w", 1), ("call", "mx", s, dst, j, g), ("rel", "w", 1)]
            ev += [("acq", "y", 1), ("call", "act", s, j), ("rel", "y", 1)]
    for s in range(L.topk):
        for i in range(L.h_elems):
            ev += [("acq", "x", 1), ("call", "prepf", L.ff, i), ("rel", "x", 1)]
        for j in range(L.down_bands[c]):
            for g in range(L.down_band_elems):
                ev += [("acq", "w", 1), ("call", "mx", s, "d", j, g), ("rel", "w", 1)]
        ev += [("call", "accfin", s)]
    for j in range(L.down_bands[c]):
        ev += [("acq", "y", 1), ("call", "out", j), ("rel", "y", 1)]
    return ev


@dataclass(frozen=True)
class Act:
    """Act offsets; the residual region runs a w element past the last core's slice, which its header fill reads."""
    xm: int = 0
    rout: int = 2 * ELEM
    res: int = 2 * ELEM + CALL_BYTES
    h: int = 2 * ELEM + CALL_BYTES + 4 * ELEM + CALL_BYTES
    h_stride: int = 3 * ELEM


def host_sequence(L: Layout, A: Act = Act()) -> list[tuple]:
    """DMA tasks in issue order, (kind, fifo, core, buffer, offset, bytes, slot); slot s reads placeholder expert s."""
    seq: list[tuple] = [("fill", "x", -1, "act", A.xm, L.xm_elems * ELEM, -1)]
    for c in range(L.n_cores):
        seq.append(("fill", "w", c, "act", A.rout, CALL_BYTES, -1))
        seq.append(("fill", "w", c, "act", A.res + 4 * L.down_row0[c], CALL_BYTES, -1))
    for s in range(L.topk):
        for c in range(L.n_cores):
            band0 = L.up_row0[c] // BAND_ROWS
            seq.append(("fill", "w", c, "pool", s * L.expert_bytes + 2 * band0 * L.band_bytes_up,
                        2 * L.up_bands[c] * L.band_bytes_up, s))
            seq.append(("drain", "y", c, "act", A.h + s * A.h_stride + 4 * L.up_row0[c],
                        Y_BYTES * L.up_bands[c], s))
    seq.append(("finish", "y", -1, "", 0, 0, -1))
    for s in range(L.topk):
        seq.append(("fill", "x", -1, "act", A.h + s * A.h_stride, L.h_elems * ELEM, s))
        for c in range(L.n_cores):
            band0 = L.down_row0[c] // BAND_ROWS
            seq.append(("fill", "w", c, "pool", s * L.expert_bytes + L.down_off + band0 * L.band_bytes_down,
                        L.down_bands[c] * L.band_bytes_down, s))
    for c in range(L.n_cores):
        seq.append(("drain", "y", c, "xres", 4 * L.down_row0[c], Y_BYTES * L.down_bands[c], -1))
    seq.append(("finish", "*", -1, "", 0, 0, -1))
    return seq


def _fill_bands(L: Layout, t: tuple) -> tuple[str, int, int]:
    """A pool fill's projection ("gu" or "d"), first band and band count, from its offset into its slot's block."""
    rel = t[4] - t[6] * L.expert_bytes
    kind, rel, per = ("d", rel - L.down_off, L.band_bytes_down) if rel >= L.down_off else \
        ("gu", rel, 2 * L.band_bytes_up)
    if rel % per or t[5] % per:
        raise AssertionError(f"core {t[2]} slot {t[6]}: a {t[5]} B fill at {t[4]} is not whole bands")
    return kind, rel // per, t[5] // per


def check(L: Layout, A: Act = Act()) -> dict:
    """OPEN-MOE-WIDE-FF's three checks, the fifo balances, the barrier and act offsets; raises on the first failure."""
    seq = host_sequence(L, A)
    rows_up = sum(L.up_bands) * BAND_ROWS
    rows_down = sum(L.down_bands) * BAND_ROWS
    if rows_up < L.ff or rows_down < L.hidden:
        raise AssertionError("the bands do not cover the projections' rows")
    if L.l1 > L1_BUDGET:
        raise AssertionError(f"core scratch {L.l1} B over the {L1_BUDGET} B budget")
    for c in range(L.n_cores):
        ev = core_program(L, c)
        for fifo, unit in (("w", CALL_BYTES), ("y", Y_BYTES)):
            used = sum(e[2] for e in ev if e[0] == "acq" and e[1] == fifo) * unit
            moved = sum(t[5] for t in seq if t[1] == fifo and t[2] == c)
            if used != moved:
                raise AssertionError(f"core {c} {fifo}: the program takes {used} B, the host moves {moved} B")
        held = {"w": 0, "x": 0, "y": 0}
        for e in ev:
            if e[0] in ("acq", "rel"):
                held[e[1]] += e[2] if e[0] == "acq" else -e[2]
                if held[e[1]] > X_DEPTH or held[e[1]] < 0:
                    raise AssertionError(f"core {c} holds {held[e[1]]} elements of a depth-{X_DEPTH} fifo {e[1]}")
        if any(held.values()):
            raise AssertionError(f"core {c} ends the token holding {held}")
    xs = sum(t[5] for t in seq if t[1] == "x")
    xu = sum(e[2] for e in core_program(L, 0) if e[0] == "acq" and e[1] == "x") * ELEM
    if xs != xu:
        raise AssertionError(f"x: the host broadcasts {xs} B, a core takes {xu} B")
    for s in range(L.topk):
        spans = sorted((t[4], t[4] + t[5]) for t in seq if t[3] == "pool" and t[6] == s)
        pos = s * L.expert_bytes
        for a, b in spans:
            if a != pos:
                raise AssertionError(f"slot {s}: pool gap or overlap at {a} (expected {pos})")
            pos = b
        if pos != (s + 1) * L.expert_bytes:
            raise AssertionError(f"slot {s}: reads end at {pos}, the block at {(s + 1) * L.expert_bytes}")
    base = [t[4] for t in seq if t[3] == "pool" and t[6] == 0]
    for s in range(1, L.topk):
        if [t[4] - s * L.expert_bytes for t in seq if t[3] == "pool" and t[6] == s] != base:
            raise AssertionError(f"slot {s} is not slot 0 moved by one expert stride")
    bar = next(i for i, t in enumerate(seq) if t[0] == "finish")
    h_out = [i for i, t in enumerate(seq) if t[0] == "drain" and t[6] >= 0]
    h_in = [i for i, t in enumerate(seq) if t[0] == "fill" and t[1] == "x" and t[6] >= 0]
    if seq[bar][1] not in ("y", "*") or not (h_out and h_in and max(h_out) < bar < min(h_in)):
        raise AssertionError("the barrier does not sit between the last h drain and the first h fill")
    for s in range(L.topk):
        d = sorted((t[4], t[4] + t[5]) for t in seq[:bar] if t[0] == "drain" and t[6] == s)
        lo = A.h + s * A.h_stride
        if d[0][0] != lo or d[-1][1] != lo + 4 * rows_up or any(d[i][1] != d[i + 1][0] for i in range(len(d) - 1)):
            raise AssertionError(f"slot {s}: the h drains do not tile its rows")
        if 4 * L.ff > L.h_elems * ELEM or A.h_stride < L.h_elems * ELEM:
            raise AssertionError(f"slot {s}: h does not fit its x elements or its act region")
    if [(t[4], t[5]) for t in seq[:bar] if t[1] == "x"] != [(A.xm, L.xm_elems * ELEM)] or \
            A.xm + L.xm_elems * ELEM > A.rout:
        raise AssertionError("xm is not the one x fill ahead of the barrier, read from its own act region")
    if [(t[4], t[5], t[6]) for t in seq[bar:] if t[1] == "x"] != \
            [(A.h + s * A.h_stride, L.h_elems * ELEM, s) for s in range(L.topk)]:
        raise AssertionError("the h fills after the barrier do not read each slot's own h, in slot order")
    for c in range(L.n_cores):
        w = [t for t in seq if t[0] == "fill" and t[1] == "w" and t[2] == c]
        if [t[3] for t in w] != ["act"] * L.hdr_elems + ["pool"] * (len(w) - L.hdr_elems):
            raise AssertionError(f"core {c}: its header fills do not lead its w stream")
        bands = [(_fill_bands(L, t), t[6]) for t in w[L.hdr_elems:]]
        if [(b[0], s) for b, s in bands] != [("gu", s) for s in range(L.topk)] + [("d", s) for s in range(L.topk)]:
            raise AssertionError(f"core {c}: its w fills are not in its program's order")
        downs = {b[1:] for b, _ in bands if b[0] == "d"}
        if len(downs) != 1:
            raise AssertionError(f"core {c}: its down rows differ between slots")
        (d0, dn), = downs
        want = [("act", A.h + s * A.h_stride + 4 * BAND_ROWS * b[1], Y_BYTES * b[2]) for b, s in bands if b[0] == "gu"]
        want.append(("xres", 4 * BAND_ROWS * d0, Y_BYTES * dn))
        if [t[3:6] for t in seq if t[0] == "drain" and t[2] == c] != want:
            raise AssertionError(f"core {c}: its drains do not land on the rows its fills compute")
        res = A.res + 4 * BAND_ROWS * d0
        if [t[4:6] for t in w[:L.hdr_elems]] != [(A.rout, CALL_BYTES), (res, CALL_BYTES)] or res + CALL_BYTES > A.h:
            raise AssertionError(f"core {c}: its header fills do not read the routing record and its own residual rows")
    # the Pipeline throttle waits on a channel's oldest drain before issuing its fourth, so its weights must be out
    for c in range(L.n_cores):
        q = [t for t in seq if t[2] == c and t[1] == "y"]
        for i in range(PIPELINE_DEPTH, len(q)):
            older = q[i - PIPELINE_DEPTH]
            if older[6] >= 0:
                fills_before = [t for t in seq[:seq.index(q[i])] if t[1] == "w" and t[2] == c and t[6] == older[6]]
                if not fills_before:
                    raise AssertionError(f"core {c}: a drain waits on slot {older[6]} before its weights are issued")
    mm2s = {t[1:3] for t in seq if t[0] == "fill"}
    s2mm = {t[1:3] for t in seq if t[0] == "drain"}
    if max(len(mm2s), len(s2mm)) > SHIM_CHANNELS:
        raise AssertionError(f"{len(mm2s)} fill and {len(s2mm)} drain streams, one shim channel each; "
                             f"the shim has {SHIM_CHANNELS} each way")
    return {"l1": L.l1, "expert_bytes": L.expert_bytes, "tasks": len(seq),
            "w_elems_core0": sum(e[2] for e in core_program(L, 0) if e[0] == "acq" and e[1] == "w")}


TILE_SHAPES = {"mxfp4": (TILE_ROWS, TILE_COLS), "q4_1": (32, 256), "q8": (16, 256)}   # rows x K per 5120 B


def expert_stream_bytes(fmt: str, hidden: int = 2880, ff: int = 2880, topk: int = 4, layers: int = 24,
                        pad_rows: bool = False, pad_k: bool = False) -> int:
    """Expert bytes one token streams over all layers; q4_1 and q8 (gemv_q8.h's half-tiles) always pad K to 256."""
    if fmt == "mxfp4_ideal":
        return int(topk * layers * 3 * hidden * ff * 17 / 32)
    if fmt not in TILE_SHAPES:
        raise ValueError(fmt)
    rows_t, kt = TILE_SHAPES[fmt]
    kp = (lambda k: -(-k // 256) * 256) if (pad_k or kt == 256) else (lambda k: k)
    rp = (lambda r: -(-r // 512) * 512) if pad_rows else (lambda r: r)
    per = 2 * rp(ff) * kp(hidden) + rp(hidden) * kp(ff)
    return topk * layers * per * TILE_BYTES // (rows_t * kt)


def gguf_blocks(raw: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """(codes, exponents) of GGUF MXFP4 blocks: 17 B per 32 values, element j low nibble of byte j, j + 16 high."""
    blk = np.asarray(raw, np.uint8).reshape(rows, cols // 32, 17)
    q = blk[:, :, 1:]
    codes = np.concatenate([q & 0x0F, q >> 4], axis=2).reshape(rows, cols)
    return codes, blk[:, :, 0].copy()


def pack_tile(codes: np.ndarray, exps: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """codes [32, 288] uint8, exps [32, 9] uint8, bias f32[32] or None -> 5120 bytes."""
    t = np.zeros(TILE_BYTES, np.uint8)
    for half in range(2):
        cc = codes[16 * half:16 * half + 16].T.reshape(-1)          # p = k * 16 + row % 16
        t[half * NIB_PLANE:(half + 1) * NIB_PLANE] = cc[0::2] | (cc[1::2] << 4)
    t[EXP_OFF:EXP_OFF + TILE_ROWS * TILE_BLOCKS] = exps.T.reshape(-1)
    if bias is not None:
        b = (np.asarray(bias, np.float32).view(np.uint32) >> 16).astype(np.uint16)
        if not np.array_equal(b.astype(np.uint32) << 16, np.asarray(bias, np.float32).view(np.uint32)):
            raise ValueError("an output bias is not exactly a bf16 value")
        t[BIAS_OFF:BIAS_OFF + 64] = b.view(np.uint8)
    return t


def unpack_tile(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, np.uint8)
    codes = np.empty((TILE_ROWS, TILE_COLS), np.uint8)
    for half in range(2):
        b = t[half * NIB_PLANE:(half + 1) * NIB_PLANE]
        p = np.empty(2 * b.size, np.uint8)
        p[0::2], p[1::2] = b & 0x0F, b >> 4
        codes[16 * half:16 * half + 16] = p.reshape(TILE_COLS, 16).T
    exps = t[EXP_OFF:EXP_OFF + TILE_ROWS * TILE_BLOCKS].reshape(TILE_BLOCKS, TILE_ROWS).T.copy()
    bias = (t[BIAS_OFF:BIAS_OFF + 64].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return codes, exps, bias


def _whole(rows: int, K: int) -> None:
    if rows % BAND_ROWS or K % TILE_COLS:
        raise ValueError(f"a {rows} x {K} projection is not whole {BAND_ROWS}-row bands of {TILE_COLS}-column tiles")


def pack_projection(codes: np.ndarray, exps: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """[rows, K] into bands, chunk i = row half i % 2, k-tile i // 2 (gemv_q4.h's law); bias in k-tile 0."""
    rows, K = codes.shape
    _whole(rows, K)
    nk = K // TILE_COLS
    out = np.empty((rows // BAND_ROWS, 2 * nk, TILE_BYTES), np.uint8)
    for b in range(rows // BAND_ROWS):
        for i in range(2 * nk):
            r0, kt = b * BAND_ROWS + TILE_ROWS * (i % 2), i // 2
            rs = slice(r0, r0 + TILE_ROWS)
            out[b, i] = pack_tile(codes[rs, kt * TILE_COLS:(kt + 1) * TILE_COLS],
                                  exps[rs, kt * TILE_BLOCKS:(kt + 1) * TILE_BLOCKS],
                                  bias[rs] if kt == 0 else None)
    return out


def pack_expert(gate, up, down) -> np.ndarray:
    """One expert block from (codes, exps, bias) per projection: gate and up interleaved by band, then down."""
    g, u, d = (pack_projection(*p) for p in (gate, up, down))
    gu = np.stack([g, u], axis=1).reshape(-1)
    return np.concatenate([gu, d.reshape(-1)])


def dequant_projection(bands: np.ndarray, rows: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """(values [rows, K] f32, bias [rows] f32) read back out of pack_projection's bytes."""
    _whole(rows, K)
    nk = K // TILE_COLS
    bands = bands.reshape(rows // BAND_ROWS, 2 * nk, TILE_BYTES)
    w = np.empty((rows, K), np.float32)
    bias = np.empty(rows, np.float32)
    for b in range(rows // BAND_ROWS):
        for i in range(2 * nk):
            codes, exps, bb = unpack_tile(bands[b, i])
            r0, kt = b * BAND_ROWS + TILE_ROWS * (i % 2), i // 2
            w[r0:r0 + TILE_ROWS, kt * TILE_COLS:(kt + 1) * TILE_COLS] = \
                KVALUES[codes] * np.repeat(e8m0(exps), 32, axis=1)
            if kt == 0:
                bias[r0:r0 + TILE_ROWS] = bb
    return w, bias


def bf16(x) -> np.ndarray:
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (u.astype(np.uint32) << 16).view(np.float32)


def table_quant(x) -> np.ndarray:
    """What gemv_tab.h's table holds: per 32-block, x rounded to int16 at 2^(14 - floor(log2 max|x|))."""
    xb = bf16(x).reshape(-1, 32).astype(np.float64)
    mx = np.abs(xb).max(1, keepdims=True)
    s = np.where(mx > 0, 14 - np.floor(np.log2(np.where(mx > 0, mx, 1))), 0)
    return (np.round(xb * 2.0 ** s) / 2.0 ** s).reshape(-1)


def emulate_block(L: Layout, xm, res, Wg, bg, Wu, bu, Wd, bd, ids, wts) -> np.ndarray:
    """The block output the cores write: f32 per core band over table-quantized activations, h through bf16."""
    from replica_gptoss import clamped_swiglu
    xq = table_quant(xm)
    acc = np.zeros(L.hidden, np.float32)
    hs = []
    for e in ids:
        h = np.zeros(L.ff, np.float32)
        for c in range(L.n_cores):
            r0, n = L.up_row0[c], BAND_ROWS * L.up_bands[c]
            g = (Wg[e][r0:r0 + n].astype(np.float32) @ xq.astype(np.float32)) + bg[e][r0:r0 + n]
            u = (Wu[e][r0:r0 + n].astype(np.float32) @ xq.astype(np.float32)) + bu[e][r0:r0 + n]
            h[r0:r0 + n] = clamped_swiglu(g, u).astype(np.float32)
        hs.append(h)
    for e, w, h in zip(ids, wts, hs):
        hq = table_quant(h).astype(np.float32)
        for c in range(L.n_cores):
            r0, n = L.down_row0[c], BAND_ROWS * L.down_bands[c]
            y = (Wd[e][r0:r0 + n].astype(np.float32) @ hq) + bd[e][r0:r0 + n]
            acc[r0:r0 + n] += np.float32(w) * y
    return np.asarray(res, np.float32) + acc


def _gguf(path: str):
    from gguf_pool import read_gguf
    meta, tensors, base = read_gguf(path)
    return tensors, base


def _read(path: str, base: int, t: dict, nbytes: int, off: int = 0) -> np.ndarray:
    return np.fromfile(path, np.uint8, count=nbytes, offset=base + t["offset"] + off)


def _q8_0_row(path, base, t, row: int, cols: int) -> np.ndarray:
    nb = cols // 32
    raw = _read(path, base, t, nb * 34, row * nb * 34).reshape(nb, 34)
    d = raw[:, :2].copy().view(np.float16).astype(np.float32)
    q = raw[:, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def slice_check(path: str, layer: int = 0, token: int = 13225) -> dict:
    """One layer's routed experts packed and read back (must be exact), then the block against fp64."""
    from replica_gptoss import expert_ffn, route
    tensors, base = _gguf(path)
    H = 2880
    pre = f"blk.{layer}."
    f32 = lambda name, n: _read(path, base, tensors[name], 4 * n).view(np.float32)  # noqa: E731
    emb = _q8_0_row(path, base, tensors["token_embd.weight"], token, H)
    ln2 = f32(pre + "post_attention_norm.weight", H)
    xm = (emb / np.sqrt(np.mean(emb.astype(np.float64) ** 2) + 1e-5) * ln2).astype(np.float32)
    E = 32
    Wr = f32(pre + "ffn_gate_inp.weight", H * E).reshape(E, H)
    br = f32(pre + "ffn_gate_inp.bias", E)
    ids, wts = route(xm, Wr, br, 4)
    nblk = H * H // 32
    W, B = {}, {}
    for proj in ("gate", "up", "down"):
        t = tensors[f"{pre}ffn_{proj}_exps.weight"]
        B[proj] = f32(f"{pre}ffn_{proj}_exps.bias", H * E).reshape(E, H)
        W[proj] = {}
        for e in ids:
            codes, exps = gguf_blocks(_read(path, base, t, nblk * 17, int(e) * nblk * 17), H, H)
            W[proj][int(e)] = (codes, exps)
    L = layout(H, H)
    exact = True
    Wf = {p: {} for p in W}
    for e in ids:
        e = int(e)
        blk = pack_expert(*[(W[p][e][0], W[p][e][1], B[p][e]) for p in ("gate", "up", "down")])
        if blk.size != L.expert_bytes:
            raise AssertionError(f"expert block {blk.size} B, the layout says {L.expert_bytes}")
        gu = blk[:L.down_off].reshape(H // BAND_ROWS, 2, -1)
        for p, part in (("gate", gu[:, 0]), ("up", gu[:, 1]), ("down", blk[L.down_off:])):
            w, bb = dequant_projection(part, H, H)
            ref = KVALUES[W[p][e][0]] * np.repeat(e8m0(W[p][e][1]), 32, axis=1)
            exact &= bool(np.array_equal(w, ref)) and bool(np.array_equal(bb, B[p][e]))
            Wf[p][e] = w
    res = np.zeros(H, np.float32)
    # expert_ffn per routed expert, not moe_block: its [32, 2880, 2880] stack would be 3 GB per projection
    ref = np.zeros(H, np.float64)
    for e, w in zip(ids, wts):
        e = int(e)
        ref += w * expert_ffn(xm, Wf["gate"][e], B["gate"][e], Wf["up"][e], B["up"][e], Wf["down"][e], B["down"][e])
    got = emulate_block(L, xm, res, Wf["gate"], B["gate"], Wf["up"], B["up"], Wf["down"], B["down"],
                        [int(e) for e in ids], wts)
    err = got.astype(np.float64) - ref
    return {"layer": layer, "token": token, "experts": [int(e) for e in ids],
            "weights": [round(float(w), 4) for w in wts], "tiles_exact": exact,
            "rel_err": float(np.linalg.norm(err) / np.linalg.norm(ref)),
            "corr": float(np.corrcoef(got.astype(np.float64), ref)[0, 1]),
            "max_abs_err": float(np.abs(err).max()), "ref_rms": float(np.sqrt(np.mean(ref ** 2)))}


def report(L: Layout) -> str:
    gb = lambda b: f"{b / 1e9:.3f} GB"  # noqa: E731
    r = check(L)
    lines = [f"bands per core, gate/up {L.up_bands}, down {L.down_bands}",
             f"expert block {L.expert_bytes} B; layer pool {L.layer_pool_bytes()} B; "
             f"resident experts over 24 layers {gb(24 * L.layer_pool_bytes())}",
             f"x elements: xm {L.xm_elems}, h {L.h_elems}; ms {L.ms_floats} floats; "
             f"table K {L.table_k}; core L1 {L.l1} of {L1_BUDGET} B",
             f"stream check: {r['tasks']} DMA tasks, core 0 takes {r['w_elems_core0']} w elements; balanced",
             "expert bytes over 24 layers, streamed per token and resident:"]
    for name, kw in (("MXFP4 as shipped (GGUF bytes)", dict(fmt="mxfp4_ideal")),
                     ("MXFP4 tiles 288 wide, rows dealt unevenly", dict(fmt="mxfp4")),
                     ("MXFP4 tiles, rows padded to 3072", dict(fmt="mxfp4", pad_rows=True)),
                     ("q4_1 re-quantized, K padded, rows dealt unevenly", dict(fmt="q4_1")),
                     ("q4_1 re-quantized, both padded to 3072", dict(fmt="q4_1", pad_rows=True)),
                     ("int8 (exact) in q8 half-tiles, K padded, uneven", dict(fmt="q8")),
                     ("int8 (exact) in q8 half-tiles, both padded", dict(fmt="q8", pad_rows=True))):
        b = expert_stream_bytes(**kw)
        lines.append(f"  {name:52s} {gb(b)}  {gb(b * L.experts // L.topk)}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", help="gpt-oss-20b MXFP4 GGUF for the one-layer slice")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=13225)
    ap.add_argument("--table-k", type=int, default=None,
                    help="the widest K the core holds a table for (4096 when it also runs o_proj)")
    a = ap.parse_args()
    print(report(layout(table_k=a.table_k)))
    if a.gguf:
        print(slice_check(a.gguf, a.layer, a.token))
    return 0


if __name__ == "__main__":
    sys.exit(main())
