r"""gemm_q4: Y[M, N] = X[M, K] . W[N, K]^T with W in q4_1 pool chunks -- the batched-prefill
experiment (issue #16 item 1; .claude/plans/issue-16-prefill-attention-vision.md).

mlir-aie's whole_array matmul (4 rows x 8 columns of cores: A row-blocks forwarded per
row, B column-blocks forwarded per column, C joined per column through the memtiles)
with W as the B operand in its own orientation. Each column owns N/8 output rows of W in
32-row chunk-rows; each row of cores owns M/4 tokens. A core takes one raw 5 KB chunk
at a time, dequantizes each 128-k half into a bf16 tile laid out for mm.cc's row-major
B (dequant_q4.cc), and runs the 4x8x8 bf16 mmul against the matching A tile (re-tiled by
the memtile from row-major X). The C tiles (m x 32, f32) join per column and land in Y
row-major.

The question it answers: does the weight stream amortize over a token block at a rate
worth a driver? Today prefill re-streams every weight byte per token (decode-as-prefill,
~8 tok/s on a 4B); the closed engine's batch kernels do ~500.

Env: GEMM_M (tokens, default 128), GEMM_K, GEMM_N (default Qwen3-4B's q projection,
2560 -> 4096). Pool order: q4_1_pack.pack_q4_1_pool(blocks, rs=2), the standard band
layout (64-row bands; chunk c in a band: row half c % 2, k-tile c // 2).
Build (WSL): python build_design.py designs/gemm_q4/gemm_q4.py designs/gemm_q4/build
Test: python designs/gemm_q4/make_test.py [--out DIR]; harness/out/run_kernel.exe DIR/run.cfg;
      python designs/gemm_q4/make_test.py --compare
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

M = int(os.environ.get("GEMM_M", 128))
K = int(os.environ.get("GEMM_K", 2560))
N = int(os.environ.get("GEMM_N", 4096))
ROWS, COLS = 4, 8
m = M // ROWS                 # tokens per core row
n = 32                        # W rows per chunk (one output tile's width)
KH = 128                      # k per A tile and per dequant half
TILE_K = 256                  # k per chunk
CHUNK = 5120                  # chunk bytes
KT = K // TILE_K              # chunks along K per chunk-row
NJ = N // n // COLS           # chunk-rows per column
PER_BAND = 2 * KT             # chunks per 64-row band (rs = 2)
POOL_BYTES = (N // 32) * KT * CHUNK
assert M % ROWS == 0 and m % 8 == 0, "M must be a multiple of 32"
assert K % TILE_K == 0 and N % (2 * n * COLS) == 0, "K a multiple of 256, N of 512"


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemm_q4(x: In, pool: In, y: InOut, *, srchash: CompileTime[int] = 0):
    x_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(M * N,), np.dtype[np.float32]]
    a_l2 = np.ndarray[(m * KH,), np.dtype[bfloat16]]        # one A tile, row-major, in L2
    a_l1 = np.ndarray[(m, KH), np.dtype[bfloat16]]          # (4, 8)-tiled by the memtile on the way in
    w_ty = np.ndarray[(CHUNK,), np.dtype[np.uint8]]         # one raw chunk
    bt_ty = np.ndarray[(KH * n,), np.dtype[bfloat16]]       # the dequantized half, mm.cc's B layout
    c_l1 = np.ndarray[(m, n), np.dtype[np.float32]]
    c_l2 = np.ndarray[(ROWS * m * n,), np.dtype[np.float32]]

    mm = kernels.mm(dim_m=m, dim_k=KH, dim_n=n, input_dtype=bfloat16, output_dtype=np.float32)
    zero = mm.zero
    r, s, t = mm.mac_dims
    inc = include_dirs()
    dq = ExternalFunction("dequant_q4_half", source_file=str(HERE / "dequant_q4.cc"),
                          arg_types=[w_ty, bt_ty, np.int32], include_dirs=inc, compile_flags=[f"-DDQ_KH={KH}"])

    # A: per row of cores, shim -> memtile -> the row's eight cores, re-tiled (r, s)
    a_dims = [(m // r, r * KH), (KH // s, s), (r, KH), (s, 1)]
    a_l3l2 = [ObjectFifo(a_l2, name=f"A_L3L2_{i}", depth=2) for i in range(ROWS)]
    a_l2l1 = [f.cons().forward(obj_type=a_l1, name=f"A_L2L1_{i}", dims_to_stream=a_dims)
              for i, f in enumerate(a_l3l2)]
    # B: per column, raw chunks, shim -> memtile -> the column's four cores
    b_l3l2 = [ObjectFifo(w_ty, name=f"B_L3L2_{c}", depth=2) for c in range(COLS)]
    b_l2l1 = [f.cons().forward(obj_type=w_ty, name=f"B_L2L1_{c}") for c, f in enumerate(b_l3l2)]
    # C: per column, the four rows' tiles joined; (r, t)-tiled -> row-major on the way out
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    c_l2l3, c_l1l2 = [], [[] for _ in range(ROWS)]
    for c in range(COLS):
        f = ObjectFifo(c_l2, name=f"C_L2L3_{c}", depth=2, dims_to_stream=c_dims)
        c_l2l3.append(f)
        parts = f.prod().join([m * n * i for i in range(ROWS)], obj_types=[c_l1] * ROWS,
                              names=[f"C_L1L2_{c}_{i}" for i in range(ROWS)], depths=[2] * ROWS)
        for i in range(ROWS):
            c_l1l2[i].append(parts[i])

    def core(in_a, in_b, out_c, bt, f_zero, f_mm, f_dq):
        for _ in range_(NJ):                                  # this column's chunk-rows
            ce = out_c.acquire(1)
            f_zero(ce)
            for _ in range_(KT):                              # along K, one chunk at a time
                we = in_b.acquire(1)
                for h in range(TILE_K // KH):                 # its halves (Python-level: constant)
                    ae = in_a.acquire(1)
                    f_dq(we, bt, h)
                    f_mm(ae, bt, ce)
                    in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    grid = Worker.grid(ROWS, COLS, lambda row, col: Worker(
        core, [a_l2l1[row].cons(), b_l2l1[col].cons(), c_l1l2[row][col].prod(),
               Buffer(bt_ty, name=f"bt_{row}_{col}"), zero, mm, dq], stack_size=0xD00))
    workers = [w for rw in grid for w in rw]

    def tap(total, off, sizes, strides):
        return TensorAccessPattern((1, total), off, sizes, strides)

    def sequence(X, P, Y, a_prods, b_prods, c_conss):
        pa, pb, pc = Pipeline(2), Pipeline(2), Pipeline(2)
        for j in range(NJ):
            band, half = j // 2, j % 2
            for col in range(COLS):                           # the tile this column emits for j
                pc.drain(c_conss[col], Y, tap(M * N, n * (col * NJ + j), [1, 1, M, n], [0, 0, N, 1]))
            for col in range(COLS):                           # its KT chunks: every other chunk of the band
                bnd = col * (NJ // 2) + band
                pb.fill(b_prods[col], P, tap(POOL_BYTES, (bnd * PER_BAND + half) * CHUNK,
                                             [1, 1, KT, CHUNK], [0, 0, 2 * CHUNK, 1]))
            for row in range(ROWS):                           # the row's tokens, as K/KH tiles
                pa.fill(a_prods[row], X, tap(M * K, row * m * K, [1, K // KH, m, KH], [0, KH, K, 1]))
        pa.finish()
        pb.finish()
        pc.finish()

    rt = Runtime(sequence, [x_ty, pool_ty, y_ty,
                            [f.prod() for f in a_l3l2], [f.prod() for f in b_l3l2], [f.cons() for f in c_l2l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = gemm_q4
_src = b"".join([(HERE / "dequant_q4.cc").read_bytes(), (HERE / "gemm_q4.py").read_bytes(),
                 f"{M}x{K}x{N}".encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
