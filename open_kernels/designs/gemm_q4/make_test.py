r"""Inputs, reference and harness config for the gemm_q4 experiment, and the comparison.

    python designs/gemm_q4/make_test.py [--out DIR] [--m 128 --k 2560 --n 4096] [--runs 5]
    open_kernels/harness/out/run_kernel.exe DIR/run.cfg
    python designs/gemm_q4/make_test.py --compare [--out DIR]

x: bf16[M, K] ~ N(0, 1); W: random q4_1 blocks packed into the standard pool order
(q4_1_pack, rs = 2). The reference is fp64 X . W^T with W dequantized exactly (bf16 d / m)
and then rounded to bf16 once, which is what dequant_q4.cc produces. The harness prints one
timing line per `run`; the first includes the context's warm-up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from q4_1_pack import dequant_pool, pack_q4_1_pool, random_q4_1_blocks  # noqa: E402

CHUNK = 5120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "test"))
    ap.add_argument("--build", default=str(HERE / "build"))
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--k", type=int, default=2560)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    M, K, N = a.m, a.k, a.n

    if a.compare:
        ref = np.fromfile(out / "ref.bin", dtype=np.float32).reshape(M, N).astype(np.float64)
        y = np.fromfile(out / "y.bin", dtype=np.float32).reshape(M, N).astype(np.float64)
        corr = np.corrcoef(ref.ravel(), y.ravel())[0, 1]
        err = np.abs(y - ref)
        rel = err.max() / (np.abs(ref).max() + 1e-30)
        print(f"corr {corr:.8f}  max|err| {err.max():.4e}  max|err|/max|ref| {rel:.3e}  "
              f"mean|err| {err.mean():.3e}  nonfinite {np.count_nonzero(~np.isfinite(y))}")
        bad = np.argwhere(err > 1e-2 * np.abs(ref).max())
        if len(bad):
            print(f"{len(bad)} elements off by > 1% of max; first: {bad[:5].tolist()}")
        macs = M * N * K
        print(f"MACs per run: {macs / 1e9:.3f} G  (a 1 ms run is {2 * macs / 1e9:.1f} TFLOPS)")
        return 0 if corr > 0.9999 and rel < 1e-2 else 1

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((M, K), dtype=np.float32).astype(bfloat16)
    blocks = random_q4_1_blocks(N, K, rng, scale=0.05)
    pool = pack_q4_1_pool(blocks, rs=2)
    assert pool.size == (N // 32) * (K // 256) * CHUNK, pool.size
    w = dequant_pool(pool, N, K, rs=2).astype(bfloat16).astype(np.float64)   # the kernel's one rounding
    ref = (x.astype(np.float64) @ w.T).astype(np.float32)
    x.tofile(out / "x.bin")
    pool.tofile(out / "pool.bin")
    ref.tofile(out / "ref.bin")
    b = Path(a.build)
    cfg = ["device", f"xclbin G {b / 'final.xclbin'}", f"kernelx k G {b / 'insts.bin'}",
           f"buf x {M * K * 2} {out / 'x.bin'}", f"buf pool {pool.size} {out / 'pool.bin'}", f"buf y {M * N * 4}"]
    cfg += ["run k x pool y"] * a.runs
    cfg += [f"dump y {out / 'y.bin'}"]
    (out / "run.cfg").write_text("\n".join(cfg) + "\n")
    print(f"wrote {out}: x {x.nbytes} B, pool {pool.size} B, ref {ref.nbytes} B, run.cfg ({a.runs} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
