r"""What re-quantizing GPT-OSS's MXFP4 experts to q4_1 does to the weights (2026-10-16).

    python -I utilities/quant-compare/mxfp4_requant.py err  <gpt-oss MXFP4 .gguf> [--layers 0,11,23] [--experts 0,17,31]
    python -I utilities/quant-compare/mxfp4_requant.py diff <a.gguf> <b.gguf>

`err` decodes sampled expert matrices, re-quantizes every 32-value block with the engine's GGUF law
(src/open_qwen36/pools.cpp std_perm_gguf: min/max, fp16 d and m, then bf16 inside the kernel) and
prints the relative weight error, the share of blocks that survive exactly, and the same for the
lossless int8 form. `diff` says which tensors two GGUFs store byte for byte the same.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
from gguf_pool import read_gguf  # noqa: E402
from mxfp4 import KVALUES, e8m0  # noqa: E402

MXFP4 = 39
BLOCK_BYTES = {0: (1, 4), 1: (1, 2), 30: (1, 2), 2: (32, 18), 3: (32, 20), 8: (32, 34), MXFP4: (32, 17)}


def tensor_bytes(t: dict) -> int:
    n = int(np.prod(t["dims"], dtype=np.int64))
    per, size = BLOCK_BYTES[t["gguf_type"]]
    return n // per * size


def mxfp4_rows(path: str, base: int, t: dict, expert: int) -> tuple[np.ndarray, np.ndarray]:
    """One expert's [rows, cols] f32 matrix out of a [cols, rows, experts] MXFP4 tensor, and its
    E8M0 exponent bytes."""
    cols, rows = t["dims"][0], t["dims"][1]
    nblk = rows * cols // 32
    raw = np.fromfile(path, np.uint8, count=nblk * 17, offset=base + t["offset"] + expert * nblk * 17)
    blk = raw.reshape(nblk, 17)
    q = blk[:, 1:]
    codes = np.concatenate([q & 0x0F, q >> 4], axis=1)
    w = (KVALUES[codes] * e8m0(blk[:, 0])[:, None]).reshape(rows, cols).astype(np.float32)
    return w, blk[:, 0].copy()


def fp16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def bf16_rne(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (u.astype(np.uint32) << 16).view(np.float32)


def requant_q4_1(w: np.ndarray, kernel_bf16: bool = True) -> np.ndarray:
    v = w.reshape(-1, 32)
    lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
    d, m = fp16((hi - lo) / 15.0), fp16(lo)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    t = (v - m) * inv
    q = np.clip(np.where(t >= 0, np.floor(t + 0.5), np.ceil(t - 0.5)), 0, 15)
    if kernel_bf16:
        d, m = bf16_rne(d), bf16_rne(m)
    return (q * d + m).reshape(w.shape).astype(np.float32)


def err(args) -> int:
    meta, tensors, base = read_gguf(args.gguf)
    layers = [int(x) for x in args.layers.split(",")]
    experts = [int(x) for x in args.experts.split(",")]
    print(f"{'tensor':32s} {'expert':>6s} {'rel rms err':>12s} {'exact blocks':>13s} "
          f"{'fp16 d/m only':>14s} {'e8m0 range':>11s}")
    tot_e2 = tot_w2 = 0.0
    emin, emax = 255, 0
    for l in layers:
        for proj in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            name = f"blk.{l}.{proj}.weight"
            t = tensors[name]
            if t["gguf_type"] != MXFP4:
                raise SystemExit(f"{name} is type {t['gguf_type']}, not MXFP4 ({MXFP4})")
            for e in experts:
                w, ex = mxfp4_rows(args.gguf, base, t, e)
                r = requant_q4_1(w)
                r16 = requant_q4_1(w, kernel_bf16=False)
                e2, w2 = float(((r - w).astype(np.float64) ** 2).sum()), float((w.astype(np.float64) ** 2).sum())
                tot_e2 += e2
                tot_w2 += w2
                exact = float((r.reshape(-1, 32) == w.reshape(-1, 32)).all(1).mean())
                rel16 = float(np.sqrt(((r16 - w).astype(np.float64) ** 2).sum() / w2))
                emin, emax = min(emin, int(ex.min())), max(emax, int(ex.max()))
                print(f"{name:32s} {e:6d} {np.sqrt(e2 / w2):12.4f} {exact:13.4f} {rel16:14.4f} "
                      f"{int(ex.min()):5d}-{int(ex.max()):<5d}")
    print(f"\nall sampled matrices: rel rms err {np.sqrt(tot_e2 / tot_w2):.4f}")
    # int8 code = KVALUES[nibble], scale = 2^(e-128): exact whenever that scale is a normal bf16
    print(f"E8M0 bytes {emin}..{emax}: the int8 form (codes = 2 x e2m1, scale 2^(e-128)) is "
          f"{'exact in bf16' if emin >= 2 and emax <= 254 else 'NOT exact (scale outside bf16 normals)'}")
    return 0


def sha(path: str, off: int, n: int) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        f.seek(off)
        left = n
        while left:
            b = f.read(min(left, 64 << 20))
            if not b:
                raise SystemExit(f"{path}: short read at {off}")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def diff(args) -> int:
    _, ta, ba = read_gguf(args.a)
    _, tb, bb = read_gguf(args.b)
    same, moved, retyped = [], [], []
    for name in sorted(set(ta) | set(tb)):
        if name not in ta or name not in tb:
            print(f"only in one file: {name}")
            continue
        x, y = ta[name], tb[name]
        if x["gguf_type"] != y["gguf_type"] or x["dims"] != y["dims"]:
            retyped.append(f"{name}: type {x['gguf_type']} -> {y['gguf_type']}")
            continue
        n = tensor_bytes(x)
        (same if sha(args.a, ba + x["offset"], n) == sha(args.b, bb + y["offset"], n) else moved).append(name)
    print(f"{len(same)} tensors byte-identical, {len(moved)} same type with different bytes, "
          f"{len(retyped)} retyped")
    for line in retyped[:6]:
        print("  retyped:", line)
    if len(retyped) > 6:
        print(f"  ... {len(retyped) - 6} more retyped")
    for name in moved:
        print("  moved:", name)
    return 0 if not moved else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("err")
    e.add_argument("gguf")
    e.add_argument("--layers", default="0,11,23")
    e.add_argument("--experts", default="0,17,31")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = p.parse_args()
    return err(a) if a.cmd == "err" else diff(a)


if __name__ == "__main__":
    sys.exit(main())
