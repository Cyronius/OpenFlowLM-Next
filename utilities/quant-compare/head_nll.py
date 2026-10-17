"""Teacher-forced NLL of a dense GGUF through the fp64 replica, its head as the engine packs it vs as the file stores it (2026-10-16)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

OK = Path(__file__).resolve().parents[2] / "open_kernels"
sys.path[:0] = [str(OK), str(OK / "model")]

import replica_dense as RD  # noqa: E402
from gguf_model import GGUFModel  # noqa: E402
from recipes.load import load_spec  # noqa: E402
from recipes.manifest import manifest  # noqa: E402


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--spec", required=True, help="the model's recipes/specs/*.json")
    ap.add_argument("--text", required=True, help="a text file, scored after the BOS token")
    ap.add_argument("--tokenizer", required=True, help="the model's tokenizer.json")
    ap.add_argument("--tokens", type=int, default=1024, help="score the first N tokens")
    ap.add_argument("--bos", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=128, help="positions per head product")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    spec = load_spec(Path(a.spec))
    ids = [a.bos] + Tokenizer.from_file(a.tokenizer).encode(Path(a.text).read_text(encoding="utf-8"),
                                                              add_special_tokens=False).ids
    ids = ids[:a.tokens + 1]
    m = manifest(spec, 4096)
    q = GGUFModel(a.gguf, embed_scale=m["gguf"]["pack"]["embed"].get("scale", 1.0))
    q.hidden = spec.hidden
    nl = spec.num_layers
    K = {l: np.zeros((0, spec.num_kv_heads, spec.head_dim)) for l in range(nl)}
    V = {l: np.zeros((0, spec.num_kv_heads, spec.head_dim)) for l in range(nl)}
    t0, hns = time.time(), []
    for t, tok in enumerate(ids[:-1]):
        xr = q.embed(tok, spec.hidden)
        for l in range(nl):
            xr, K[l], V[l] = RD.dense_decode(q, spec, l, xr.copy(), K[l], V[l], t)
        hns.append((RD.rms(xr, spec.norm_eps) * q.bf16("model.norm.weight")).astype(np.float32))
        if t % 64 == 0:
            print(f"  position {t} / {len(ids) - 1} ({time.time() - t0:.0f} s)", flush=True)
    H, tgt = np.stack(hns), np.array(ids[1:])
    np_, nf, tp, tf, kl, margin = [], [], [], [], [], []
    for c0 in range(0, len(H), a.chunk):
        y = tgt[c0:c0 + a.chunk]
        lp = log_softmax(q.head_logits(H[c0:c0 + a.chunk])[:spec.real_vocab].T)
        lf = log_softmax(q.head_logits(H[c0:c0 + a.chunk], exact=True)[:spec.real_vocab].T)
        np_.append(-lp[np.arange(len(y)), y])
        nf.append(-lf[np.arange(len(y)), y])
        tp.append(lp.argmax(1))
        tf.append(lf.argmax(1))
        kl.append(np.sum(np.exp(lf) * (lf - lp), 1))
        top2 = np.partition(lf, -2, axis=1)[:, -2:]
        margin.append(top2[:, 1] - top2[:, 0])
    np_, nf, tp, tf, kl, margin = map(np.concatenate, (np_, nf, tp, tf, kl, margin))
    g = q.name("lm_head.weight")
    print(f"{Path(a.gguf).name}: head {g} ({q.t[g]['type']}), {len(tgt)} positions of {Path(a.text).name}, "
          f"{nl} layers, fp64 replica ({time.time() - t0:.0f} s)")
    for name, n, top1 in (("packed (q4_1)", np_, tp), ("file", nf, tf)):
        print(f"  head {name:14s} mean NLL {n.mean():.4f} nats   ppl {np.exp(n.mean()):.3f}   "
              f"top-1 = next token {int((top1 == tgt).sum())}/{len(tgt)}")
    d, flips = np_ - nf, tp != tf
    print(f"  NLL(packed) - NLL(file): mean {d.mean():+.4f}  median {np.median(d):+.4f}  "
          f"worst {d.max():+.4f} nats")
    print(f"  argmax agrees at {int((~flips).sum())}/{len(tgt)} positions; of {int(flips.sum())} flips, "
          f"{int((margin[flips] < 0.5).sum())} are near-ties (file head's top-2 margin < 0.5 logits)")
    print(f"  KL(file || packed): mean {kl.mean():.5f}  max {kl.max():.5f} nats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
