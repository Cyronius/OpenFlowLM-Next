r"""What the open engine's q8 head costs a qwen35 GGUF whose head is not Q8_0 (OPEN-GGUF-Q4NX-VIEW).

The view re-quantizes such a head (MiniCPM-V-4.7-1B ships Q6_K) to the q8 the family's kernel
reads. The NPU compare cannot see that: its reference reads the same q8 head. This runs the fp64
reference (open_kernels/model/replica_qwen35.py through the NumPy view) over a text once, then
scores every next token with the file's own head and with the view's q8 head.

    python utilities/quant-compare/gguf_head_nll.py --model-dir <dir with model.gguf + tokenizer.json> \
        --text README.md --tokens 96
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
sys.path.insert(0, str(ROOT / "open_kernels" / "model"))

from gguf_q4nx import GgufAsQ4nx  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
import replica_qwen35 as R35  # noqa: E402


def nll(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    m = logits.max(1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(logits - m).sum(1)))
    return lse - logits[np.arange(len(targets)), targets]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    md = Path(a.model_dir)
    spec = spec_from_model_dir(md)
    tk = Tokenizer.from_file(str(md / "tokenizer.json"))
    ids = tk.encode(Path(a.text).read_text(encoding="utf-8"), add_special_tokens=False).ids[:a.tokens + 1]
    q = GgufAsQ4nx(md / "model.gguf")
    q.hidden = spec.hidden
    plain, cache = q.matmul_w, {}
    q.matmul_w = lambda n, o, i: cache[n] if n in cache else cache.setdefault(n, plain(n, o, i))
    xs = [q.embed(t, spec.hidden).astype(np.float32) for t in ids[:-1]]
    t0 = time.time()
    for l in range(spec.num_layers):
        cache.clear()
        conv = np.zeros((spec.conv_kernel - 1, spec.lin_qkv_dim))
        S = np.zeros((spec.lin_value_heads, spec.lin_value_dim, spec.lin_value_dim))
        K = np.zeros((0, spec.num_kv_heads, spec.head_dim))
        V = np.zeros((0, spec.num_kv_heads, spec.head_dim))
        for t in range(len(xs)):
            xs[t], conv, S, K, V = R35.layer_decode(q, spec, l, xs[t], conv, S, K, V, t)
            xs[t] = np.asarray(xs[t], np.float32)
        print(f"  layer {l:2d} ({time.time() - t0:.0f} s)", flush=True)
    nw = q.bf16("model.norm.weight")
    hn = np.stack([(R35.rms(x, spec.norm_eps) * nw).astype(np.float32) for x in xs]).astype(np.float64)
    head = q.source_name("lm_head.weight")
    rows = q.g.shape(head)[0]
    real = min(spec.real_vocab, rows)
    exact = np.concatenate([q.g.rows_f32(head, r, min(rows, r + 16384)) for r in range(0, rows, 16384)])[:real]
    lq = (hn @ exact.T.astype(np.float64))
    w8 = q.dq_tile(q.raw("lm_head.weight"), -(-rows // 256) * 256, spec.hidden, 8704, requant=False)[:real]
    l8 = (hn @ w8.T.astype(np.float64))
    tg = np.asarray(ids[1:], np.int64)
    a_n, b_n = nll(lq, tg), nll(l8, tg)
    print(f"{len(tg)} next tokens, the file's {q.g.type_of(head)} head vs the view's q8 head:")
    print(f"  mean NLL {a_n.mean():.4f} vs {b_n.mean():.4f} nats (difference {b_n.mean() - a_n.mean():+.4f})")
    print(f"  argmax agrees at {(lq.argmax(1) == l8.argmax(1)).sum()} of {len(tg)} positions; "
          f"worst logit change {np.abs(lq - l8).max():.4f}")
    top2 = np.sort(lq, 1)[:, -2:]
    for p in np.nonzero(lq.argmax(1) != l8.argmax(1))[0]:
        print(f"    position {p}: the file's top two are {top2[p, 1] - top2[p, 0]:.4f} apart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
