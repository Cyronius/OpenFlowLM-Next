r"""fp64 reference for a Gemma 3 prompt holding an image, read from the GGUF the way the engine packs it
(OPEN-VISION-GEMMA3).

    python open_kernels/model/replica_gemma3_image.py --gguf FILE --spec open_kernels/recipes/specs/gemma3-4b.json \
        --ids-file IDS --rows ROWS.bin --mask bidir --out DIR

The prompt runs layer by layer over every position at once. `--rows` (f32 [256 * images, hidden]) replaces
the image tokens' embeddings, unscaled, as transformers' masked_scatter does. `--mask bidir` is Gemma 3's:
each image's tokens see each other both ways and everything before them causally, inside the sliding
window on the local layers; `--mask causal` is the one Gemma 3 does not use, for the A/B. Writes
ref_logits_p<position>.bin for every text position after the last image token, as open_qwen36_cli's
--prefill-logits names them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from gguf_model import GGUFModel  # noqa: E402
from recipes.load import load_spec  # noqa: E402
from recipes.manifest import manifest  # noqa: E402
from recipes.spec import DENSE_LOCAL  # noqa: E402
from replica_dense import final_logits, gelu_tanh, rms  # noqa: E402


def allowed(T: int, groups: np.ndarray, window: int, bidir: bool) -> np.ndarray:
    """[T, T] True where query i may read key j: causal (inside `window` when it is set), OR'd with the
    query's own image block when `bidir`; transformers 5's create_masks_for_vision_model."""
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    a = j <= i
    if bidir:
        a = a | ((groups[:, None] >= 0) & (groups[:, None] == groups[None, :]))
    if window:
        a = a & (i - j < window)
    return a


def rope_rows(t: np.ndarray, pos: np.ndarray, rot: int, inv: np.ndarray) -> np.ndarray:
    """t [T, heads, hd] rotated per row by its own position, half-split pairs over the first `rot` dims."""
    half = rot // 2
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    y = t.copy()
    x1, x2 = t[..., :half], t[..., half:rot]
    y[..., :half] = x1 * c - x2 * s
    y[..., half:rot] = x2 * c + x1 * s
    return y


def layer(m, spec, l: int, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pre = f"model.layers.{l}."
    hid, nh, kvh, hd, ff = spec.hidden, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.intermediate
    local = spec.layer_types[l] == DENSE_LOCAL
    eps = spec.norm_eps
    inv = np.asarray(spec.rope_inv_freq(local=local), np.float64)
    T = X.shape[0]
    x = (rms(X, eps) * m.bf16(pre + "input_layernorm.weight")).astype(np.float32)
    q = (x @ m.matmul_w(pre + "self_attn.q_proj.weight", nh * hd, hid).T).reshape(T, nh, hd).astype(np.float64)
    k = (x @ m.matmul_w(pre + "self_attn.k_proj.weight", kvh * hd, hid).T).reshape(T, kvh, hd).astype(np.float64)
    v = (x @ m.matmul_w(pre + "self_attn.v_proj.weight", kvh * hd, hid).T).reshape(T, kvh, hd).astype(np.float64)
    q = rms(q, eps) * m.bf16(pre + "self_attn.q_norm.weight")
    k = rms(k, eps) * m.bf16(pre + "self_attn.k_norm.weight")
    pos = np.arange(T, dtype=np.float64)
    q = rope_rows(q, pos, spec.rotary_dim, inv)
    k = rope_rows(k, pos, spec.rotary_dim, inv)
    o = np.zeros((T, nh, hd))
    grp = nh // kvh
    for h in range(nh):
        s = (q[:, h] @ k[:, h // grp].T) / np.sqrt(hd)
        s = np.where(mask, s, -np.inf)
        a = np.exp(s - s.max(-1, keepdims=True))
        o[:, h] = (a / a.sum(-1, keepdims=True)) @ v[:, h // grp]
    out = o.reshape(T, nh * hd).astype(np.float32) @ m.matmul_w(pre + "self_attn.o_proj.weight", hid, nh * hd).T
    t = (rms(out, eps) * m.bf16(pre + "post_attention_layernorm.weight")).astype(np.float32)
    res = X + t
    xm = (rms(res, eps) * m.bf16(pre + "pre_feedforward_layernorm.weight")).astype(np.float32)
    g = xm @ m.matmul_w(pre + "mlp.gate_proj.weight", ff, hid).T
    u = xm @ m.matmul_w(pre + "mlp.up_proj.weight", ff, hid).T
    out2 = (gelu_tanh(g) * u).astype(np.float32) @ m.matmul_w(pre + "mlp.down_proj.weight", hid, ff).T
    t2 = (rms(out2, eps) * m.bf16(pre + "post_feedforward_layernorm.weight")).astype(np.float32)
    m._w.clear()
    return (res + t2).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--image-token", type=int, default=262144)
    ap.add_argument("--mask", choices=("bidir", "causal"), default="bidir")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = load_spec(Path(a.spec))
    man = manifest(spec)
    m = GGUFModel(a.gguf, embed_scale=man["gguf"]["pack"]["embed"].get("scale", 1.0))
    m.hidden = spec.hidden
    ids = [int(t) for t in Path(a.ids_file).read_text().replace("\n", ",").split(",") if t.strip()]
    rows = np.fromfile(a.rows, np.float32).reshape(-1, spec.hidden)
    T = len(ids)
    is_img = np.array([t == a.image_token for t in ids])
    if is_img.sum() != rows.shape[0]:
        sys.exit(f"{is_img.sum()} image tokens but {rows.shape[0]} rows")
    starts = is_img & ~np.concatenate([[False], is_img[:-1]])
    groups = np.where(is_img, np.cumsum(starts) - 1, -1)
    X = np.zeros((T, spec.hidden), np.float32)
    r = 0
    for p, t in enumerate(ids):
        if is_img[p]:
            X[p] = rows[r]
            r += 1
        else:
            X[p] = m.embed(t, spec.hidden)
    nl = a.layers or spec.num_layers
    masks = {w: allowed(T, groups, w, a.mask == "bidir") for w in {0, spec.sliding_window}}
    for l in range(nl):
        w = spec.sliding_window if spec.layer_types[l] == DENSE_LOCAL else 0
        X = layer(m, spec, l, X, masks[w])
        print(f"  layer {l} ({spec.layer_types[l]}): |res| {np.abs(X).max():.2f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    last_img = int(np.nonzero(is_img)[0].max()) if is_img.any() else T - 2
    for p in range(last_img + 1, T):
        _, lg = final_logits(m, spec, X[p])
        lg = np.asarray(lg, np.float32)[: spec.vocab]
        lg.tofile(out / f"ref_logits_p{p}.bin")
        if p == T - 1:
            top = np.argsort(-lg[: spec.real_vocab])[:5]
            print(f"position {p} ({a.mask}): top-5 {top.tolist()} logits {[round(float(lg[t]), 3) for t in top]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
