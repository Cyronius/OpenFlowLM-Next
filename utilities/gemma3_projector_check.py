# Gemma 3's projector in numpy from an mmproj's own weights on fp32 tower rows, against a port's rows (2026-10-17).
import sys
import numpy as np
from gguf import GGUFReader


def tensor(r, name):
    for t in r.tensors:
        if t.name == name:
            a = np.array(t.data)
            return a.astype(np.float32).reshape([int(x) for x in reversed(t.shape)])
    raise SystemExit(f"no {name}")


def main():
    mmproj, tower_bin, rows_bin = sys.argv[1:4]
    r = GGUFReader(mmproj)
    w_norm = tensor(r, "mm.soft_emb_norm.weight")
    w_proj = tensor(r, "mm.input_projection.weight")
    print("soft_emb_norm", w_norm.shape, "mean", float(w_norm.mean()), "input_projection", w_proj.shape)
    h = np.fromfile(tower_bin, dtype=np.float32).reshape(64, 64, 1152).astype(np.float64)
    pooled = h.reshape(16, 4, 16, 4, 1152).mean(axis=(1, 3)).reshape(256, 1152)
    x = pooled / np.sqrt((pooled ** 2).mean(-1, keepdims=True) + 1e-6)
    # llama.cpp's converter stores this norm as 1 + w; a mean near 0 would mean it did not
    scale = w_norm if w_norm.mean() > 0.5 else 1.0 + w_norm
    x = x * scale
    W = w_proj if w_proj.shape == (1152, 2560) else w_proj.T
    y = x @ W.astype(np.float64)
    got = np.fromfile(rows_bin, dtype=np.float32).reshape(256, 2560).astype(np.float64)
    cos = (y * got).sum(-1) / np.linalg.norm(y, axis=-1) / np.linalg.norm(got, axis=-1)
    err = np.abs(y - got).max() / np.abs(y).max()
    i = int(cos.argmin())
    print(f"port vs numpy on the same file: worst per-row cosine {cos.min():.8f} (row {i}), max error {err:.2e} of max")
    return 0 if cos.min() >= 0.99999 else 1


if __name__ == "__main__":
    sys.exit(main())
