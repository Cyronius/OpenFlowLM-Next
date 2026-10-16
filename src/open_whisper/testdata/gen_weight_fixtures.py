"""weight_source_test fixtures: a tiny Whisper as checkpoint and GGUFs, hashed through q4nx-build's converter (2026-10-16)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFWriter, dequantize, quantize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "utilities" / "q4nx-build"))
from q4nx.open_whisper import whisper_tensors, write_safetensors  # noqa: E402

# Distinct sizes, so a reader that swaps two dimensions fails on shape before it fails on value.
D, N_ENC, N_DEC, HEADS, FFN, MEL, SRC, TGT, VOCAB = 32, 1, 2, 2, 64, 2, 2, 3, 34
SHAPE = {"d_model": D, "n_enc": N_ENC, "n_dec": N_DEC, "n_heads": HEADS, "ffn": FFN,
         "n_mel": MEL, "max_src": SRC, "max_tgt": TGT, "vocab": VOCAB}

# The GGUF names, written out from handy-computer/whisper-large-v3-turbo-gguf's own tensor list.
TOP = {"model.encoder.conv1": "enc.conv.0", "model.encoder.conv2": "enc.conv.1",
       "model.encoder.embed_positions": "enc.pos_emb", "model.encoder.layer_norm": "enc.final_norm",
       "model.decoder.embed_tokens": "dec.token_embd", "model.decoder.embed_positions": "dec.pos_emb",
       "model.decoder.layer_norm": "dec.final_norm"}
ENC = {"self_attn.q_proj": "attn.q", "self_attn.k_proj": "attn.k", "self_attn.v_proj": "attn.v",
       "self_attn.out_proj": "attn.out", "self_attn_layer_norm": "norm_attn",
       "final_layer_norm": "norm_ffn", "fc1": "ffn.fc1", "fc2": "ffn.fc2"}
DEC = {"self_attn.q_proj": "self_attn.q", "self_attn.k_proj": "self_attn.k",
       "self_attn.v_proj": "self_attn.v", "self_attn.out_proj": "self_attn.out",
       "encoder_attn.q_proj": "cross_attn.q", "encoder_attn.k_proj": "cross_attn.k",
       "encoder_attn.v_proj": "cross_attn.v", "encoder_attn.out_proj": "cross_attn.out",
       "self_attn_layer_norm": "norm_self", "encoder_attn_layer_norm": "norm_cross",
       "final_layer_norm": "norm_ffn", "fc1": "ffn.fc1", "fc2": "ffn.fc2"}


def gguf_name(hf: str) -> str:
    stem, _, kind = hf.rpartition(".")
    if stem in TOP:
        return f"{TOP[stem]}.{kind}"
    side, _, rest = stem.partition(".layers.")
    idx, _, layer = rest.partition(".")
    table, pre = (ENC, "enc") if side.endswith("encoder") else (DEC, "dec")
    return f"{pre}.blocks.{idx}.{table[layer]}.{kind}"


def checkpoint(rng) -> dict:
    """Normal draws, so the bf16 rounding (ties included) is exercised."""
    t = {}
    f16 = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float16)  # noqa: E731
    e, d = "model.encoder.", "model.decoder."
    t[e + "conv1.weight"], t[e + "conv1.bias"] = f16(D, MEL, 3), f16(D)
    t[e + "conv2.weight"], t[e + "conv2.bias"] = f16(D, D, 3), f16(D)
    t[e + "embed_positions.weight"] = f16(SRC, D)
    t[e + "layer_norm.weight"], t[e + "layer_norm.bias"] = f16(D), f16(D)
    t[d + "embed_tokens.weight"], t[d + "embed_positions.weight"] = f16(VOCAB, D), f16(TGT, D)
    t[d + "layer_norm.weight"], t[d + "layer_norm.bias"] = f16(D), f16(D)

    def block(p, attns, norms):
        for a in attns:
            for n in ("q", "k", "v", "out"):
                t[f"{p}{a}.{n}_proj.weight"] = f16(D, D)
                if n != "k":
                    t[f"{p}{a}.{n}_proj.bias"] = f16(D)
        for n in norms:
            t[f"{p}{n}.weight"], t[f"{p}{n}.bias"] = f16(D), f16(D)
        t[p + "fc1.weight"], t[p + "fc1.bias"] = f16(FFN, D), f16(FFN)
        t[p + "fc2.weight"], t[p + "fc2.bias"] = f16(D, FFN), f16(D)

    for i in range(N_ENC):
        block(f"{e}layers.{i}.", ["self_attn"], ["self_attn_layer_norm", "final_layer_norm"])
    for i in range(N_DEC):
        block(f"{d}layers.{i}.", ["self_attn", "encoder_attn"],
              ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"])
    return t


def is_matmul(hf: str, a: np.ndarray) -> bool:
    return a.ndim == 2 and "embed_positions" not in hf


def write_gguf(path: Path, ckpt: dict, quant: str) -> None:
    """handy-computer's type layout: conv kernels are F32 beside Q8_0, F16 beside F16."""
    w = GGUFWriter(str(path), "whisper")
    for key, v in (("encoder.n_layers", N_ENC), ("decoder.n_layers", N_DEC), ("encoder.d_model", D),
                   ("decoder.d_model", D), ("encoder.n_heads", HEADS), ("decoder.n_heads", HEADS),
                   ("encoder.ffn_dim", FFN), ("decoder.ffn_dim", FFN), ("encoder.num_mel_bins", MEL),
                   ("encoder.max_source_positions", SRC), ("decoder.max_target_positions", TGT),
                   ("decoder.vocab_size", VOCAB)):
        w.add_uint32(f"stt.whisper.{key}", v)
    w.add_bool("stt.whisper.decoder.scale_embedding", False)
    w.add_bool("stt.whisper.decoder.tie_word_embeddings", True)
    for hf, a in sorted(ckpt.items()):
        name = gguf_name(hf)
        if is_matmul(hf, a) and quant == "Q8_0":
            w.add_tensor(name, quantize(a.astype(np.float32), Q.Q8_0), raw_dtype=Q.Q8_0)
        elif is_matmul(hf, a) or (a.ndim == 3 and quant == "F16"):
            w.add_tensor(name, a)
        else:
            w.add_tensor(name, a.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class Source:

    def __init__(self, tensors: dict):
        self.tensors = tensors
        self.meta = {k: None for k in tensors}

    def __call__(self, name: str) -> np.ndarray:
        return self.tensors[name].astype(np.float32)


def gguf_source(path: Path, ckpt: dict) -> Source:
    by_name = {t.name: t for t in GGUFReader(str(path)).tensors}
    out = {}
    for hf, a in ckpt.items():
        t = by_name[gguf_name(hf)]
        v = np.asarray(t.data) if t.tensor_type in (Q.F32, Q.F16) else dequantize(t.data, t.tensor_type)
        out[hf] = np.asarray(v, dtype=np.float32).reshape(a.shape)
    return Source(out)


def fnv1a64(b: bytes) -> str:
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def expected(src: Source) -> dict:
    out = {}
    for name, (dtype, a) in whisper_tensors(src, N_ENC, N_DEC, D).items():
        out[name] = {"dtype": dtype, "shape": list(a.shape), "fnv1a64": fnv1a64(np.ascontiguousarray(a).tobytes())}
    return out


def kquant_blocks(out: Path, rng) -> None:
    w = GGUFWriter(str(out / "kquants.gguf"), "whisper")
    values = []
    for name, qt in (("q4_k", Q.Q4_K), ("q5_k", Q.Q5_K), ("q6_k", Q.Q6_K)):
        from gguf.constants import GGML_QUANT_SIZES
        block, nbytes = GGML_QUANT_SIZES[qt]
        raw = rng.integers(0, 256, size=(2, nbytes), dtype=np.uint8)
        scale = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
        if qt == Q.Q6_K:
            raw[:, nbytes - 2:] = scale                       # d sits last
        else:
            raw[:, 0:2] = scale                               # d, then dmin
            raw[:, 2:4] = np.frombuffer(np.float16(0.003).tobytes(), dtype=np.uint8)
        w.add_tensor(name, raw, raw_dtype=qt)
        values.append(dequantize(raw, qt).astype("<f4").reshape(-1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    np.concatenate(values).tofile(out / "kquants_expected.bin")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "weights_tiny"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    ckpt = checkpoint(rng)
    write_safetensors(out / "model.safetensors", {k: ("F16", v) for k, v in ckpt.items()})
    write_gguf(out / "model-F16.gguf", ckpt, "F16")
    write_gguf(out / "model-Q8_0.gguf", ckpt, "Q8_0")
    oracle = {"shape": SHAPE,
              "f16": expected(Source(ckpt)),
              "q8_0": expected(gguf_source(out / "model-Q8_0.gguf", ckpt))}
    if expected(gguf_source(out / "model-F16.gguf", ckpt)) != oracle["f16"]:
        raise SystemExit("the F16 GGUF does not hold the checkpoint's values")
    (out / "expected.json").write_text(json.dumps(oracle, indent=1, sort_keys=True) + "\n")
    kquant_blocks(out, rng)
    print(f"wrote {sorted(p.name for p in out.iterdir())} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
