"""Tests for --open-whisper (issue #72): the open Whisper engine's container.

A tiny synthetic checkpoint with Whisper's tensor names stands in for the real 1.6 GB one,
so this runs anywhere. Weights are drawn from values bf16 represents exactly, which makes
every layout check an equality rather than a tolerance.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.open_causal import safetensors_header  # noqa: E402
from q4nx.open_whisper import (  # noqa: E402
    GEOMETRY,
    WEIGHTS_NAME,
    bf16_bits,
    build_open_whisper_repo,
    gguf_name_map,
)

D, FFN, MELS, NENC, NDEC, VOCAB, SRC, TGT = 32, 64, 4, 2, 2, 40, 6, 5
TINY = {"d_model": D, "encoder_layers": NENC, "decoder_layers": NDEC,
        "encoder_ffn_dim": FFN, "decoder_ffn_dim": FFN, "num_mel_bins": MELS,
        "max_source_positions": SRC, "max_target_positions": TGT, "vocab_size": VOCAB}


def _exact(rng, *shape):
    """fp16 values with the low 3 mantissa bits clear: exact in bf16 and in fp16."""
    return (rng.integers(-64, 64, size=shape) / 32.0).astype(np.float16)


def _checkpoint(root: Path, seed: int = 0, mat=_exact) -> dict:
    """`mat` draws the matmul weights and the embedding, the tensors a GGUF quantizes."""
    from safetensors.numpy import save_file
    rng = np.random.default_rng(seed)
    t = {}
    e, d = "model.encoder.", "model.decoder."
    t[e + "conv1.weight"], t[e + "conv1.bias"] = _exact(rng, D, MELS, 3), _exact(rng, D)
    t[e + "conv2.weight"], t[e + "conv2.bias"] = _exact(rng, D, D, 3), _exact(rng, D)
    t[e + "embed_positions.weight"] = _exact(rng, SRC, D)
    t[e + "layer_norm.weight"], t[e + "layer_norm.bias"] = _exact(rng, D), _exact(rng, D)
    for i in range(NENC):
        p = f"{e}layers.{i}."
        for n in "qkv":
            t[p + f"self_attn.{n}_proj.weight"] = mat(rng, D, D)
        t[p + "self_attn.q_proj.bias"] = _exact(rng, D)
        t[p + "self_attn.v_proj.bias"] = _exact(rng, D)
        t[p + "self_attn.out_proj.weight"], t[p + "self_attn.out_proj.bias"] = mat(rng, D, D), _exact(rng, D)
        t[p + "fc1.weight"], t[p + "fc1.bias"] = mat(rng, FFN, D), _exact(rng, FFN)
        t[p + "fc2.weight"], t[p + "fc2.bias"] = mat(rng, D, FFN), _exact(rng, D)
        for n in ("self_attn_layer_norm", "final_layer_norm"):
            t[p + n + ".weight"], t[p + n + ".bias"] = _exact(rng, D), _exact(rng, D)
    t[d + "embed_tokens.weight"] = mat(rng, VOCAB, D)
    t[d + "embed_positions.weight"] = _exact(rng, TGT, D)
    t[d + "layer_norm.weight"], t[d + "layer_norm.bias"] = _exact(rng, D), _exact(rng, D)
    for l in range(NDEC):
        p = f"{d}layers.{l}."
        for a in ("self_attn", "encoder_attn"):
            for n in "qkvo":
                name = "out_proj" if n == "o" else f"{n}_proj"
                t[p + f"{a}.{name}.weight"] = mat(rng, D, D)
                if n != "k":
                    t[p + f"{a}.{name}.bias"] = _exact(rng, D)
        t[p + "fc1.weight"], t[p + "fc1.bias"] = mat(rng, FFN, D), _exact(rng, FFN)
        t[p + "fc2.weight"], t[p + "fc2.bias"] = mat(rng, D, FFN), _exact(rng, D)
        for n in ("self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"):
            t[p + n + ".weight"], t[p + n + ".bias"] = _exact(rng, D), _exact(rng, D)
    save_file(t, str(root / "model.safetensors"))
    cfg = dict(TINY, model_type="whisper", bos_token_id=50257, eos_token_id=50257,
               encoder_attention_heads=2, decoder_attention_heads=2)
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<|endoftext|>"}))
    (root / "generation_config.json").write_text(json.dumps({
        "decoder_start_token_id": 50258, "eos_token_id": 50257,
        "no_timestamps_token_id": 50363, "max_length": 448}))
    return {k: v.astype(np.float32) for k, v in t.items()}


def _read(out: Path) -> dict:
    """The container back as fp32, decoding bf16 by hand (numpy has no bf16)."""
    path = out / WEIGHTS_NAME
    res = {}
    for name, m in safetensors_header(path).items():
        n = int(np.prod(m["shape"]))
        if m["dtype"] == "BF16":
            raw = np.fromfile(path, np.uint16, n, offset=m["offset"])
            a = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            a = np.fromfile(path, "<f4", n, offset=m["offset"])
        res[name] = (m["dtype"], a.reshape(m["shape"]))
    return res


class OpenWhisperContainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "src").mkdir()
        cls.src = _checkpoint(root / "src")
        cls.out = root / "out"
        build_open_whisper_repo(str(root / "src"), str(cls.out), geometry=TINY)
        cls.c = _read(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fused_qkv_is_wT_with_zero_k_bias(self):
        s = self.src
        for i in range(NENC):
            p = f"model.encoder.layers.{i}.self_attn."
            dt, b = self.c[f"enc.{i}.qkv.B"]
            self.assertEqual(dt, "BF16")
            want = np.concatenate([s[p + f"{n}_proj.weight"] for n in "qkv"]).T
            np.testing.assert_array_equal(b, want)
            _, bias = self.c[f"enc.{i}.qkv.bias"]
            np.testing.assert_array_equal(bias[:D], s[p + "q_proj.bias"])
            np.testing.assert_array_equal(bias[D:2 * D], 0)
            np.testing.assert_array_equal(bias[2 * D:], s[p + "v_proj.bias"])

    def test_cross_kv_order_and_bias(self):
        _, b = self.c["dec.xkv.B"]
        _, bias = self.c["dec.xkv.bias"]
        self.assertEqual(b.shape, (D, 2 * NDEC * D))
        for l in range(NDEC):
            p = f"model.decoder.layers.{l}.encoder_attn."
            np.testing.assert_array_equal(b[:, 2 * l * D:(2 * l + 1) * D], self.src[p + "k_proj.weight"].T)
            np.testing.assert_array_equal(b[:, (2 * l + 1) * D:(2 * l + 2) * D], self.src[p + "v_proj.weight"].T)
            np.testing.assert_array_equal(bias[2 * l * D:(2 * l + 1) * D], 0)
            np.testing.assert_array_equal(bias[(2 * l + 1) * D:(2 * l + 2) * D], self.src[p + "v_proj.bias"])
        self.assertFalse(any(".encoder_attn.k_proj." in n or ".encoder_attn.v_proj." in n for n in self.c),
                         "cross K/V must live only in dec.xkv")

    def test_conv_im2col_operand_computes_the_convolution(self):
        """im2col GEMM with the stored operand == a direct conv1d (pad 1), both convs."""
        rng = np.random.default_rng(1)
        for name, cin, stride in (("conv1", MELS, 1), ("conv2", D, 2)):
            w = self.src[f"model.encoder.{name}.weight"]          # [D, cin, 3]
            _, B = self.c[f"enc.{name}.B"]
            T = 8
            x = rng.standard_normal((T, cin)).astype(np.float32)   # time-major
            xp = np.pad(x, ((1, 1), (0, 0)))
            t_out = (T - 1) // stride + 1
            direct = np.stack([sum(w[:, :, k] @ xp[t * stride + k] for k in range(3))
                               for t in range(t_out)])
            idx = np.arange(t_out) * stride
            a = np.concatenate([xp[idx], xp[idx + 1], xp[idx + 2]], axis=1)
            np.testing.assert_allclose(a @ B, direct, rtol=1e-5, atol=1e-5)

    def test_dtypes(self):
        for name, (dt, a) in self.c.items():
            vector = a.ndim == 1 or name.endswith(("embed_positions.weight", "enc.pos"))
            self.assertEqual(dt, "F32" if vector else "BF16", name)

    def test_tokenizer_config_gets_the_ids_the_host_reads(self):
        cfg = json.loads((self.out / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["bos_token_id"], 50257)
        self.assertEqual(cfg["eos_token_id"], [50257])
        self.assertEqual(cfg["bos_token"], "<|endoftext|>")

    def test_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as t2:
            build_open_whisper_repo(str(Path(self.tmp.name) / "src"), t2, geometry=TINY)
            for f in sorted(p.name for p in self.out.iterdir()):
                self.assertEqual((self.out / f).read_bytes(), (Path(t2) / f).read_bytes(), f)

    def test_refuses_another_geometry(self):
        with tempfile.TemporaryDirectory() as t2, self.assertRaises(ValueError):
            build_open_whisper_repo(str(Path(self.tmp.name) / "src"), t2)   # the real GEOMETRY
        self.assertEqual(GEOMETRY["d_model"], 1280)

    def test_ships_generation_config_json(self):
        # Required, not optional (task 0180 Part 12): the hf decode protocol, now this
        # engine's own default, reads it at load time and refuses without it.
        self.assertTrue((self.out / "generation_config.json").is_file())
        cfg = json.loads((self.out / "generation_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["decoder_start_token_id"], 50258)

    def test_refuses_a_source_with_no_generation_config_json(self):
        with tempfile.TemporaryDirectory() as root:
            src = Path(root) / "src"
            shutil.copytree(Path(self.tmp.name) / "src", src)
            (src / "generation_config.json").unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                build_open_whisper_repo(str(src), str(Path(root) / "out"), geometry=TINY)
            self.assertIn("generation_config.json", str(ctx.exception))

    def test_bf16_rounds_to_nearest_even(self):
        x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5], np.float32)
        # 1+2^-8 is a tie between 1 and 1+2^-7: even mantissa wins (1.0).
        # 1+3*2^-8 ties between 1+2^-7 and 1+2^-6: even is 1+2^-6.
        np.testing.assert_array_equal(bf16_bits(x), [0x3F80, 0x3F80, 0x3F82, 0xC020])


def _q8_exact(rng, *shape):
    """Values Q8_0 stores and bf16 holds exactly: integers in [-127, 127] over 64, with a
    127 first in every block of 32 so the block scale is exactly 1/64."""
    q = rng.integers(-126, 127, size=shape)
    q[..., ::32] = 127
    return (q / 64.0).astype(np.float16)


VOCAB_TOKENS = [f"t{i}" for i in range(VOCAB - 3)] + ["<|endoftext|>", "<|startoftranscript|>", "<|nots|>"]
MERGES = ["t1 t2", "t3 t4"]


def _write_sidecars(root: Path) -> None:
    """tokenizer.json and generation_config.json that agree with the tiny GGUF below."""
    vocab = {t: i for i, t in enumerate(VOCAB_TOKENS[:-3])}
    added = [{"id": VOCAB - 3 + k, "content": c} for k, c in enumerate(VOCAB_TOKENS[-3:])]
    (root / "tokenizer.json").write_text(json.dumps(
        {"model": {"vocab": vocab, "merges": MERGES}, "added_tokens": added}))
    gen = json.loads((root / "generation_config.json").read_text())
    gen["suppress_tokens"] = [1, 2]
    (root / "generation_config.json").write_text(json.dumps(gen))


# Written out by hand from a real turbo GGUF, so the production name map is not checked
# against itself: a round trip through one map is blind to a wrong entry in it.
_ENC_BLOCK = {"self_attn.q_proj": "attn.q", "self_attn.k_proj": "attn.k", "self_attn.v_proj": "attn.v",
              "self_attn.out_proj": "attn.out", "self_attn_layer_norm": "norm_attn",
              "final_layer_norm": "norm_ffn", "fc1": "ffn.fc1", "fc2": "ffn.fc2"}
_DEC_BLOCK = {"self_attn.q_proj": "self_attn.q", "self_attn.k_proj": "self_attn.k",
              "self_attn.v_proj": "self_attn.v", "self_attn.out_proj": "self_attn.out",
              "encoder_attn.q_proj": "cross_attn.q", "encoder_attn.k_proj": "cross_attn.k",
              "encoder_attn.v_proj": "cross_attn.v", "encoder_attn.out_proj": "cross_attn.out",
              "self_attn_layer_norm": "norm_self", "encoder_attn_layer_norm": "norm_cross",
              "final_layer_norm": "norm_ffn", "fc1": "ffn.fc1", "fc2": "ffn.fc2"}
_TOP = {"model.encoder.conv1": "enc.conv.0", "model.encoder.conv2": "enc.conv.1",
        "model.encoder.embed_positions": "enc.pos_emb", "model.encoder.layer_norm": "enc.final_norm",
        "model.decoder.embed_tokens": "dec.token_embd", "model.decoder.embed_positions": "dec.pos_emb",
        "model.decoder.layer_norm": "dec.final_norm"}


def _gguf_name(hf: str) -> str:
    stem, _, kind = hf.rpartition(".")
    if stem in _TOP:
        return f"{_TOP[stem]}.{kind}"
    side, _, rest = stem.partition(".layers.")
    idx, _, layer = rest.partition(".")
    table, pre = (_ENC_BLOCK, "enc") if side.endswith("encoder") else (_DEC_BLOCK, "dec")
    return f"{pre}.blocks.{idx}.{table[layer]}.{kind}"


def _write_gguf(path: Path, src: dict, quant: str, **override) -> None:
    from gguf import GGMLQuantizationType as Q
    from gguf import GGUFWriter, quantize
    w = GGUFWriter(str(path), "whisper")
    ints = {"stt.whisper.encoder.n_layers": NENC, "stt.whisper.decoder.n_layers": NDEC,
            "stt.whisper.encoder.d_model": D, "stt.whisper.encoder.ffn_dim": FFN,
            "stt.whisper.encoder.num_mel_bins": MELS, "stt.whisper.encoder.max_source_positions": SRC,
            "stt.whisper.decoder.max_target_positions": TGT, "stt.whisper.decoder.vocab_size": VOCAB,
            "stt.whisper.decoder_start_token_id": 50258, "stt.whisper.no_timestamps_token_id": 50363,
            "tokenizer.ggml.eos_token_id": 50257} | override
    for k, v in ints.items():
        w.add_uint32(k, v)
    w.add_array("stt.whisper.suppress_tokens", [1, 2])
    w.add_array("tokenizer.ggml.tokens", VOCAB_TOKENS)
    w.add_array("tokenizer.ggml.merges", MERGES)
    for hf, a in src.items():
        gg = _gguf_name(hf)
        if a.ndim == 2 and "embed_positions" not in hf and "layer_norm" not in hf:
            if quant == "Q8_0":
                w.add_tensor(gg, quantize(a, Q.Q8_0), raw_dtype=Q.Q8_0)
            else:
                w.add_tensor(gg, a.astype(np.float16))
        else:
            w.add_tensor(gg, a.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class GgufInputTest(unittest.TestCase):
    """A GGUF of the same weights builds the same container as the HuggingFace checkpoint."""
    # Traces: OPEN-WHISPER-GGUF

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "src").mkdir()
        cls.src = _checkpoint(root / "src", mat=_q8_exact)
        _write_sidecars(root / "src")
        build_open_whisper_repo(str(root / "src"), str(root / "from_hf"), geometry=TINY)
        for quant in ("F16", "Q8_0"):
            _write_gguf(root / f"m-{quant}.gguf", cls.src, quant)
            build_open_whisper_repo(str(root / f"m-{quant}.gguf"), str(root / f"from_{quant}"),
                                    geometry=TINY, sidecars=str(root / "src"))
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tensors_match_the_checkpoint_build_exactly(self):
        want = (self.root / "from_hf" / WEIGHTS_NAME).read_bytes()
        for quant in ("F16", "Q8_0"):
            got = (self.root / f"from_{quant}" / WEIGHTS_NAME).read_bytes()
            self.assertEqual(got, want, quant)

    def test_manifest_records_the_gguf_and_its_tensor_types(self):
        m = json.loads((self.root / "from_Q8_0" / "weights_manifest.json").read_text())
        self.assertEqual(m["source"]["file"], "m-Q8_0.gguf")
        self.assertEqual(m["source"]["format"], "gguf")
        self.assertIn("Q8_0", m["source"]["tensor_types"])

    def test_production_name_map_equals_the_hand_written_one(self):
        mapped = gguf_name_map(NENC, NDEC)
        self.assertEqual(mapped, {hf: _gguf_name(hf) for hf in mapped})
        self.assertEqual(set(mapped), set(self.src))

    def test_name_map_matches_a_real_turbo_gguf(self):
        real = json.loads((Path(__file__).parent / "data" / "whisper_turbo_gguf_tensors.json").read_text())
        mapped = gguf_name_map(GEOMETRY["encoder_layers"], GEOMETRY["decoder_layers"])
        self.assertEqual(set(mapped.values()), {n for n in real if not n.startswith("frontend.")})
        d, ffn, mels = GEOMETRY["d_model"], GEOMETRY["encoder_ffn_dim"], GEOMETRY["num_mel_bins"]
        want = {"model.encoder.conv1.weight": [d, mels, 3], "model.encoder.conv2.weight": [d, d, 3],
                "model.encoder.embed_positions.weight": [GEOMETRY["max_source_positions"], d],
                "model.decoder.embed_tokens.weight": [GEOMETRY["vocab_size"], d],
                "model.decoder.layers.0.fc1.weight": [ffn, d],
                "model.decoder.layers.3.encoder_attn.v_proj.weight": [d, d]}
        for hf, shape in want.items():
            self.assertEqual(real[mapped[hf]], shape, hf)

    def test_refuses_a_vocabulary_that_differs_from_the_sidecars(self):
        bad = Path(self.tmp.name) / "bad_sidecars"
        shutil.copytree(self.root / "src", bad)
        tok = json.loads((bad / "tokenizer.json").read_text())
        tok["model"]["merges"] = ["t1 t3"]
        (bad / "tokenizer.json").write_text(json.dumps(tok))
        with tempfile.TemporaryDirectory() as t2, self.assertRaises(ValueError) as ctx:
            build_open_whisper_repo(str(self.root / "m-F16.gguf"), t2, geometry=TINY, sidecars=str(bad))
        self.assertIn("merges", str(ctx.exception))

    def test_refuses_special_token_ids_that_differ(self):
        _write_gguf(self.root / "bad.gguf", self.src, "F16", **{"stt.whisper.decoder_start_token_id": 50259})
        with tempfile.TemporaryDirectory() as t2, self.assertRaises(ValueError) as ctx:
            build_open_whisper_repo(str(self.root / "bad.gguf"), t2, geometry=TINY,
                                    sidecars=str(self.root / "src"))
        self.assertIn("decoder_start_token_id", str(ctx.exception))

    def test_refuses_a_gguf_of_another_architecture(self):
        from gguf import GGUFWriter
        w = GGUFWriter(str(self.root / "llama.gguf"), "llama")
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        with tempfile.TemporaryDirectory() as t2, self.assertRaises(ValueError) as ctx:
            build_open_whisper_repo(str(self.root / "llama.gguf"), t2, geometry=TINY,
                                    sidecars=str(self.root / "src"))
        self.assertIn("llama", str(ctx.exception))

    def test_sidecars_without_a_gguf_is_an_error(self):
        with tempfile.TemporaryDirectory() as t2, self.assertRaises(ValueError):
            build_open_whisper_repo(str(self.root / "src"), t2, geometry=TINY, sidecars="x")


if __name__ == "__main__":
    unittest.main()
