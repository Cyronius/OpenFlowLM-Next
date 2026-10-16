# Traces: OPEN-FAMILY-GEMMA3, OPEN-PACK-PAD, OPEN-SPEC-DERIVE (canonical spec: specs/open-engine/spec.md)
"""Gemma 3 1B: the spec its GGUF derives, and the zero-padded layout the dense recipe builds
it at (hidden 1152 is 4.5 k-tiles of 256, so the pools and kernels run at 1280)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from recipes import dense as DR
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes.spec import DENSE, ModelSpec, _f32_decimal

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"

# What google/gemma-3-1b-it-qat-q4_0-gguf's gemma-3-1b-it-q4_0.gguf carries, as read_gguf returns it.
GGUF_GEMMA3_1B = {
    "general.architecture": "gemma3", "gemma3.context_length": 32768, "gemma3.block_count": 26,
    "gemma3.embedding_length": 1152, "gemma3.feed_forward_length": 6912, "gemma3.attention.head_count": 4,
    "gemma3.attention.head_count_kv": 1, "gemma3.attention.key_length": 256, "gemma3.attention.value_length": 256,
    "gemma3.attention.layer_norm_rms_epsilon": 9.999999974752427e-07, "gemma3.rope.scaling.type": "linear",
    "gemma3.rope.scaling.factor": 1.0, "gemma3.rope.freq_base": 1000000.0, "gemma3.attention.sliding_window": 1024,
    "tokenizer.ggml.tokens": [""] * 262144,
}
# google/gemma-3-1b-it-qat-q4_0-unquantized's config.json, the checkpoint the GGUF was made from.
HF_GEMMA3_1B = {
    "model_type": "gemma3_text", "hidden_size": 1152, "intermediate_size": 6912, "num_hidden_layers": 26,
    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256, "rope_theta": 1000000,
    "rope_local_base_freq": 10000, "rope_scaling": None, "sliding_window": 512, "sliding_window_pattern": 6,
    "vocab_size": 262144, "query_pre_attn_scalar": 256, "hidden_activation": "gelu_pytorch_tanh",
    "final_logit_softcapping": None, "attn_logit_softcapping": None, "rms_norm_eps": 1e-06,
}


@pytest.fixture
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def spec_1b():
    return load_spec(SPECS / "gemma3-1b.json")


def test_gguf_and_config_agree_but_for_the_window():
    g = ModelSpec.from_gguf_metadata(GGUF_GEMMA3_1B)
    h = ModelSpec.from_hf_config(HF_GEMMA3_1B, real_vocab=262145)   # tokenizer.json's 262144 + 1
    assert (g.hidden, g.intermediate, g.num_layers, g.num_heads, g.num_kv_heads, g.head_dim) == (1152, 6912, 26, 4, 1, 256)
    assert [i for i, t in enumerate(g.layer_types) if t == DENSE] == [5, 11, 17, 23]
    assert g.norm_eps == 1e-06 and g.rope_scaling is None, "an f32 1e-6 and a linear factor of 1 are the config's own"
    assert h.real_vocab == h.vocab == 262144, "the image token past the head's rows is not a logit"
    assert (g.sliding_window, h.sliding_window) == (1024, 512)
    dg, dh = g.to_dict(), h.to_dict()
    for d in (dg, dh):
        d.pop("extra")
        d.pop("sliding_window")
    assert dg == dh


def test_the_shipped_spec_is_the_checkpoints():
    s = spec_1b()
    assert s.spec_hash() == ModelSpec.from_hf_config(HF_GEMMA3_1B, real_vocab=262145).spec_hash()
    assert s.sliding_window == 512 and s.extra["model"] == "Gemma3-1B-NPU2"


@pytest.mark.parametrize("v, want", [(9.999999974752427e-07, 1e-06), (9.999999747378752e-06, 1e-05),
                                     (1000000.0, 1000000.0), (0.0625, 0.0625)])
def test_f32_metadata_reads_back_as_its_decimal(v, want):
    assert _f32_decimal(v) == want


def test_padded_widths():
    s = spec_1b()
    assert DR.cores_for(s) == 2, "unpadded, 1152 = 18 bands only splits over two cores"
    assert DR.widths(s) == (4, 1280, 6912) and DR.padded(s)


@pytest.mark.parametrize("path", sorted(p.name for p in SPECS.glob("*.json") if p.name != "gemma3-1b.json"))
def test_no_shipped_dense_spec_is_padded(path):
    s = load_spec(SPECS / path)
    if s.family not in DR.DENSE_FAMILIES:
        return
    assert DR.widths(s) == (DR.cores_for(s), s.hidden, s.intermediate) and not DR.padded(s)
    for q in (s, dataclasses.replace(s, quant="q4_1_f32")):
        plan = DR.pack_plan(q)
        ops = [op for d in plan["layer_types"].values() for op in d["pool"]] + plan["lm_head"]["ops"]
        assert not any("src_dim" in op or "src_rows" in op for op in ops)
    assert "LN_DIV" not in DR.builds(s)["ln"]["env"]


def test_the_gguf_ops_name_the_files_own_shape():
    s = dataclasses.replace(spec_1b(), quant="q4_1_f32")
    plan = DR.pack_plan(s)
    pool = {op["tensor"].split(".")[-2]: op for op in plan["layer_types"]["dense"]["pool"]}
    for name in ("q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"):
        assert (pool[name]["op"], pool[name]["in_dim"], pool[name]["src_dim"]) == ("std_perm_gguf", 1280, 1152)
        assert "src_rows" not in pool[name]
    for name in ("o_proj", "down_proj"):
        assert pool[name]["src_rows"] == 1152 and "src_dim" not in pool[name]
    assert pool["o_proj"]["in_dim"] == 1024 and pool["down_proj"]["in_dim"] == 6912
    assert pool["down_proj"]["nch"] == 1280 * 6912 // 8192, "the pool holds the padded rows"
    head = plan["lm_head"]["ops"][0]
    assert (head["in_dim"], head["src_dim"], head["nch"]) == (1280, 1152, 262144 * 1280 // 8192)
    assert plan["embed"] == {"tensor": "model.embed_tokens.weight", "dim": 1152, "scale": 1152 ** 0.5}
    assert plan["norm"]["bytes"] == 1280 * 2


def test_the_norms_divide_by_the_models_width():
    b = DR.builds(spec_1b())
    assert b["ln"]["env"] == {"LN_N": "1280", "LN_EPS": "1e-06", "LN_DIV": "1152"}
    assert b["lm_head_q4_f32"]["env"]["LMHEAD_K"] == "1280" and b["lm_head_q4_f32"]["env"]["LMHEAD_CORES"] == "4"


def test_the_manifest(unvalidated):
    m = manifest(spec_1b())
    g = m["gguf"]
    assert m["layout"]["hidden"] == g["layout"]["hidden"] == 1280
    assert m["hf_config_check"]["hidden_size"] == 1152 and m["hf_config_check"]["sliding_window"] == 512
    assert g["globals"]["xres"] == 1280 * 4 and g["globals"]["hn"] == 1280 * 2
    assert all("gemm_block" not in lt for lt in g["layer_types"].values())
    assert sorted(g["kernels"]) == ["dx", "dx_local", "lm", "ln"]
    assert g["kernels"]["dx_local"]["window"] == 512 and g["kernels"]["dx"]["window"] == 0
    json.dumps(m)


def test_a_q4nx_container_cannot_take_the_padded_pool():
    plan = DR.pack_plan(spec_1b())
    assert all(op["op"] == "std_perm" for op in plan["layer_types"]["dense"]["pool"])
    assert any("src_dim" in op for op in plan["layer_types"]["dense"]["pool"]), \
        "the q4nx layout names its padding so pools.cpp std_perm can refuse it by name"
