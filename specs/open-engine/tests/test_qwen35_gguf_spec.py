"""OPEN-GGUF-QWEN35-SPEC: the spec a qwen35 GGUF derives, and the config fields the engine checks it by."""
# Traces: OPEN-GGUF-QWEN35-SPEC
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import qwen35 as Q35
from recipes.load import load_spec, narrow_to_buildable
from recipes.spec import ModelSpec, SpecError, gguf_real_vocab, quant_map_from_gguf_types

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"
# not in the default export (specs/pending/README.md): whether its set ships is the maintainer's call
MINICPM_SPEC = SPECS / "pending" / "minicpm-v-4.7-1b.json"

# openbmb/MiniCPM-V-4.7-1B-gguf, MiniCPM-V-4_7-1B-Q4_1.gguf, read by range request 2026-10-16
MINICPM_V_47_1B = {
    "general.architecture": "qwen35", "qwen35.block_count": 24, "qwen35.context_length": 262144,
    "qwen35.embedding_length": 1024, "qwen35.feed_forward_length": 3584,
    "qwen35.attention.head_count": 8, "qwen35.attention.head_count_kv": 2,
    "qwen35.rope.freq_base": 10000000.0, "qwen35.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
    "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
    "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.state_size": 128, "qwen35.ssm.group_count": 16,
    "qwen35.ssm.time_step_rank": 16, "qwen35.ssm.inner_size": 2048, "qwen35.full_attention_interval": 4,
    "qwen35.rope.dimension_count": 64,
    "tokenizer.ggml.tokens": [f"t{i}" for i in range(248094)],
}
MINICPM_TYPES = {f"blk.{l}.{r}": "Q4_1" for l in range(24)
                 for r in ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ffn_up.weight",
                           "ffn_gate.weight", "ffn_down.weight", "attn_q.weight", "attn_k.weight",
                           "attn_v.weight", "attn_output.weight")}
MINICPM_TYPES.update({"output.weight": "Q6_K", "token_embd.weight": "Q4_1"})


def test_minicpm_gguf_derives_the_checked_in_spec():
    s = ModelSpec.from_gguf_metadata(MINICPM_V_47_1B)
    assert (s.family, s.hidden, s.num_layers, s.intermediate) == ("qwen35", 1024, 24, 3584)
    assert (s.num_heads, s.num_kv_heads, s.head_dim, s.rotary_dim) == (8, 2, 256, 64)
    assert (s.lin_key_heads, s.lin_value_heads, s.lin_key_dim, s.lin_value_dim) == (16, 16, 128, 128)
    assert s.vocab == 248320 and s.real_vocab == 248094       # padded to 256 rows; the tokenizer's own ids
    assert s.norm_eps == 1e-6                                  # the f32 field read back as config.json says it
    assert quant_map_from_gguf_types("qwen35", MINICPM_TYPES) == {}
    want = load_spec(MINICPM_SPEC)
    assert s.spec_hash() == want.spec_hash()
    assert want.spec_hash() == "sha256:b6087971182e34fcaa1f8c0132409ef9f14f82224bde33ee44fa11d5da7e25d1"
    assert want.extra["model"] == "MiniCPM-V-4.7-1B-gguf"           # the directory oflm-add installs it as


def test_real_vocab_moves_the_hash_but_not_the_kernels():
    """The spec-hash finding: a spec that differs only in real_vocab builds the same kernels."""
    mini = load_spec(MINICPM_SPEC)
    base = ModelSpec.from_dict({**mini.to_dict(), "real_vocab": 248070})
    assert mini.to_dict().keys() == base.to_dict().keys()
    diff = {k for k in mini.to_dict() if k != "extra" and mini.to_dict()[k] != base.to_dict()[k]}
    assert diff == {"real_vocab"}
    assert mini.spec_hash() != base.spec_hash()
    assert Q35.builds(mini) == Q35.builds(base)


def test_head_pads_to_whole_256_rows():
    for rows, want in ((248094, 248320), (248144, 248320), (248320, 248320), (248321, 248576)):
        md = dict(MINICPM_V_47_1B, **{"tokenizer.ggml.tokens": [f"t{i}" for i in range(rows)]})
        assert ModelSpec.from_gguf_metadata(md).vocab == want


def test_llama_cpp_pad_tokens_are_not_vocabulary():
    toks = [f"t{i}" for i in range(248077)] + [f"[PAD{i}]" for i in range(248077, 248320)]
    assert gguf_real_vocab({"tokenizer.ggml.tokens": toks}, 248320) == 248077
    assert gguf_real_vocab({}, 1000) == 1000


def test_mtp_blocks_are_not_layers():
    md = dict(MINICPM_V_47_1B, **{"qwen35.block_count": 25, "qwen35.nextn_predict_layers": 1})
    s = ModelSpec.from_gguf_metadata(md)
    assert s.num_layers == 24 and len(s.layer_types) == 24


def test_a_q8_role_streams_at_q8_where_the_core_fits():
    types = dict(MINICPM_TYPES, **{f"blk.{l}.ssm_out.weight": "Q8_0" for l in range(24)})
    s = ModelSpec.from_gguf_metadata(MINICPM_V_47_1B)
    m = quant_map_from_gguf_types("qwen35", types)
    assert m == {"linear_out": "q8"}
    assert narrow_to_buildable(s, m, Q35.Q8_ROLES) == {"linear_out": "q8"}      # (qwen35, 1024) fits


def test_moe_metadata_is_refused():
    with pytest.raises(SpecError):
        ModelSpec.from_gguf_metadata(dict(MINICPM_V_47_1B, **{"qwen35.expert_count": 256}))


def test_engine_config_fields_match_the_recipe_check():
    """src/open_qwen36/gguf_view.cpp derive_config writes these for a qwen35 GGUF; gguf_pack_test
    asserts the same values on its synthetic GGUF (hidden 256, 4 blocks, 100 rows, 4 value heads)."""
    md = {"general.architecture": "qwen35", "qwen35.block_count": 4, "qwen35.embedding_length": 256,
          "qwen35.feed_forward_length": 256, "qwen35.attention.head_count": 4, "qwen35.attention.head_count_kv": 2,
          "qwen35.attention.key_length": 64, "qwen35.attention.value_length": 64, "qwen35.ssm.conv_kernel": 4,
          "qwen35.ssm.state_size": 64, "qwen35.ssm.group_count": 2, "qwen35.ssm.time_step_rank": 4,
          "qwen35.ssm.inner_size": 256, "qwen35.full_attention_interval": 4, "qwen35.rope.dimension_count": 16,
          "qwen35.rope.freq_base": 1e7, "tokenizer.ggml.tokens": [f"t{i}" for i in range(100)]}
    s = ModelSpec.from_gguf_metadata(md)
    check = Q35.hf_config_check(s)
    engine = {"model_type": "qwen3_5_text", "hidden_size": 256, "num_hidden_layers": 4, "vocab_size": 256,
              "intermediate_size": 256, "head_dim": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
              "linear_num_value_heads": 4,
              "layer_types": ["linear_attention"] * 3 + ["full_attention"]}
    assert {k: engine[k] for k in check} == check
    assert engine["model_type"] in ("qwen3_5", "qwen3_5_text")
    assert json.dumps(engine["layer_types"]) == json.dumps(list(s.layer_types))


def test_the_minicpm_set_is_not_in_the_default_export():
    import importlib.util
    path = SPECS.parents[2] / "utilities" / "export-kernels.py"
    spec = importlib.util.spec_from_file_location("export_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert MINICPM_SPEC not in mod.spec_list("")        # the default: what CMake and release staging build
    assert mod.spec_list("pending/minicpm-v-4.7-1b") == [MINICPM_SPEC]
