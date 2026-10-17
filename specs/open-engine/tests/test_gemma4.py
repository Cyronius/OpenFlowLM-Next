# Traces: OPEN-FAMILY-GEMMA4, OPEN-SPEC-DERIVE, OPEN-GGUF-BF16-POOL (canonical spec: specs/open-engine/spec.md)
"""Gemma 4 E2B / E4B / 12B (text) on the dense recipe: the spec from Google's GGUF and from transformers'
config, the KV-sharing layout, the tensor names the plan reads against the real file's, the layer kinds'
attention geometry, and the 12B's K = V global layers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import dense as DR
from recipes import gemma4 as G4
from recipes.manifest import manifest
from recipes.spec import DENSE, DENSE_LOCAL, DENSE_LOCAL_SHARED, DENSE_SHARED, ModelSpec, SpecError

FIX = Path(__file__).resolve().parent / "fixtures"
HDR = json.loads((FIX / "gguf_gemma4_e4b_header.json").read_text(encoding="utf-8"))
HDR_E2B = json.loads((FIX / "gguf_gemma4_e2b_header.json").read_text(encoding="utf-8"))
HDR_12B = json.loads((FIX / "gguf_gemma4_12b_header.json").read_text(encoding="utf-8"))
CFG_E4B = json.loads((FIX / "config_gemma4_e4b.json").read_text(encoding="utf-8"))
CFG_E2B = json.loads((FIX / "config_gemma4_e2b.json").read_text(encoding="utf-8"))
CFG_12B = json.loads((FIX / "config_gemma4_12b.json").read_text(encoding="utf-8"))
SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"


def gguf_md(with_rope_freqs=True, hdr=HDR):
    md = dict(hdr["metadata"])
    md["tokenizer.ggml.tokens"] = [""] * hdr["vocab"]
    if with_rope_freqs:
        md["gemma4.rope_freqs"] = hdr["rope_freqs"]
    return md


@pytest.fixture
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


@pytest.fixture
def e4b():
    return ModelSpec.from_gguf_metadata(gguf_md())


# ---- spec derivation
def test_gguf_and_hf_config_derive_the_same_spec(e4b):
    hf = ModelSpec.from_hf_config(CFG_E4B)
    assert e4b.spec_hash() == hf.spec_hash()


def test_e4b_shape(e4b):
    assert (e4b.hidden, e4b.num_layers, e4b.num_heads, e4b.num_kv_heads) == (2560, 42, 8, 2)
    assert (e4b.head_dim, e4b.global_head_dim, e4b.rotary_dim, e4b.global_rotary_dim) == (256, 512, 256, 128)
    assert (e4b.intermediate, e4b.intermediate_shared, e4b.ple_dim, e4b.logit_softcap) == (10240, 0, 256, 30.0)
    assert (e4b.rope_theta, e4b.rope_local_theta, e4b.sliding_window, e4b.norm_eps) == (1e6, 1e4, 512, 1e-6)


def test_the_global_rotation_is_read_from_rope_freqs():
    # llama.cpp writes transformers' partial_rotary_factor as 64 ones then 1e30 (pairs that never turn)
    assert sum(1 for f in HDR["rope_freqs"] if f == 1.0) == 64
    assert ModelSpec.from_gguf_metadata(gguf_md()).global_rotary_dim == 128
    assert ModelSpec.from_gguf_metadata(gguf_md(with_rope_freqs=False)).global_rotary_dim == 128


def test_layer_kinds_and_the_kv_shared_tail(e4b):
    lt = e4b.layer_types
    assert [i for i, t in enumerate(lt) if t in (DENSE, DENSE_SHARED)] == [5, 11, 17, 23, 29, 35, 41]
    assert all(t in (DENSE_LOCAL_SHARED, DENSE_SHARED) for t in lt[24:])
    assert not any(t in (DENSE_LOCAL_SHARED, DENSE_SHARED) for t in lt[:24])


def test_kv_sources_are_the_last_own_layer_of_each_kind(e4b):
    src = G4.kv_sources(e4b)
    assert src[:24] == [-1] * 24
    assert all(src[l] == (23 if e4b.layer_types[l] == DENSE_SHARED else 22) for l in range(24, 42))


def test_e2b_doubles_the_shared_layers_ffn_and_has_one_kv_head():
    s = ModelSpec.from_hf_config(CFG_E2B)
    assert (s.num_layers, s.num_kv_heads, s.intermediate, s.intermediate_shared) == (35, 1, 6144, 12288)
    assert G4.kv_sources(s)[15:] == [13 if t == DENSE_LOCAL_SHARED else 14 for t in s.layer_types[15:]]


def test_e2b_gguf_and_hf_config_derive_the_same_spec():
    # the GGUF lists feed_forward_length per layer (6144 x 15, then 12288 x 20)
    g = ModelSpec.from_gguf_metadata(gguf_md(hdr=HDR_E2B))
    assert g.spec_hash() == ModelSpec.from_hf_config(CFG_E2B).spec_hash()
    assert (g.hidden, g.global_head_dim, g.global_rotary_dim, g.ple_dim) == (1536, 512, 128, 256)


def test_the_shipped_e2b_spec_is_the_derived_one():
    from recipes.load import load_spec
    shipped = load_spec(Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" / "gemma4-e2b.json")
    assert shipped.spec_hash() == ModelSpec.from_gguf_metadata(gguf_md(hdr=HDR_E2B)).spec_hash()


@pytest.mark.parametrize("key,val", [("enable_moe_block", True), ("use_bidirectional_attention", "all")])
def test_the_moe_and_bidirectional_variants_are_refused(key, val):
    cfg = json.loads(json.dumps(CFG_E4B))
    cfg["text_config"][key] = val
    with pytest.raises(SpecError):
        ModelSpec.from_hf_config(cfg)


def test_vision_only_bidirectional_attention_leaves_text_causal():
    cfg = json.loads(json.dumps(CFG_E4B))
    cfg["text_config"]["use_bidirectional_attention"] = "vision"
    assert ModelSpec.from_hf_config(cfg).spec_hash() == ModelSpec.from_hf_config(CFG_E4B).spec_hash()


def test_no_other_spec_hash_moved_by_the_new_fields():
    # every new field serialises only when set, so a spec without them hashes as before
    d = ModelSpec.from_gguf_metadata(gguf_md()).to_dict()
    assert {"global_head_dim", "global_rotary_dim", "ple_dim", "logit_softcap"} <= set(d)
    assert not {"intermediate_shared", "global_kv_heads", "k_eq_v"} & set(d)
    from recipes.load import load_spec
    q = load_spec(Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" / "qwen3-4b.json")
    assert not {"global_head_dim", "global_rotary_dim", "intermediate_shared", "ple_dim", "logit_softcap",
                "global_kv_heads", "k_eq_v"} & set(q.to_dict())


# ---- the tensor names, against the real file's table
def plan_names(spec):
    p = G4.pack_plan(spec)
    by_kind = {k: [o["tensor"] for o in v["pool"] + v["consts"]] for k, v in p["layer_types"].items()}
    glob = [o["tensor"] for o in p["lm_head"]["ops"]] + [p["embed"]["tensor"], p["norm"]["tensor"]]
    hs = G4.host_stages(spec)
    if "ple" not in hs:                                   # the 12B
        return by_kind, glob, [hs["layer_scalar"]]
    st = hs["ple"]
    return by_kind, glob + [st["embed"], st["model_proj"], st["model_norm"]], \
        [st["gate"], st["proj"], st["norm"], hs["layer_scalar"]]


def test_every_tensor_the_plan_reads_is_in_the_file(unvalidated, e4b):
    have = set(HDR["tensors"])
    by_kind, glob, per_layer = plan_names(e4b)
    assert set(glob) <= have
    for l, kind in enumerate(e4b.layer_types):
        names = [n.replace("{l}", str(l)) for n in by_kind[kind] + per_layer]
        assert set(names) <= have, sorted(set(names) - have)


def test_the_shared_layers_read_no_k_or_v_and_the_file_has_none(unvalidated, e4b):
    have = set(HDR["tensors"])
    by_kind, _, _ = plan_names(e4b)
    for kind in (DENSE_LOCAL_SHARED, DENSE_SHARED):
        assert not any(t in n for n in by_kind[kind] for t in ("attn_k.", "attn_v.", "attn_k_norm"))
    for l in range(24, 42):
        assert not any(f"blk.{l}.{t}" in have for t in ("attn_k.weight", "attn_v.weight", "attn_k_norm.weight"))


def test_projection_widths_match_the_file(unvalidated, e4b):
    # GGUF dims are [in, out]: the plan's nch * 8192 values must be the tensor's whole size
    p = G4.pack_plan(e4b)
    for kind, l in ((DENSE_LOCAL, 0), (DENSE, 5), (DENSE_LOCAL_SHARED, 24), (DENSE_SHARED, 29)):
        for op in p["layer_types"][kind]["pool"]:
            dims, ty = HDR["tensors"][op["tensor"].replace("{l}", str(l))]
            assert ty == "Q4_0" and op["op"] == "std_perm_gguf"
            assert op["in_dim"] == dims[0] and op["nch"] * 8192 == dims[0] * dims[1]


# ---- the four kinds' attention geometry
@pytest.mark.parametrize("part,hd,rot,pair,cs_off,elems,shared", [
    ("local", 256, 256, 128, 1024, 2, False), ("global", 512, 128, 256, 512, 1, False),
    ("local_shared", 256, 256, 128, 1024, 2, True), ("global_shared", 512, 128, 256, 512, 1, True)])
def test_part_geometry(unvalidated, monkeypatch, e4b, part, hd, rot, pair, cs_off, elems, shared):
    monkeypatch.setenv("G4_PART", part)
    R = G4.recipe(e4b)
    G, L = R.geo, R.layout
    assert (G.HD, G.ROT, G.ROPE_PAIR, G.PTAB_CS_OFF, G.PTAB_ELEMS, G.KV_SHARED) == (hd, rot, pair, cs_off, elems, shared)
    assert G.SCALE_ONE and G.VNORM and G.VEXP and G.ACORES == 4 and G.RB == 2
    assert L.PTAB_ROW == 2048 and L.KV_ROW == 2 * 2 * hd * 2     # [K | V] rows of 2 kv heads, bf16
    if shared:                       # no k / v projection in the pool
        assert L.POOL_K == L.POOL_V == L.POOL_O


# ---- E2B: one kv head, and a twice-wider FFN in the KV-shared layers
@pytest.fixture
def e2b():
    return ModelSpec.from_gguf_metadata(gguf_md(hdr=HDR_E2B))


def test_e2b_plan_reads_tensors_the_file_has_at_their_widths(unvalidated, e2b):
    have = HDR_E2B["tensors"]
    by_kind, glob, per_layer = plan_names(e2b)
    assert set(glob) <= set(have)
    for l, kind in enumerate(e2b.layer_types):
        names = [n.replace("{l}", str(l)) for n in by_kind[kind] + per_layer]
        assert set(names) <= set(have), sorted(set(names) - set(have))
    p = G4.pack_plan(e2b)
    for kind, l in ((DENSE_LOCAL, 0), (DENSE, 4), (DENSE_LOCAL_SHARED, 15), (DENSE_SHARED, 19)):
        for op in p["layer_types"][kind]["pool"]:
            dims, ty = have[op["tensor"].replace("{l}", str(l))]
            assert ty == "Q4_0" and op["op"] == "std_perm_gguf"
            assert op["in_dim"] == dims[0] and op["nch"] * 8192 == dims[0] * dims[1]
    assert have["blk.15.ffn_up.weight"][0] == [1536, 12288] and have["blk.14.ffn_up.weight"][0] == [1536, 6144]
    assert not any(f"blk.{l}.attn_k.weight" in have for l in range(15, 35))


@pytest.mark.parametrize("part,hd,rot,ff,n,e_a,elems,shared", [
    ("local", 256, 256, 6144, 4, 1024, 2, False), ("global", 512, 128, 6144, 8, 2048, 1, False),
    ("local_shared", 256, 256, 12288, 4, 1024, 2, True), ("global_shared", 512, 128, 12288, 8, 2048, 1, True)])
def test_e2b_part_geometry(unvalidated, monkeypatch, e2b, part, hd, rot, ff, n, e_a, elems, shared):
    """One kv head: the element is the whole cache row; the 256-row kv projection splits over 4 cores."""
    monkeypatch.setenv("G4_PART", part)
    R = G4.recipe(e2b)
    G, L = R.geo, R.layout
    assert (G.HD, G.ROT, G.FF, G.N_CORES, G.KVH, G.KV_SHARED) == (hd, rot, ff, n, 1, shared)
    assert (L.E_A, L.KV_ROW, L.PTAB_ROW, G.PTAB_ELEMS, G.PTAB_CS_ELEM, G.PTAB_CS_OFF) == (e_a, e_a, 2048, elems, 1, 512)
    assert G.PER_CALL == 1


def test_only_e2b_caps_the_chunks_per_element(e4b, e2b):
    # the 12288-wide FFN table leaves room for one chunk per element; the generated TUs are shared by every kind
    assert G4.per_call_cap(e4b) is None
    assert G4.per_call_cap(e2b) == 1


def test_e2b_manifest_rows_and_sources(unvalidated, e2b):
    m = manifest(e2b)
    assert m["kv_source"][15:] == [13 if t == DENSE_LOCAL_SHARED else 14 for t in e2b.layer_types[15:]]
    assert {k: v["kv_row"] for k, v in m["kernels"].items() if "kv_row" in v} == {
        "dx_local": 1024, "dx": 2048, "dx_local_shared": 1024, "dx_shared": 2048}
    assert (m["globals"]["ptab_local"]["per_row"], m["globals"]["ptab_local"]["cs_off"]) == (2048, 512)


def test_other_families_keep_their_attention_defaults():
    from recipes.load import load_spec
    base = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"
    for name in ("gemma3-4b", "qwen3-4b", "qwen25-3b", "phi4-mini-4b"):
        G = DR.geometry(load_spec(base / f"{name}.json"))
        assert (G.SCALE_ONE, G.VNORM, G.ROPE_PAIR, G.PTAB_CS_OFF, G.KV_SHARED) == (False, False, G.ROT // 2, 512, False)


# ---- the manifest
def test_manifest_carries_the_host_stages_and_one_gguf_view(unvalidated, e4b):
    m = manifest(e4b)
    assert m["kv_source"] == G4.kv_sources(e4b) and m["logit_softcap"] == 30.0
    assert m["ple"]["dim"] == 256 and m["ple"]["embed_scale"] == 16.0
    assert m["gguf"]["kernels"] == m["kernels"] and m["gguf"]["pack"]["chunk_bytes"] == 5120
    assert {k: v["kv_row"] for k, v in m["kernels"].items() if "kv_row" in v} == {
        "dx_local": 2048, "dx": 4096, "dx_local_shared": 2048, "dx_shared": 4096}
    g = m["globals"]["ptab"]
    assert g["rotary_dim"] == 128 and len(g["inv_freq"]) == 64
    assert g["inv_freq"][1] == pytest.approx(1e6 ** (-2 / 512))       # proportional: over the head, not the rotation
    assert m["globals"]["ptab_local"]["cs_off"] == 1024


# ---- the 12B: no per-layer inputs, no KV sharing, one kv head and K = V on the global layers
def no_attn_v(hdr):
    """What a GGUF caller passes as gemma4.attention.k_eq_v: a global layer with its own cache has no attn_v."""
    pat = hdr["metadata"]["gemma4.attention.sliding_window_pattern"]
    g = pat.index(False)
    return f"blk.{g}.attn_k.weight" in hdr["tensors"] and f"blk.{g}.attn_v.weight" not in hdr["tensors"]


@pytest.fixture
def g12():
    md = gguf_md(hdr=HDR_12B)
    md["gemma4.attention.k_eq_v"] = no_attn_v(HDR_12B)
    return ModelSpec.from_gguf_metadata(md)


def test_12b_gguf_and_its_unified_config_derive_the_shipped_spec(g12):
    from recipes.load import load_spec
    assert CFG_12B["model_type"] == "gemma4_unified" and CFG_12B["text_config"]["attention_k_eq_v"] is True
    hf = ModelSpec.from_hf_config(CFG_12B)
    assert g12.spec_hash() == hf.spec_hash() == load_spec(SPECS / "gemma4-12b.json").spec_hash()
    # the file has no K = V key: the per-layer kv counts imply it when the caller does not say
    assert ModelSpec.from_gguf_metadata(gguf_md(hdr=HDR_12B)).spec_hash() == g12.spec_hash()


def test_12b_shape(g12):
    assert (g12.hidden, g12.num_layers, g12.num_heads, g12.num_kv_heads, g12.global_kv_heads) == (3840, 48, 16, 8, 1)
    assert (g12.head_dim, g12.global_head_dim, g12.rotary_dim, g12.global_rotary_dim) == (256, 512, 256, 128)
    assert (g12.intermediate, g12.intermediate_shared, g12.ple_dim, g12.logit_softcap) == (15360, 0, 0, 30.0)
    assert (g12.rope_theta, g12.rope_local_theta, g12.sliding_window, g12.k_eq_v) == (1e6, 1e4, 1024, True)
    assert [i for i, t in enumerate(g12.layer_types) if t == DENSE] == list(range(5, 48, 6))
    assert set(g12.layer_types) == {DENSE_LOCAL, DENSE} and G4.kv_sources(g12) == [-1] * 48


def test_12b_global_layers_have_no_v_projection_in_the_file():
    have = HDR_12B["tensors"]
    for l in range(48):
        glob = (l + 1) % 6 == 0
        assert (f"blk.{l}.attn_v.weight" in have) != glob
        assert have[f"blk.{l}.attn_k.weight"][0] == [3840, 512 if glob else 2048]


def test_12b_plan_reads_tensors_the_file_has_at_their_widths(unvalidated, g12):
    have = HDR_12B["tensors"]
    by_kind, glob, _ = plan_names(g12)
    assert set(glob) <= set(have)
    for l, kind in enumerate(g12.layer_types):
        names = [n.replace("{l}", str(l)) for n in by_kind[kind] + [G4.T_SCALAR]]
        assert set(names) <= set(have), sorted(set(names) - set(have))
    p = G4.pack_plan(g12)
    for kind, l in ((DENSE_LOCAL, 0), (DENSE, 5)):
        for op in p["layer_types"][kind]["pool"]:
            dims, ty = have[op["tensor"].replace("{l}", str(l))]
            assert ty == "Q4_0" and op["op"] == "std_perm_gguf"
            assert op["in_dim"] == dims[0] and op["nch"] * 8192 == dims[0] * dims[1]
    # K = V: the global kind's v slot streams the k projection, the sliding kind's its own v
    tensors = {k: [o["tensor"] for o in p["layer_types"][k]["pool"]] for k in (DENSE_LOCAL, DENSE)}
    assert tensors[DENSE].count(G4.T_K) == 2 and G4.T_V not in tensors[DENSE]
    assert tensors[DENSE_LOCAL].count(G4.T_K) == 1 and G4.T_V in tensors[DENSE_LOCAL]


@pytest.mark.parametrize("part,hd,rot,kvh,e_a,kv_row,ptab_row", [
    ("local", 256, 256, 8, 4096, 8192, 4096), ("global", 512, 128, 1, 2048, 2048, 2048)])
def test_12b_part_geometry(unvalidated, monkeypatch, g12, part, hd, rot, kvh, e_a, kv_row, ptab_row):
    monkeypatch.setenv("G4_PART", part)
    R = G4.recipe(g12)
    G, L = R.geo, R.layout
    assert (G.HD, G.ROT, G.KVH, G.N_CORES, G.KV_SHARED, G.ACORES, G.NHL, G.RB) == (hd, rot, kvh, 4, False, 4, 4, 1)
    assert (L.E_A, L.KV_ROW, L.PTAB_ROW, G.PTAB_ELEMS, G.PTAB_CS_OFF) == (e_a, kv_row, ptab_row, 1, 512)


def test_12b_manifest_has_the_scalar_but_no_per_layer_inputs(unvalidated, g12):
    m = manifest(g12)
    assert "ple" not in m and m["layer_scalar"] == G4.T_SCALAR and m["logit_softcap"] == 30.0
    assert m["kv_source"] == [-1] * 48
    assert {k: v["kv_row"] for k, v in m["kernels"].items() if "kv_row" in v} == {"dx_local": 8192, "dx": 2048}
    c = m["hf_config_check"]
    assert (c["num_global_key_value_heads"], c["attention_k_eq_v"], c["hidden_size_per_layer_input"]) == (1, True, 0)
    assert {"gemma4", "gemma4_unified", "gemma4_unified_text"} <= set(c["model_type"])


def test_the_e_series_config_checks_did_not_move(unvalidated, e4b):
    c = manifest(e4b)["hf_config_check"]
    assert c["model_type"] == ["gemma4", "gemma4_text"]
    assert not {"num_global_key_value_heads", "attention_k_eq_v"} & set(c)
