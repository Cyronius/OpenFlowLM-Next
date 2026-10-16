"""Gemma 4 E2B / E4B (text): Gemma 3's dense layer -- sandwich norms, GeGLU-tanh, sliding and global
layers -- with five changes, every one a property of the family:

- global layers have 512-wide heads that rotate only their first quarter, each pair spanning the head
  (`proportional` RoPE: pairs (i, i + HD/2) for i < ROT/2, frequencies theta^(-2i/HD)); so a sliding and
  a global layer are two attention points and two dx builds;
- scores are unscaled (scaling 1.0) and each v head is RMS-normalised, without a weight, before it is
  cached;
- the last layers compute no K or V: each attends over the cache of the last earlier layer of its own kind,
  that layer's newest row included (`kv_sources`), so their dx build has no k / v projection at all;
- E2B doubles the FFN width in those KV-shared layers;
- each layer ends with a per-layer input gated into the residual and a per-layer scalar, and the logits are
  soft-capped -- all three on the host, from the manifest's `ple`, `layer_scalar` and `logit_softcap`.

GGUF only (Google ships the QAT Q4_0 file; there is no q4nx container), so every pool is GGUF-direct. The
pools are the bf16-scale 5120-byte chunk (`std_perm_gguf` narrows the fp16 block scales at pack time), not
the f32-scale twin: the memory rule keeps f32 out of device memory, and the kernel narrows f32 to bf16 on
load anyway, so the numbers are the same (open_kernels/gguf_scale_width.py).

One dx build per layer KIND: (local | global) x (own | shared KV). `G4_PART` names the kind a design build
is for, and recipe(spec) returns that kind's dense recipe; with no part set it returns the first kind's, which
is what the kernel-TU generator reads (it only needs the chunks-per-element, the same in every kind: _check).
"""
from __future__ import annotations

import os
from dataclasses import replace

from . import dense as D
from .attnknobs import probe_env  # noqa: F401  (cache.py reads it off the family module)
from .catalogue import OpRangeError, check_buffer_args, require
from .qwen36moe import CHUNK, MB, q4_chunks, roundup
from .spec import DENSE, DENSE_LOCAL, DENSE_LOCAL_SHARED, DENSE_SHARED, ModelSpec

band_bytes = D.band_bytes    # noqa: F401  (dx.py: band_bytes(K, quant))
chunk_bytes = D.chunk_bytes  # noqa: F401  (dx.py: chunk_bytes(quant))

KINDS = (DENSE_LOCAL, DENSE, DENSE_LOCAL_SHARED, DENSE_SHARED)
OWN = {DENSE_LOCAL_SHARED: DENSE_LOCAL, DENSE_SHARED: DENSE}          # a shared kind -> whose cache it reads
PART = {DENSE_LOCAL: "local", DENSE: "global", DENSE_LOCAL_SHARED: "local_shared", DENSE_SHARED: "global_shared"}
KERNEL = {DENSE_LOCAL: "dx_local", DENSE: "dx", DENSE_LOCAL_SHARED: "dx_local_shared", DENSE_SHARED: "dx_shared"}

# GGUF tensor names, llama.cpp's gemma4 (read off google/gemma-4-E4B-it-qat-q4_0-gguf). The plan names
# them directly: a GGUF-only family has no other naming to translate from.
BLK = "blk.{l}."
T_Q, T_K, T_V, T_O = BLK + "attn_q.weight", BLK + "attn_k.weight", BLK + "attn_v.weight", BLK + "attn_output.weight"
T_UP, T_GATE, T_DOWN = BLK + "ffn_up.weight", BLK + "ffn_gate.weight", BLK + "ffn_down.weight"
T_QN, T_KN = BLK + "attn_q_norm.weight", BLK + "attn_k_norm.weight"
T_LN, T_POSTLN = BLK + "attn_norm.weight", BLK + "post_attention_norm.weight"
T_PREFFN, T_POSTFFN = BLK + "ffn_norm.weight", BLK + "post_ffw_norm.weight"
T_PLE_GATE, T_PLE_PROJ, T_PLE_NORM = BLK + "inp_gate.weight", BLK + "proj.weight", BLK + "post_norm.weight"
T_SCALAR = BLK + "layer_output_scale.weight"
T_EMBED, T_NORM = "token_embd.weight", "output_norm.weight"
T_PLE_EMBED, T_PLE_MODEL, T_PLE_MODEL_NORM = ("per_layer_token_embd.weight", "per_layer_model_proj.weight",
                                              "per_layer_proj_norm.weight")


def kinds(spec: ModelSpec) -> list[str]:
    return [k for k in KINDS if k in spec.layer_types]


def _kind_spec(spec: ModelSpec, kind: str) -> ModelSpec:
    glob = kind in (DENSE, DENSE_SHARED)
    hd = (spec.global_head_dim or spec.head_dim) if glob else spec.head_dim
    rot = (spec.global_rotary_dim or hd) if glob else spec.rotary_dim
    ff = (spec.intermediate_shared or spec.intermediate) if kind in OWN else spec.intermediate
    return replace(spec, layer_types=(kind,), num_layers=1, head_dim=hd, rotary_dim=rot, intermediate=ff,
                   global_head_dim=0, global_rotary_dim=0, intermediate_shared=0, extra={})


def per_call_cap(spec: ModelSpec) -> int | None:
    """E2B's KV-shared layers have a twice-wider FFN, whose activation table leaves room for one chunk
    per weight element where the other kinds take two. The dense kernel TUs are generated once per
    export, so every kind takes the smaller; None when the kinds agree (E4B), which changes nothing."""
    pcs = {D.per_call(_kind_spec(spec, k)) for k in kinds(spec)}
    return min(pcs) if len(pcs) > 1 else None


def sub_spec(spec: ModelSpec, kind: str) -> ModelSpec:
    """One layer kind as the dense recipe sees it: its head width, rotation and FFN width."""
    s = _kind_spec(spec, kind)
    cap = per_call_cap(spec)
    return replace(s, extra={"per_call": cap}) if cap else s


def kv_sources(spec: ModelSpec) -> list[int]:
    """Per layer: -1 for a layer with its own cache, else the layer whose cache it attends over -- the last
    layer of its own kind before the first shared one (transformers' store_full_length_kv)."""
    lt = spec.layer_types
    first = next((i for i, t in enumerate(lt) if t in OWN), len(lt))
    out = []
    for t in lt:
        if t in OWN:
            src = [j for j in range(first) if lt[j] == OWN[t]]
            if not src:
                raise OpRangeError(f"gemma4: a {t} layer has no earlier {OWN[t]} layer to share the cache of")
            out.append(src[-1])
        else:
            out.append(-1)
    if any(t not in OWN for t in lt[first:]):
        raise OpRangeError("gemma4: a layer with its own cache after the first KV-shared one")
    return out


def _check(spec: ModelSpec) -> None:
    if spec.family != "gemma4":
        raise OpRangeError(f"gemma4 recipe on a {spec.family!r} spec")
    bad = sorted(set(spec.layer_types) - set(KINDS))
    if bad:
        raise OpRangeError(f"gemma4: layer types {bad}")
    if not spec.ple_dim:
        raise OpRangeError("gemma4: no per-layer input width; the E-series always has one")
    kv_sources(spec)
    pcs = {D.per_call(sub_spec(spec, k)) for k in kinds(spec)}
    if len(pcs) != 1:
        raise OpRangeError(f"gemma4: the layer kinds disagree on chunks per weight element {sorted(pcs)}; "
                           f"the dense kernel TUs are generated once per export")


def _part() -> str | None:
    p = os.environ.get("G4_PART")
    if p and p not in PART.values():
        raise OpRangeError(f"G4_PART={p!r} (one of {sorted(PART.values())})")
    return p


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> D.DenseRecipe:
    _check(spec)
    p = _part()
    kind = next((k for k, v in PART.items() if v == p), None) if p else kinds(spec)[0]
    if kind not in spec.layer_types:
        raise OpRangeError(f"gemma4: G4_PART={p} but the spec has no {kind} layer")
    return D.recipe(sub_spec(spec, kind), max_ctx)


def layout(spec: ModelSpec, max_ctx: int = 4096) -> D.DenseLayout:
    return recipe(spec, max_ctx).layout


def pool_bytes(spec: ModelSpec, max_ctx: int = 4096) -> int:
    return max(D.layout(sub_spec(spec, k), max_ctx).POOL_BYTES for k in kinds(spec))


def _gguf_op(op: dict) -> dict:
    """A dense plan op, aimed at the GGUF file: projections pack through std_perm_gguf (bf16 scales into
    5120-byte chunks), and the HF-style tensor name becomes the GGUF one."""
    names = {"model.layers.{l}.self_attn.q_proj.weight": T_Q, "model.layers.{l}.self_attn.k_proj.weight": T_K,
             "model.layers.{l}.self_attn.v_proj.weight": T_V, "model.layers.{l}.self_attn.o_proj.weight": T_O,
             "model.layers.{l}.mlp.up_proj.weight": T_UP, "model.layers.{l}.mlp.gate_proj.weight": T_GATE,
             "model.layers.{l}.mlp.down_proj.weight": T_DOWN,
             "model.layers.{l}.self_attn.q_norm.weight": T_QN, "model.layers.{l}.self_attn.k_norm.weight": T_KN,
             "model.layers.{l}.input_layernorm.weight": T_LN,
             "model.layers.{l}.post_attention_layernorm.weight": T_POSTLN,
             "model.layers.{l}.pre_feedforward_layernorm.weight": T_PREFFN,
             "model.layers.{l}.post_feedforward_layernorm.weight": T_POSTFFN}
    o = dict(op)
    if o["tensor"] not in names:
        raise OpRangeError(f"gemma4: no GGUF name for {o['tensor']}")
    o["tensor"] = names[o["tensor"]]
    if o["op"] == "std_perm":
        o["op"] = "std_perm_gguf"
    return o


def pack_plan(spec: ModelSpec) -> dict:
    L0 = D.layout(sub_spec(spec, kinds(spec)[0]))
    hid = spec.hidden
    types = {}
    for k in kinds(spec):
        plan = D.pack_plan(sub_spec(spec, k))["layer_types"][k]
        types[k] = {"pool": [_gguf_op(o) for o in plan["pool"]], "consts": [_gguf_op(o) for o in plan["consts"]]}
    return {
        "pool_bytes": pool_bytes(spec), "chunk_bytes": CHUNK,
        "layer_types": types,
        # the head is the tied embedding (token_embd), re-quantized to q4_1 when the file is not 4-bit
        "lm_head": {"pool_bytes": L0.LMHEAD_POOL_BYTES,
                    "ops": [{"op": "std_perm_gguf", "tensor": T_EMBED, "dst": 0,
                             "nch": q4_chunks(D.lm_rows(spec), hid), "in_dim": hid}]},
        "embed": {"tensor": T_EMBED, "dim": hid, "scale": hid ** 0.5},
        "norm": {"tensor": T_NORM, "bytes": hid * 2},
    }


def proportional_inv_freq(theta: float, rot: int, hd: int) -> list[float]:
    """transformers' _compute_proportional_rope_parameters: rot/2 rotated pairs at theta^(-2i/hd)."""
    return [theta ** (-2.0 * i / hd) for i in range(rot // 2)]


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    """One dx kernel per kind on its own xclbin context (the kinds are different designs), one position
    table per attention kind (the shared kinds read their owner's), and the host stages' keys."""
    L0 = D.layout(sub_spec(spec, kinds(spec)[0]), max_ctx)
    out = {
        "contexts": {"ln": "ln/final.xclbin", "lm": "lm_head_q4/final.xclbin"},
        "kernels": {"ln": {"context": "ln", "insts": "ln/insts.bin", "build": "ln"},
                    "lm": {"context": "lm", "insts": "lm_head_q4/insts.bin", "build": "lm_head_q4"}},
        "layer_types": {},
        "tail": [{"op": "run", "kernel": "ln", "args": ["xres", "zero", "normw", "xresf", "hn"]},
                 {"op": "run", "kernel": "lm", "args": ["lmpool", "hn", "logits"]}],
        "globals": {"xres": spec.hidden * 4, "zero": spec.hidden * 4, "normw": spec.hidden * 2,
                    "xresf": spec.hidden * 4, "hn": spec.hidden * 2, "logits": D.lm_rows(spec) * 4,
                    "lmpool": L0.LMHEAD_POOL_BYTES},
    }
    for k in kinds(spec):
        s = sub_spec(spec, k)
        L, G = D.layout(s, max_ctx), D.geometry(s)
        local = k in (DENSE_LOCAL, DENSE_LOCAL_SHARED)
        kn, part, tab = KERNEL[k], PART[k], "ptab_local" if local else "ptab"
        window = spec.sliding_window if local else 0
        args = ["pool", "xres", "consts", "state", "act", tab]
        check_buffer_args(kn, args)
        out["contexts"][kn] = f"dx_{part}/final.xclbin"
        out["kernels"][kn] = {"context": kn, "insts": f"dx_{part}/insts.bin", "patch": "attnpos",
                              "build": f"dx_{part}", "window": window, "kv_row": L.KV_ROW}
        if k not in OWN:
            inv = ([spec.rope_local_theta ** (-2.0 * i / s.rotary_dim) for i in range(s.rotary_dim // 2)] if local
                   else proportional_inv_freq(spec.rope_theta, s.rotary_dim, s.head_dim))
            out["globals"][tab] = {"per_row": L.PTAB_ROW, "inv_freq": inv, "window": window,
                                   "cs_off": G.PTAB_CS_OFF, "rotary_dim": s.rotary_dim}
        out["layer_types"][k] = {
            "buffers": {"consts": L.CD_BYTES, "act": L.AD_BYTES,
                        "state": ({"kind": "kv_shared", "row": L.KV_ROW} if k in OWN
                                  else {"kind": "kv", "row": L.KV_ROW})},
            "program": [{"op": "run", "kernel": kn, "args": args}],
        }
    return out


def host_stages(spec: ModelSpec) -> dict:
    """What the engine computes on the host around the dispatches (Core's gemma4 host stages)."""
    return {
        "kv_source": kv_sources(spec),
        "ple": {"dim": spec.ple_dim, "embed": T_PLE_EMBED, "embed_scale": spec.ple_dim ** 0.5,
                "model_proj": T_PLE_MODEL, "model_proj_scale": spec.hidden ** -0.5, "model_norm": T_PLE_MODEL_NORM,
                "input_scale": 2.0 ** -0.5, "gate": T_PLE_GATE, "proj": T_PLE_PROJ, "norm": T_PLE_NORM,
                "act": spec.activation, "eps": spec.norm_eps},
        "layer_scalar": T_SCALAR,
        "logit_softcap": spec.logit_softcap,
    }


def builds(spec: ModelSpec) -> dict[str, dict]:
    n = D.cores_for(spec)
    b = {f"dx_{PART[k]}": {"design": "dense/dx.py",
                           "build_dir": f"dense/build_gemma4_h{spec.hidden}_{PART[k]}",
                           "env": {"G4_PART": PART[k]}} for k in kinds(spec)}
    b["ln"] = {"design": "ln/ln.py", "build_dir": f"ln/build_{spec.hidden}_{spec.norm_eps:g}",
               "env": {"LN_N": str(spec.hidden), "LN_EPS": f"{spec.norm_eps:g}"}}
    b["lm_head_q4"] = {"design": "lm_head_q4/lm_head_q4.py", "build_dir": f"lm_head_q4/build_{D.lm_rows(spec)}",
                       "env": {"LMHEAD_N": str(D.lm_rows(spec)), "LMHEAD_K": str(spec.hidden),
                               "LMHEAD_CORES": str(n)}}
    return b


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    s = sub_spec(spec, kinds(spec)[0])
    L = D.layout(s, max_ctx)
    return {"hidden": spec.hidden, "vocab": D.lm_rows(spec), "real_vocab": spec.real_vocab,
            "chunk_bytes": CHUNK, "pool_bytes": pool_bytes(spec, max_ctx),
            "lmhead_pool_bytes": L.LMHEAD_POOL_BYTES,
            "kv_row": L.KV_ROW, "ptab_row": L.PTAB_ROW, "rotary_dim": s.rotary_dim, "rope_theta": spec.rope_theta,
            "rope_inv_freq": [spec.rope_local_theta ** (-2.0 * i / s.rotary_dim) for i in range(s.rotary_dim // 2)]}


def hf_config_check(spec: ModelSpec) -> dict:
    return {"hidden_size": spec.hidden, "num_hidden_layers": spec.num_layers, "vocab_size": spec.vocab,
            "num_attention_heads": spec.num_heads, "num_key_value_heads": spec.num_kv_heads,
            "intermediate_size": spec.intermediate, "head_dim": spec.head_dim,
            "global_head_dim": spec.global_head_dim or spec.head_dim, "sliding_window": spec.sliding_window,
            "hidden_size_per_layer_input": spec.ple_dim, "final_logit_softcapping": spec.logit_softcap}


GEN_KERNELS = D.GEN_KERNELS
KERNEL_SOURCES = list(D.KERNEL_SOURCES)
Q8_ROLES = frozenset()          # Google's file is Q4_0 throughout the layers; no q8 role
GGUF_ONLY = True                # recipes/manifest.py: the manifest's own view is the GGUF one
