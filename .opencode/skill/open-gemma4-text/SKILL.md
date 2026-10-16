---
name: open-gemma4-text
description: Build, verify and run Gemma 4 E4B / E2B (text) on the open XDNA2 kernels from Google's GGUF -- the gemma4 recipe (four dx kinds: sliding / global x own / shared KV cache), the attn.h knobs it adds, the engine's KV sharing, per-layer inputs and soft cap, and the oflm serve path. Use when exporting the gemma4 kernel set, bringing up E2B, debugging a Gemma 4 load, or adding the 12B (which shares the 512-wide attention).
---

# Gemma 4 E-series text on the open kernels

Spec: `specs/open-engine/spec.md` OPEN-FAMILY-GEMMA4 and OPEN-GGUF-BF16-POOL. Recipe:
`open_kernels/recipes/gemma4.py`. Reference: `open_kernels/model/replica_gemma4.py` (fp64, from the GGUF),
held to transformers by `gemma4_transformers_check.py`.

## What the model is (read off the GGUF, not the plan)

- E4B: 42 layers (5 sliding then 1 global, x7), hidden 2560, ff 10240, 8 heads over 2 kv, head dims
  256 (sliding) / 512 (global), window 512, per-layer input 256, layers 24-41 KV-shared (sources 22 and
  23), softcap 30, vocab 262144, tied head. `use_double_wide_mlp` is FALSE on E4B (E2B: true).
- E2B: 35 layers (4 + 1), hidden 1536, ff 6144 (12288 in its 20 shared layers), ONE kv head.
- Scores are unscaled (`scaling = 1.0`); v gets a weightless RMSNorm; norms are stored as `w` (no `1 + w`).
- Global RoPE is `proportional`: pairs (i, i + 256) for i < 64 at theta 1e6 ^ (-2i/512). llama.cpp stores
  it as `rope_freqs` (64 ones, then 1e30).
- Google's QAT GGUF differs from the bf16 HF checkpoint (norms, layer scalars): validate against the
  GGUF's own weights, never the HF safetensors.
- GGUF names (llama.cpp gemma4): `attn_norm` (input), `post_attention_norm`, `ffn_norm` (pre-FFN),
  `post_ffw_norm`, `inp_gate` / `proj` / `post_norm` (per-layer input), `layer_output_scale`,
  `per_layer_token_embd` (Q6_K), `per_layer_model_proj` (F16), `per_layer_proj_norm`, `rope_freqs`.
  The recipe names them directly (`gemma4.py` T_*), not through `GgufFile::gguf_name` -- that map
  swaps the two sandwich norms for gemma3-style HF names.

## The kernel side

- Four `dx` builds, `G4_PART` = local | global | local_shared | global_shared, one hardware context each.
- attn.h knobs, all default-off (every other family preprocesses to the same tokens --
  `utilities/mlir_equiv.py ppcheck`): `ATTN_SCALE_ONE`, `ATTN_VNORM`, `ATTN_ROPE_PAIR` (partner distance),
  `ATTN_PTAB_CS_OFF` (the sliding kinds' 1024-byte element cannot hold 512 + 1024 bytes of record, so
  cos / sin move to the second element).
- `dx.py` KV-shared variant: no k / v GEMV, no k / v attention work, no cache-row drain; the window and then
  this position's row (the `attnpos` kind-1 patch, now a fill) are read from the source layer's cache.
- Pools are GGUF-direct at 5120-byte chunks (bf16 scales narrowed at pack time); no `*_f32` twin.

## Export (WSL, under `dibs run build`)

```
source ~/ironenv142/bin/activate; export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib:$LD_LIBRARY_PATH
cd open_kernels
OPEN_KERNELS_UNVALIDATED=1 python export_qwen36_kernels.py --spec recipes/specs/gemma4-e4b.json \
   --out <dir> --only dx_global,dx_local,dx_global_shared,dx_local_shared,ln,lm_head_q4 --force
python ../utilities/core_sizes.py designs/dense/build_gemma4_h2560_global      # program memory per core
```

## Equivalence evidence for the shared design files

`python utilities/mlir_equiv.py dump <dir> --root <pristine tree>` before, `dump` after, `diff`, then
`ppcheck <after> <pristine tree>`: MLIR identical for every dense spec, and every (kernel source, flags)
pair preprocesses to the same tokens.
