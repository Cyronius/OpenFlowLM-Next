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
- attn.h knobs, all default-off (every other family preprocesses to the same tokens -- see the
  equivalence section): `ATTN_SCALE_ONE`, `ATTN_VNORM`, `ATTN_ROPE_PAIR` (partner distance),
  `ATTN_PTAB_CS_OFF` (the sliding kinds' 1024-byte element cannot hold 512 + 1024 bytes of record, so
  cos / sin move to the second element), `ATTN_KV_SHARED` (E2B's shared kinds).
- `dx.py` KV-shared variant: no k / v GEMV, no k / v attention work, no cache-row drain; the window and then
  this position's row (the `attnpos` kind-1 patch, now a fill) are read from the source layer's cache.
- Pools are GGUF-direct at 5120-byte chunks (bf16 scales narrowed at pack time); no `*_f32` twin.

## Export (WSL, under `dibs run build`)

```
source ~/ironenv142/bin/activate; export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib:$LD_LIBRARY_PATH
cd open_kernels
OPEN_KERNELS_UNVALIDATED=1 python export_qwen36_kernels.py --spec recipes/specs/gemma4-e4b.json \
   --out <dir> --only dx_global,dx_local,dx_global_shared,dx_local_shared,ln,lm_head_q4 --force -j 2
python ../utilities/core_sizes.py designs/dense/build_gemma4_h2560_global      # program memory per core
```

E2B: `--spec recipes/specs/gemma4-e2b.json`, build dirs `build_gemma4_h1536_*`. About 8 minutes at `-j 2`
(the global kinds take ~200 s each); the exporter prints each set's fullest core. Export straight into
the installed model's `open_kernels` dir and pass that dir to `oflm-add --open-kernels` ("already in
place"). oflm-add copies the GGUF to `model.gguf`; on a tight disk, replace the copy with a hard link.

## Equivalence evidence for the shared design files

Three checks, each against a pristine tree (`git archive <rev> open_kernels utilities | tar -x -C <dir>`),
all in WSL with the mlir-aie env (`C:\models\gemma4-text\scripts\equiv_e2b.sh` is a worked example):

- `python utilities/attn_flags.py` in both trees (copy the new script into the old one): every spec's
  attention flags, one `<spec>@<build>` line per Gemma 4 kind. Identical lines = identical build lines.
- `python utilities/mlir_equiv.py --root <tree>/open_kernels --out <dir>` for both, then `cmp` each
  `<spec>.<build>.mlir`: identical MLIR.
- `ALL_SPECS=1 OLD_DIR=<pristine>/open_kernels bash utilities/attn_pp_identical.sh`: every attention TU
  preprocesses to the same tokens at every spec's flags. The worktree's copy has CRLF line endings
  (autocrlf), which WSL bash refuses: run an LF copy inside a snapshot of the tree, not the worktree file.

## E2B (one kv head)

- Needs the one-kv-head attention point (OPEN-ATTN-ONE-KV-HEAD, cherry-picked from feat/gemma3-1b).
- Its shared layers' 12288-wide FFN fits one chunk per weight element, the others two; the generated
  dense TUs are per export, so `gemma4.per_call_cap` pins every kind to one (E4B: no cap, no change).
- Sliding kinds run on 4 cores (256-row kv projection), global kinds on 8; each kind is its own xclbin.
- A KV-shared kind at one kv head reads this position's row back as one element: `ATTN_KV_SHARED`
  selects the one-pointer `attn_step_new` (dx.py adds the flag only there).
- Validate the replica first: `gemma4_transformers_check.py <e2b gguf> --layers 0,4,15,19`.

## Serving (`oflm serve`)

- Google's tokenizer_config.json names `<bos>` with no ids, and oflm exits at load without
  `bos_token_id` / `eos_token_id`: oflm-add merges them from the source directory's
  generation_config.json (bos 2, eos [1, 106, 50]).
- Google's config.json nests the text model under `text_config`; the adapter lifts absent keys to the
  top level on the open path, or the sampler is built with vocab 0 and the server crashes on the first
  token.
