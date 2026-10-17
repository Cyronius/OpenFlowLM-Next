---
name: open-gemma4-text
description: Build, verify and run Gemma 4 E4B / E2B (text) on the open XDNA2 kernels from Google's GGUF -- the gemma4 recipe (four dx kinds: sliding / global x own / shared KV cache), the attn.h knobs it adds, E2B's one kv head and double-wide FFN, the engine's KV sharing, per-layer inputs and soft cap, the guard that keeps FLM's official container off these kernels, and the oflm serve path. Use when exporting a gemma4 kernel set, debugging a Gemma 4 load or serve, or adding the 12B (which shares the 512-wide attention).
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
all in WSL with the mlir-aie env, the new side a snapshot of the working tree taken the same way:

- `python utilities/attn_flags.py` in both trees (copy the new script into the old one): every spec's
  attention flags, one `<spec>@<build>` line per Gemma 4 kind. Identical lines = identical build lines.
- `python utilities/mlir_equiv.py --root <tree>/open_kernels --out <dir>` for both, then `cmp` each
  `<spec>.<build>.mlir`: identical MLIR.
- `ALL_SPECS=1 OLD_DIR=<pristine>/open_kernels bash utilities/attn_pp_identical.sh`: every attention TU
  preprocesses to the same tokens at every spec's flags; a TU that fails to preprocess is an ERROR and
  the run exits 1. The worktree's copy has CRLF line endings (autocrlf), which WSL bash refuses: snapshot
  with `git -c core.autocrlf=false archive`, which writes LF, and run the snapshot's copy. It writes
  ~3.5 GB of preprocessed `.i` files under `<tree>/.claude/tmp/ppcheck`; delete them after.
- Where a kernel set that passed on the NPU exists, re-export with `--check <that dir>` (E4B: 12 of 12
  files match). `--no-build --check` re-runs only the comparison on the existing build dirs.

## E2B (one kv head)

- Needs the one-kv-head attention point (OPEN-ATTN-ONE-KV-HEAD, cherry-picked from feat/gemma3-1b).
- Its shared layers' 12288-wide FFN fits one chunk per weight element, the others two; the generated
  dense TUs are per export, so `gemma4.per_call_cap` pins every kind to one (E4B: no cap, no change).
- Sliding kinds run on 4 cores (256-row kv projection), global kinds on 8; each kind is its own xclbin.
- A KV-shared kind at one kv head reads this position's row back as one element: `ATTN_KV_SHARED`
  selects the one-pointer `attn_step_new` (dx.py adds the flag only there).
- Validate the replica first: `gemma4_transformers_check.py <e2b gguf> --layers 0,4,15,19`.
- Its GGUF lists `feed_forward_length` per layer; the engine's GGUF reader skips arrays, so
  `derive_config` takes the FFN width from layer 0's `ffn_up` tensor.
- Result 2026-10-16: logits corr 0.999998 / 0.999991 / 0.999988 at 6 / 20 / 35 layers against the
  replica with the q4_1 head, same top 5; greedy "The capital of France is Paris." The slice run:
  `open_qwen36_cli --model <dir> --kernels <set> --layers N --ids ... --dump-logits <prefix>`, then
  `model/compare_gemma4.py <prefix>_t0.bin <replica.npz> --key npu_logits_lN`.

## The official container must not pick up these kernels

The set is exported under the official model's name (`Gemma4-E4B-IT-NPU2`), where FLM's own `.q4nx`
container also looks. Its one view packs GGUF-direct (`std_perm_gguf`), so `Engine::find_kernels`
declines it unless the model directory reads a GGUF (`Manifest::reads_gguf_only`,
`Core::reads_gguf`), and the official tag stays on the closed engine. feat/gemma3-1b added the same
guard for zero-padded sets (3f701d3d); when the branches meet, keep one `reads_gguf_only` that is true
for either reason.

## Serving (`oflm serve`)

- Google's tokenizer_config.json names `<bos>` with no ids, and oflm exits at load without
  `bos_token_id` / `eos_token_id`: oflm-add merges them from the source directory's
  generation_config.json (bos 2, eos [1, 106, 50]).
- Google's config.json nests the text model under `text_config`; the adapter lifts absent keys to the
  top level on the open path, or the sampler is built with vocab 0 and the server crashes on the first
  token.
- Scratch serving: install with `oflm-add <dir> --tag <tag> --config <root>\model_list.json --models-root
  <root>\models --open-kernels <set> --no-xclbin`, then `oflm serve <tag> --port <p>` with OFLM_MODEL_PATH
  = `<root>` and OFLM_CONFIG_PATH = `<root>\model_list.json`. Start it with PowerShell `Start-Process
  -PassThru` writing the PID to a file, not through `$(...)`: the server inherits the pipe and the
  substitution waits for it to exit. Stop it by that PID.
- Prefill is token at a time (no block-prefill route for Gemma 4 yet), so a full `oflm-test --tools`
  (both modes) on E4B runs past 15 minutes; run the non-stream and stream halves separately. No speed
  has been measured (nothing ran under `dibs run --measure`).
- Results 2026-10-16: E4B `oflm-test --llm` 5 of 5, `--tools` 14 of the 15 checks reached. E2B
  `oflm-test --llm` 5 of 5.
- Garbled arithmetic in an answer ("$64 - $6", an answer that stops mid-sum) is the sampler, not the
  kernels: `Gemma4e::load_model` sets frequency and presence penalties of 1.0 over the last 64 tokens,
  which push down digits already written. Replay with `"frequency_penalty": 0, "presence_penalty": 0`
  in the request before suspecting the engine; the failing tool-loop request then answers right.
- Images and audio: the open path refuses them, and `OFLM_GEMMA4_ENGINE=closed` cannot load a GGUF
  install. They need a curated `gemma4-it` tag (FLM's `.q4nx` container).
- Never rewrite a shell script while a bash instance is still running it: bash reads scripts as it
  goes, and a half-read old instance ran the new text, restarted a server and overwrote its logs.
