---
name: open-qwen35-gguf
description: Run a Qwen3.5-family GGUF (llama.cpp arch qwen35 -- Qwen3.5 / 3.6 / 3.8 dense, MiniCPM-V-4.7, the 0.8B embedder and reranker) on the open kernels straight from the file, through the q4nx view. Use when adding another qwen35 GGUF, linking it to a kernel set, debugging "has no q4nx view" / "needs the kernel set's GGUF-direct build", or checking that the engine packs a GGUF the way the NumPy packer does.
---

# qwen35 GGUFs on the open kernels

## How it works

The qwen35 kernel set has ONE packing plan, written against the q4nx container
(5120-byte q4_1 chunks with bf16 scales, the 8704-byte q8 head). The engine serves a
llama.cpp GGUF to that plan as the container it expects -- `GgufQ4nxView`
(`src/open_qwen36/gguf_view.cpp`), mirrored byte for byte by
`open_kernels/model/gguf_q4nx.py`. No f32-scale twins, no nested `gguf` manifest. What the
view undoes (OPEN-GGUF-Q4NX-VIEW in `specs/open-engine/spec.md`):

- names: `attn_qkv` -> `linear_attn.qkv_proj`, `attn_gate` -> `self_attn.gate_proj` (the
  linear layer's z), `ssm_out` -> `ssm_out_proj`, `ssm_alpha`/`ssm_beta` -> the `.bf16` copies,
  `post_attention_norm` -> `post_attention_layernorm` (NOT `ffn_norm` as in the dense map);
- value heads untiled when `ssm.time_step_rank > ssm.group_count` (grouped head h is file head
  `(h % grp) * num_k + h / grp`);
- `attn_q` is `[q_h | gate_h]` per head in the file; the plan wants `[q | gate]`;
- `ssm_conv1d` is `[channels, taps]` in the file, `[taps, channels]` in the container;
- `ssm_a` (already `-exp(A_log)`), `ssm_dt.bias` and the `1 + w` norms pass unchanged.

Weight formats: Q4_1 / Q4_0 pack exactly (codes untouched, d / m narrowed fp16 -> bf16 RNE);
Q8_0 stays q8 (a q8 role streams it natively, else `requant_q4_1_chunks`); Q4_K is transcoded
(each sub-block's `d * sc` / `-dmin * m` rounded to bf16, codes kept); everything else (Q6_K,
F16, ...) is re-quantized by `requant_q4_1`'s law; the head is always q8 (Q8_0 exact, else per
32 values s = max|v|/127 toward +inf, code = floor(v/s + 1/2)), rows padded to 256. Q5_K is not
read yet (unsloth's Qwen3.5 Q4_K_M files carry it; oflm-add refuses them).

## The spec, and why a fine-tune may need its own

`recipes/load.py spec_from_model_dir` derives a GGUF model dir's spec from the GGUF itself
(OPEN-GGUF-QWEN35-SPEC): vocab padded to 256 rows (248320 for every Qwen3.5 vocabulary),
real vocab from `tokenizer.json`, the role formats from the tensor types. `real_vocab` is in
`spec_hash()`, so a fine-tune that adds tokens (MiniCPM-V-4.7-1B: 248094) does not link to the
base model's set even though every kernel is the same. Today such a model gets a spec file of
its own and a separate export (`recipes/specs/minicpm-v-4.7-1b.json`); whether to drop
`real_vocab` from the kernel-match hash is the maintainer's decision.

```
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/minicpm-v-4.7-1b.json
#   -> src/xclbins/MiniCPM-V-4.7-1B-gguf/open_kernels (41 kernels, the 0.8B q4_1 set's)
```

## Install and serve

```
python utilities/oflm-add/oflm-add.py openbmb/MiniCPM-V-4.7-1B-gguf --tag minicpm-v-4.7:1b \
    --open-kernels <xclbins>/MiniCPM-V-4.7-1B-gguf/open_kernels
```

oflm-add picks the Q4_1 file, takes family `qwen3.5` from the GGUF architecture, fetches the
tokenizer and `config.json` from the base repo, writes a text-only entry (`vlm: false`), and
links the kernel set. Traps:

- The GGUF gate checks the kernel set of the *official match* (`qwen3.5:0.8b` ->
  `Qwen3.5-0.8B-NPU2`), not the set whose spec hash matches. Without that set installed, or
  without `--open-kernels`, the install falls back to the base repo's safetensors (for MiniCPM-V,
  the VLM wrapper's). Pass `--open-kernels` for a fine-tune with its own set.
- llama.cpp writes the head as Q6_K over a Q4_1 / Q4_0 body; for a qwen35 GGUF oflm-add leaves the
  head out of its quant-name check (the engine repacks it as the q8 head anyway). A body tensor of
  another quant is still refused: MiniCPM-V's Q4_K_M file mixes Q6_K into qkv / v / down, so
  oflm-add refuses it even though the engine reads it.
- MiniCPM-V's `config.json` keeps `vocab_size` under `text_config`; the Qwen3.5 adapter
  (`modeling_qwen3_5vl.cpp`) republishes it at the top level for the sampler. The engine itself
  checks the kernel set against the GGUF's metadata, not that file.

Then `oflm serve minicpm-v-4.7:1b` and `oflm-test --llm --model minicpm-v-4.7:1b`.

## Checks

- Unit: `src/open_qwen36/gguf_pack_test` (ctest GGUF-PACK) and
  `specs/open-engine/tests/test_gguf_q4nx_view.py` / `test_qwen35_gguf_spec.py`. The two views
  must hash to the same FNV-1a on the shared synthetic GGUF; if one changes, both tests fail.
- A real file: `gguf_pack_test --pack <manifest.json> <model.gguf> <dir> 0 3` then
  `python open_kernels/model/gguf_q4nx_check.py <manifest.json> <model.gguf> <dir> 0 3`
  (engine pools vs recipes/pack.py, byte for byte).
- Slice vs fp64: `open_kernels/model/make_decode.py --model-dir <dir with model.gguf>` reads the
  GGUF through the NumPy view, so the reference and the NPU see the same weights.

Validated 2026-10-17 on MiniCPM-V-4.7-1B Q4_1 (OPEN-GGUF-Q4NX-VIEW's results): slice and full
model against fp64, block prefill against sequential, `oflm serve`, `oflm-test --llm` 5/5. The
export takes about 35 minutes with `-j 4` on a busy box; `lx` uses 15 712 of 16 384 B of program
memory at h1024.

## Not covered yet

Vision (the mmproj GGUF), `qwen35moe` (expert tensors), Q5_K, and the release export list (the
MiniCPM set is not in it).
