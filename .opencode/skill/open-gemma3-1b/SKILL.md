---
name: open-gemma3-1b
description: Build, validate and install Gemma 3 1B (gemma3:1b) on the open kernels from Google's QAT Q4_0 GGUF -- the zero-padded hidden width (1152 -> 1280), the one-kv-head attention element, the GGUF norm names and the F16 tied head. Use when rebuilding its kernel set, bringing up another dense model whose widths are not whole 256-column k-tiles or that has a single kv head (Gemma 4), or when a Gemma 3 GGUF loads but answers badly.
---

# Gemma 3 1B on the open kernels (GGUF-direct)

## What is different about it

| | Gemma 3 1B | why it matters |
|---|---|---|
| hidden | 1152 = 4.5 k-tiles of 256 | the q4 GEMV reads whole k-tiles: the model runs **zero-padded to 1280** (OPEN-PACK-PAD) |
| heads | 4 over **1** kv head, head dim 256 | the attention element was half a KV row; at one kv head it is the whole row (OPEN-ATTN-ONE-KV-HEAD) |
| intermediate | 6912 = 27 k-tiles | fine at 4 cores (27 bands each); only the 8-core band group (512) misses it |
| cores | 4 | the kv width (256) is 4 bands, so 4 is the most any width allows |
| head | tied to an **F16** embedding | re-quantized to q4_1 at load (`std_perm_gguf`, the Q8_0 head's law); no measurable NLL cost over 1024 tokens (`utilities/quant-compare/README.md`) |
| window | 512 (checkpoint), 1024 (GGUF metadata) | the kernel set is built at 512; see "traps" |

No `.q4nx` container is read: FLM's `Gemma3-1B-NPU2` uses 1280-byte chunks, and a padded
`std_perm` refuses by name. Install the GGUF.

The kernel set still lives under the official name (`xclbins/Gemma3-1B-NPU2/open_kernels`),
which is how a GGUF install finds it. So `Engine::find_kernels` declines a set whose q4nx view
is padded (`Manifest::reads_gguf_only`) unless the model directory holds a `model.gguf` and no
`model.q4nx`, and the official `gemma3:1b` container keeps the closed `gemma_text_npu` engine.
Without that check it picks the open set and fails to load.

## The padding law (dense recipe)

- `recipes/dense.py widths(spec)`: unchanged widths for every spec whose hidden and
  intermediate are whole k-tiles (no shipped set moves); otherwise the most cores the q / kv
  widths allow and `pad_width` for hidden / intermediate. The 1B: `(4, 1280, 6912)`.
- Pack ops name the file's own shape: `src_dim` (columns, on q k v up gate and the head) and
  `src_rows` (rows, on o and down). `pools.cpp std_perm_gguf` writes the real blocks and leaves
  the pad's scales, mins and codes at zero; it refuses a file whose shape the op misstates. An
  op without `src_rows` needs the file to hold exactly the plan's rows; only a head may be short,
  by less than one 64-row band (a vocabulary the plan rounds up).
- The residual is 1280 wide. The embedding row and every norm weight are zero-extended
  (`GgufFile::embed_row`, `pack_norm`), and every padded weight row is zero, so the pad stays
  zero through all 26 layers.
- Every RMSNorm divides by 1152: `-DLN_DIV=1152` on the `ln` design and on dx.py's norm core.
  Without `LN_DIV` the norm kernels preprocess exactly as before.
- `hf_config_check.hidden_size` stays 1152; `layout.hidden` (the buffer) is 1280.

## Build

```sh
# WSL, under `dibs run build`
source ~/ironenv142/bin/activate
export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib:$LD_LIBRARY_PATH
python open_kernels/export_qwen36_kernels.py \
  --spec open_kernels/recipes/specs/gemma3-1b.json --out <xclbins root>/Gemma3-1B-NPU2/open_kernels
```

The catalogue lists the 1B's points (gemv_q4 K 1280 / 6912, ln (1280, 1), ln_div (1280, 1152),
lm_head_q4 K 1280, attention (256, 4, 1, 256)), so no `OPEN_KERNELS_UNVALIDATED`.
`utilities/build-all.sh` builds it with every other spec.

## Validate (smoke, ~15 min)

1. Reference from the GGUF itself (`model/gguf_model.py` reads it the way the engine packs it):
   `python open_kernels/model/make_decode.py --model-dir <dir with the HF config.json +
   tokenizer.json> --gguf <gemma-3-1b-it-q4_0.gguf> --ids <prompt ids> --tokens N --layers 2
   --out <out>` -> `ref_logits*.bin`, `ref_ids.txt`.
2. `open_qwen36_cli --model <dir with model.gguf + config.json> --kernels <set> --layers 2 --ids
   <ref_ids.txt> --prefill-logits --dump-logits <out>/y --max-tokens 1`, rename `y_p<t>.bin` to
   `y_logits*.bin`, `python open_kernels/model/compare_decode.py --tokens N --out <out>`.
3. The same at 26 layers. Expect a near tie (a 0.02-logit top-2 gap) to flip now and then.
4. `oflm-add google/gemma-3-1b-it-qat-q4_0-gguf [--tag <t>]` (on a shared box add `--config`,
   `--models-root` and `--xclbin-dir` pointing at a scratch root). It resolves to the official
   `gemma3:1b` entry (family `gemma3-text`) and links the kernel set beside the model. Without
   `--tag` the tag is `gemma-it:1b`.
5. `oflm serve <tag> --port <p>` (the load logs `Gemma 3 (text) on the open kernels`), then
   `oflm-test --llm --model <tag> --port <p> --backend-os windows --gen-lim 1024`. The cap keeps
   the run near five minutes; "Teach me Maxwell's equations" otherwise runs long on a 1B.

The 2026-10-16 bring-up's logs are in the worktree's `.claude/plans/gemma3-1b-run/`.

## Traps

- **The GGUF's sliding window is wrong.** `gemma3.attention.sliding_window` says 1024; the QAT
  checkpoint's own config.json (and google/gemma-3-1b-it's) says 512. The kernel set is built
  at 512 and a model dir without a config.json is refused (its derived 1024 fails the check).
- **Norm names.** llama.cpp calls HF's `post_attention_layernorm` `post_attention_norm` and
  `pre_feedforward_layernorm` `ffn_norm` in a Gemma GGUF; a llama GGUF's `ffn_norm` is HF's
  `post_attention_layernorm`. `GgufFile::resolve` decides by whether the file has
  `post_attention_norm`. The old map swapped Gemma's two (52 apart in value).
- **real_vocab.** tokenizer.json has id 262144 (`<image_soft_token>`) past the 262144-row head;
  `from_hf_config` clamps `real_vocab` to the head's rows, or the engine reads past its logits.
- **f32 metadata.** A GGUF stores `1e-6` as 9.99999997e-07 and a linear RoPE factor of 1;
  `_gemma3_gguf` reads them back as the config's values, so the two derivations agree in every
  field but the sliding window (and their `spec_hash` differs by that).
- The reference must narrow the GGUF scales to bf16 (the kernel's `load_scale`) and
  re-quantize the F16 head with pools.cpp's law, or it measures the quantization, not the kernels.
- **The family name.** A Gemma 3 GGUF says `gemma3`, but the official list files the 1B under
  `gemma3-text`; `oflm-add` matches the text-only entry at the same size (`TEXT_ONLY_FAMILY`).
  Before that it picked the 4B's kernels.
- **Prefill is token by token.** No dense GGUF view has a block-prefill route (its GEMMs read
  only the bf16-scale pool), and `gemm_route` also refuses the 1B for its two-element position
  record, its one kv head (both `dx_attn.py` limits) and its padded widths (the host norms).
  Bringing it back is kernel work in `dx_attn.py` plus f32-scale GEMM twins, not a host fix.
