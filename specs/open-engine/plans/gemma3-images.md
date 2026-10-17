# Gemma 3 images on the open engine

2026-10-16. Plan item 8 of `.claude/plans/next5-models-plan.md`. Branch `feat/gemma3-images`,
stacked on `feat/gemma3-1b`.

Today an image sent to a Gemma 3 model fails on the open engine ("images need the closed
engine"), and only the closed `gemma_npu` DLL can read one. This plan adds the image path to the
open engine for the GGUF-direct Gemma 3 4B: the vision tower on the host, read from the model's
`mmproj` GGUF, and the image block through the existing decode kernels with the attention mask
Gemma 3 needs.

## Spec impact

- **New:** `OPEN-VISION-GEMMA3` (the image path: tower, projector, the bidirectional image
  block, serving and selection).
- **Modified:** `OPEN-FAMILY-GEMMA3` ("Images always need the closed engine" becomes "images
  need the closed engine only when the model has no `mmproj.gguf`").
- **Modified:** `OPEN-PACK-PAD`'s head allowance. Google's own 4B GGUF stores a 262144-row head
  against the kernel set's 262208 rows (the HF vocabulary's 64 padding rows were trimmed by the
  converter); the packer refused it. A head may now also fall short when it holds exactly the
  GGUF's own vocabulary, and the engine clamps `real_vocab` to the head's rows.

## The mask (the gate)

Gemma 3 attends both ways inside an image: each of an image's 256 soft tokens sees every token of
the same image, plus everything before the image (causally, inside the sliding window on the
local layers). Text stays causal. transformers builds this as `token_type_ids_mask_function`
OR-ed into the causal and sliding masks; llama.cpp decodes the image's batch non-causally.

Where it can go, given what the GGUF route has:

- **The batched prefill route.** A GGUF kernel set carries none (its GEMMs read only the
  bf16-scale pool; `recipes/dense.py` emits no `gemm_block` for `q4_1_f32`). Bringing it back is
  kernel work (f32-scale GEMM twins). Not taken.
- **Host attention for the image block.** The dense layer is ONE fused dispatch (`dx`: norms,
  projections, attention, FFN). Splitting attention out needs a new kernel program. Not taken.
- **Taken: the decode kernel, run twice per image token, layer by layer.** No kernel change.

Why it works: a decode dispatch attends over `[start, start + valid)` cached rows plus the row it
computes itself, and all four quantities that say where are patched per dispatch (`attnpos`:
the window fill's offset and length, the new row's drain, the position record's row). The record
carries `valid` / `nf`; its RoPE angles stay those of the token's own position. And a token's K/V
at layer l depend only on its input to layer l.

So the image block runs layer-major. For each layer l, with `S[0..N-1]` the block's cache rows:

1. **Pass A** writes K/V: token t (t >= 1) runs layer l from its layer-l input and drains into
   `S[t-1]`. Its attention output is discarded.
2. **Pass B** computes the outputs: token t's fill is `[s_t, P0 + N - 1)` (the prefix inside its
   window, then `S[0..N-2]`) and its own row comes from the dispatch itself; the drain goes to
   `S[N-1]`. Before token t (t >= 1) the host copies `S[N-1]` (token t-1's row) over `S[t-1]`
   (token t's). By induction `S[0..N-2]` holds every block token but t, exactly once, and after
   the last token the rows are back in natural order.
3. The last layer needs no pass B (its outputs feed nothing but the image tokens' own logits):
   its pass A drains every token into its natural row.

The attention set of every pass-B dispatch is exactly transformers': the prefix rows the query's
own window admits (`s_t = max(0, p_t + 1 - W)` on a local layer, 0 on a global one) and all N
image rows. Only the order of summation differs, which is fp32 rounding. The position records
of the block's rows are written with the custom counts for the block and restored after it.

Cost: `(2 (L - 1) + 1) N` single-layer dispatches instead of the `L N` of a causal pass, so
about twice the token-by-token prefill of 256 tokens, plus a 4 KB row copy and a record patch per
pass-B dispatch. No resident memory is added.

## The vision tower

SigLIP from the `mmproj` GGUF (`clip` arch, projector `gemma3`): 27 blocks of width 1152, 16
heads of 72, MLP 4304, patch 14 at 896 px (64 x 64 patches, a learned 4096-row position table),
LayerNorm with bias, GELU-tanh, then `post_ln`. The projector average-pools the 64 x 64 grid by
4 to 16 x 16 = 256 tokens, applies the soft-embedding RMSNorm (the GGUF stores `1 + w`), and
multiplies by `mm.input_projection` ([1152, 2560], `y = x W`).

- A third tower family in `src/open_qwen36/vision/` beside Qwen3-VL and Qwen2.5-VL, reusing its
  `linear`, `layer_norm` and `attention` (identity rotary tables).
- Weights stay as shipped: F16 linears are kept as F16 and widened to f32 inside the panel
  (`Linear` gains the element format); F32 small tensors stay F32.
- **The two MLP matrices are told apart by shape, not by name.** Converters disagree: Google's
  and the older ggml-org files name fc1 `ffn_down`, the newer ggml-org, unsloth and mradermacher
  files name it `ffn_up`.
- The geometry comes from the GGUF's `clip.vision.*` keys and is checked against the tensor
  shapes.

## Engine and serving

- `Engine::prefill` with a payload on a `gemma3` kernel set: text tokens step as before; each
  run of `image_token_index` ids (256) takes the next image of the payload (the app's
  preprocessed 3 x 896 x 896 bf16) through the tower and `Core::step_bidir_block`.
- Selection stays where it is (`_shared_select_open_engine` at load). An image on the open engine
  needs `mmproj.gguf` beside the model; without it the request is refused with a message naming
  the file and `OFLM_GEMMA_ENGINE=closed`, never sent to the closed DLL. The load and each image
  log which engine and which file ran.
- `oflm-add` installs the repo's `mmproj` (F16, else BF16) as `mmproj.gguf` and marks the entry
  `vlm`.

## Tests

- Unit (C++, no XRT): the custom window patch (`attn_apply_window`) reproduces `attn_apply` at
  every position, window and row-block it serves, and writes the requested words otherwise; the
  pass schedule covers each image token exactly once per pass-B fill and ends in natural order.
- Unit (C++, no XRT): the SigLIP tower and projector against a transformers-built fixture from a
  small random tower written as an mmproj GGUF in both naming conventions.
- Unit (Python): `oflm-add` picks the mmproj file and marks the entry; the packer's head rule.
- Manual (NPU): the 4B through `open_qwen36_cli` with an image against the fp64 GGUF replica
  with the bidirectional mask (and the causal one, to show the difference is visible), against
  transformers on the same pixels; then `oflm serve` and `oflm-test --vision` once.
