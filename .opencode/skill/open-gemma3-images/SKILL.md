---
name: open-gemma3-images
description: Run images through Gemma 3 4B (gemma3:4b, MedGemma, TranslateGemma) on the open engine -- the SigLIP tower read from the model's mmproj GGUF on the host, and the image block's both-ways attention through the decode kernel, layer by layer. Use when an image request to a Gemma 3 model fails or answers as if blind, when bringing up another model with an mmproj (MiniCPM-V shares the tower's shape), or when touching the attnpos patch or the image block's schedule.
---

# Gemma 3 images on the open engine (OPEN-VISION-GEMMA3)

## What runs where

- **The tower** (`src/open_qwen36/vision/vit.cpp`, `VitFamily::Siglip`): host CPU, read from
  `<model dir>/mmproj.gguf` (llama.cpp's `clip` arch, projector `gemma3`). 27 blocks x 1152, 16 heads
  of 72, MLP 4304, 896 px in 14 px patches = 4096 patches, a learned 4096-row position table, no
  rotary. Then the projector: 4 x 4 average pool to 256 tokens, RMSNorm with the stored `1 + w`, and
  `mm.input_projection` ([1152, 2560], `y = x W`). ~0.85 GB resident as f16.
- **The image block** (`Core::step_bidir_block`, schedule in `bidir_block.hpp`): the 256 rows go
  through the ordinary decode kernel (`dx` / `dx_local`), layer-major, twice per token -- pass A writes
  each token's K/V, pass B reads all of them. No kernel change; ~2x a causal token-by-token prefill of
  the block, plus the tower.
- **Text** before and after: `Core::step` as always.

## The traps

- **fc1 / fc2 names.** Google's and the older ggml-org mmprojs name SigLIP's first MLP matrix
  `ffn_down` (it maps 1152 -> 4304); the newer ggml-org, unsloth and mradermacher files name it
  `ffn_up`. The loader decides by shape. Reading by name gives a tower that runs and is wrong.
- **Google's mmproj is not the QAT checkpoint's.** In `google/gemma-3-4b-it-qat-q4_0-gguf` the tower
  matches `google/gemma-3-4b-it-qat-q4_0-unquantized` but the projector differs (soft-embedding norm up
  to 0.008, projection up to 5e-4). ggml-org's `gemma-3-4b-it-qat-GGUF` mmproj matches it to f16
  rounding. Compare a port against the checkpoint its file came from.
- **Google's own 4B GGUF does not load today.** Its head (the tied F16 embedding) has 262144 rows;
  the kernel set and the HF config say 262208 (64 padding rows the converter trimmed), so the
  config check derives vocab 262144 and the packer refuses the short head. ggml-org's QAT Q4_0 file
  (Q8_0 head, 262208 rows) loads. The fix is a head rule that accepts the file's own vocabulary.
- **The chat template.** The server hands images to the model class as `message['images']`
  (`rest_handler.cpp`). Google's template reads image content parts instead, so a GGUF install
  with the base repo's template sends no image tokens. `oflm-add` writes the official entry's
  template (which emits `<start_of_image>`, 256 `<image_soft_token>`, `<end_of_image>`) as
  `chat_template.jinja` when it installs an mmproj.
- **The bf16 transformers tower drifts** (per-token cosine down to 0.92 against fp32 over 27 blocks).
  Use the fp32 tower (`gemma3_vision_ref.py tower`) as the reference, and inject its rows into the
  bf16 LM (`logits --features`).
- **`python -I` hides the user site-packages** on this box (torch, transformers, gguf live there).
  Run the references with `python -E -P`.
- **The record counts.** Each pass-B dispatch reads `[s_t, p0 + n - 1)`: the prefix inside the
  query's own window, then every image row but its own. The position record of row `p0 + t` carries
  that `valid` while the block runs; `step_bidir_block` restores the standard counts after.

## Validate (smoke, ~15 min plus the build)

```sh
# references (CPU)
python -E -P open_kernels/model/gemma3_vision_ref.py tower --fixture FX --hf <QAT ckpt> --image <896 px png>
src/open_qwen36/out/vit_test.exe --siglip <mmproj.gguf> FX
python -E -P open_kernels/model/replica_gemma3_image.py --gguf <model.gguf> --spec open_kernels/recipes/specs/gemma3-4b.json \
  --ids-file IDS --rows FX/ref.bin --mask bidir --out REF_BIDIR      # and --mask causal
# NPU
open_qwen36_cli --model <dir: model.gguf + config.json> --kernels <4B set> --ids-file IDS --embeds FX/ref.bin \
  --prefill-logits --dump-logits OUT/y --max-tokens 8      # add --causal-image for the A/B
```

Unit tests: `ctest -R OPEN-VISION-GEMMA3` (the schedule, and the tower against two tiny
transformers-built mmprojs, one per naming convention); `pytest utilities/oflm-add/tests`.
