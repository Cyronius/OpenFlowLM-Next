---
name: open-hunyuan-kernels
description: Build, validate, install and serve Tencent HunYuan V1 dense models (Hy-MT2-1.8B, Hy-MT2-7B) on the open XDNA2 kernels through the dense recipe, straight from the official GGUF. Use when adding another HunYuan size, rebuilding the hy-mt2 kernel sets, installing tencent/Hy-MT2-*-GGUF with oflm-add, serving a hunyuan tag with oflm serve, or scoring Hy-MT2 translation quality.
---

# HunYuan dense (Hy-MT2) on the dense recipe

HunYuan V1 dense is Llama 3's GQA shape with two twists the recipe already handles:
the q/k RMSNorm weight multiplies **after** RoPE (`recipes.dense.QKNORM_POST_ROPE`,
`attn.h` `ATTN_QKNORM_POST`), and the NTK alpha (1000) is folded into one static RoPE
base by the spec builder. Requirement: OPEN-FAMILY-HUNYUAN in `specs/open-engine/spec.md`.

| model | hidden / ff | heads / kv | vocab (head rows) | spec | kernel set |
|---|---|---|---|---|---|
| Hy-MT2-1.8B | 2048 / 6144 | 16 / 4 | 120818 (120832) | `recipes/specs/hy-mt2-1.8b.json` | `src/xclbins/Hy-MT2-1.8B-NPU2/open_kernels` |
| Hy-MT2-7B | 4096 / 14336 | 32 / 8 | 128167 (128192) | `recipes/specs/hy-mt2-7b.json` | `src/xclbins/Hy-MT2-7B-NPU2/open_kernels` |

## Weights: the official GGUF, nothing converted

`tencent/Hy-MT2-1.8B-GGUF` ships Q4_K_M, Q6_K and Q8_0 (no Q4_0 / Q4_1), architecture
`hunyuan-dense`, tied head (no `output.weight`). The engine requantizes any of them to
q4_1 at load (`pools.cpp` `std_perm_gguf`); Q8_0 is one rounding from 8-bit and is what
was validated. The head is `token_embd.weight`, padded from 120818 to 120832 rows
(whole 64-row bands); the padded rows are zero and `real_vocab` bounds the argmax.

The spec comes from the base repo's `config.json` (what oflm-add hashes at install).
The GGUF agrees on every shape field but stores the folded base (11158840.0) and eps as
f32, so a GGUF-derived spec hashes differently; never export from it.

## Export (WSL, serial, under `dibs run build`)

```sh
source ~/ironenv142/bin/activate
export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib:$LD_LIBRARY_PATH
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/hy-mt2-1.8b.json
```

`--only dx_f32,lm_head_q4_f32,ln` builds just what a GGUF model runs, which is enough for
the harness, but the engine refuses that set: `Engine::find_kernels` wants every file the
manifest names, the q4nx-layout sets and block-prefill GEMMs included. Install the full
export. `dx`, `dx_f32` and `dx_attn` at hidden 2048 each leave 16 bytes of program memory
on their fullest core; change anything on those cores and check `program memory:` in the log.

## Slice parity (no engine needed)

`model/make_decode.py` reads `model.gguf` through `model/gguf_src.py`, which reproduces
the engine's pool bytes (the Q8_0 requant law, the padded tied head) and reads weights
back with bf16 scales, as the kernel does. Start from the BOS id, 120000, not the 7B's
127958 (out of range here).

```sh
python open_kernels/model/make_decode.py --model-dir <installed model dir> --layers 8 --tokens 8 \
    --token 120000 --out <out> --pool-dir <pools>
open_kernels/harness/out/run_kernel.exe <out>/run_decode.cfg      # dibs run npu
python open_kernels/model/compare_decode.py --tokens 8 --out <out>
```

The replica re-quantizes per call: 8 layers x 8 tokens takes ~10 minutes on the host.

## Install and serve

```sh
oflm-add tencent/Hy-MT2-1.8B-GGUF          # or a local dir holding the .gguf, README.md and the base repo's files
```

- Family: the repo name (`hy-mt`), the GGUF arch (`hunyuan-dense`) and `model_type`
  (`hunyuan_v1_dense`) all map to `hunyuan`; the derived tag is `hy-mt2:1.8b`.
- Hy-MT2's `generation_config.json` has `eos_token_id: 120020` as an int; oflm-add writes
  it into `tokenizer_config.json` as `[120020]`, without which the app exits at load.
- With no official registry entry, the GGUF gate cannot see a kernel set and falls back
  to safetensors; pass `--open-kernels <set>` (a set with the nested `gguf` manifest).
- Picking the file: oflm-add prefers Q4_K over Q8_0, so the remote install takes
  `Hy-MT2-1.8B-Q4_K_M.gguf` (Q4_K plus Q6_K tensors, both requantized at load; its
  translation quality is not scored). Pass a local dir with only the Q8_0 file to get
  the validated one.
- `oflm serve` routes `details.family: hunyuan` to `HunYuan` (`modeling_hunyuan.cpp`):
  open kernels only, the model's own `chat_template.jinja`, model-card sampling
  (temperature 0.7, top_p 0.6, top_k 20, repetition 1.05).

## Translation quality

`utilities/quant-compare/flores_chrf.py` scores FLORES-200 devtest with chrF through any
OpenAI-compatible server, with the model card's "Default Translation" prompt (Chinese
prompt for a Chinese target). Run the same sentences through llama-server on the Q8_0 file
(`C:\tools\llama-prism\vulkan`, build 10754, knows `hunyuan-dense`) for the reference.
FLORES+ on Hugging Face is gated; the public FLORES-200 tarball
(`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`) has the same devtest files.

## Validated 2026-10-16 (Hy-MT2-1.8B, Q8_0 GGUF)

- 8-layer slice, 8 positions: logits corr >= 0.99995, argmax and top-5 equal everywhere.
- `oflm serve`: all 32 layers resident; a German translation matched llama.cpp's on the
  same file token for token; `oflm-test --llm --gen-lim 256` 5/5.
- FLORES chrF, q4_1 on the NPU vs llama.cpp Q8_0: en->zh 38.63 / 39.52 (100 sentences),
  en->de 58.47 / 59.35 (67). About 0.9 lower, inside noise at this size.
- Not run: the 7B from its GGUF, the Q4_K_M / Q6_K files, measured speed.
