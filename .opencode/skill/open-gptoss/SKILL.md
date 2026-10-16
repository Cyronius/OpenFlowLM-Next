---
name: open-gptoss
description: GPT-OSS 20B on the open kernels - where the design stands (a proposed expert layout waiting on the user's format and deal decisions, no kernels yet), the MXFP4 GGUF it loads from, the host prototype of the expert layout and how to check it, and how the MXFP4-vs-q4_1 quality cost was measured. Use when starting the gpt-oss build, changing the expert tile or the core deal, re-measuring the re-quantization cost, or when someone asks why gpt-oss still raises NotImplementedError.
---

# GPT-OSS 20B: a proposed layout, not yet kernels

`families.family_module("gptoss")` still raises `NotImplementedError` on purpose
(OPEN-FAMILY-GPTOSS): no kernel computes this family yet. What exists is the
arithmetic (`open_kernels/model/replica_gptoss.py`, OPEN-GPTOSS-FFN-REF) and, since
2026-10-16, a host prototype of the expert layout the design spike recommends
(OPEN-GPTOSS-EXPERT-LAYOUT, `open_kernels/model/gptoss_layout.py`). The recommendation,
the decisions it leaves to the user and the ordered work are in
`specs/open-engine/plans/gptoss-expert-layout.md`. Read that first.

## Source weights

`ggml-org/gpt-oss-20b-GGUF` / `gpt-oss-20b-MXFP4.gguf`, 12,109,566,624 bytes,
sha256 `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`. Experts are
MXFP4 (type 39, 17-byte blocks: one E8M0 byte, then 16 bytes holding element j in the
low nibble of byte j and j + 16 in its high nibble), carried over from OpenAI without
re-quantizing. Attention, embedding and head are Q8_0; router, biases, norms and sinks
f32. Expert tensors are `[2880 K, 2880 rows, 32 experts]` with K fastest; the biases
`[2880, 32]`. Gate and up are already split (HF's interleave is undone by the converter).

The older `GPT-OSS-20B-NPU2` q4nx container (2560-byte chunks, supertile raster,
OPEN-PACK-CHUNK-FUSE / OPEN-PACK-EXPERT-ORDER / OPEN-QUANT-MXFP4) is the legacy path;
under the weights rule nothing new reads it.

## The proposed layout in one breath

Not decided: the expert format (MXFP4, q4_1 or exact int8) and the deal (uneven or
padded) are the user's calls. As recommended, MXFP4 stays MXFP4. Tile = 32 rows x 288
columns in 5120 bytes (nibble planes at 0 and 2304 in gemv_q4.h's interleave, E8M0 at
4608, the 32 rows' bf16 biases at 4928 in k-tile 0 of each band half). 2880 = 10 tiles,
so no K padding, and two tiles are the 10,240-byte w element q4_1 already uses. 45 bands
of 64 rows dealt 6,6,6,6,6,5,5,5. One contiguous block per expert (gate and up
interleaved by band, then down). Two phases per token: every slot's gate|up + clamped
SwiGLU, one barrier, every slot's down.

```
python -I open_kernels/model/gptoss_layout.py                        # geometry + stream check
python -I open_kernels/model/gptoss_layout.py --gguf <gpt-oss-20b-MXFP4.gguf> --layer 11 --token 3417
python -m pytest -q specs/open-engine/tests/test_gptoss_layout.py    # 22 tests, no GGUF needed
```

The `--gguf` slice packs one layer's four routed experts into tiles, reads them back
(must be exact) and compares the layout's block output with `replica_gptoss` (2-3e-3 at
layer 0 token 13225, layers 11 and 23 token 3417, on 2026-10-16).

`check()` is a hand-written model of the core program and the host sequence, not the
stub-IRON tracer (no IRON design exists yet). It catches a mismatch between the two,
not an assumption they share, and no DMA timing.

## Measuring a re-quantization's cost (behind the MXFP4 recommendation)

- Use a llama.cpp build that knows gpt-oss (b10754, Vulkan, was used). **Not** one whose
  folder carries `ggml-xdna.dll`: llama.cpp loads that NPU backend at start-up.
- Expert-only copy: `llama-quantize.exe --allow-requantize --pure --tensor-type
  ffn_gate_exps=q4_1 --tensor-type ffn_up_exps=q4_1 --tensor-type ffn_down_exps=q4_1 in.gguf
  out.gguf Q8_0 8` (~2 min; 13.9 GB out). Then `utilities/quant-compare/mxfp4_requant.py diff
  in out` must say the other 387 tensors are byte-identical.
- `llama-perplexity -c 512 --chunks 20 -ngl 99 --kl-divergence-base base.kld` on the
  original, then `--kl-divergence` on the copy, under `dibs run gpu`. The base logits file
  is ~2 GB for 20 chunks (201,088-entry vocabulary).
- **Do not judge by perplexity.** Re-quantized experts LOWER raw-text perplexity on both
  backends (393 -> 358 wikitext, 3.68 -> 3.30 code). Why was not measured (a chat-trained
  model flattened by noise is a guess). Read mean KLD and "Same top p": 0.200 / 57.6%
  (wikitext), 0.182 / 87.1% (code and docs).
- `mxfp4_requant.py err` gives the weight error without llama.cpp: 9-10.5% RMS per expert
  matrix, 0.1-0.6% of blocks exact.

## Traps

- **The MXFP4 decode is the expensive part, not the bytes.** `kernel_remarks.py` on
  `utilities/aie-probes/gemv_mx4_probe.cc` (the proposed tile's GEMV body; the command is in
  its header, run under WSL) says 289 cycles per 32-column block against q4_1's 59; eight
  cores need roughly 120-220 to keep pace with the weight stream. Sorting codes by value
  (`-DMX4_DECODE=2`) made it worse (333); the high-nibble shift costs only 24. Plan on
  spreading the MoE half over rows 4-5. Measure on the NPU with the `GEMV_NULL` ablation
  before building on it.
- **More cores need a memory-tile split.** One shim stream per core runs out of the shim's 16
  channels past 15 cores, and `check()` refuses 16 and 24 as modeled. `gemm_q4`'s fan-out
  mode and `moe_batch` already split a column's stream at its memory tile; a split gives
  every core of a column the same element count, so the cores of one column need equal band
  counts (45 bands at 24 cores: two each in seven columns, one each in the eighth).
- **The exact int8 fallback is 2.4x the MXFP4 tiles' bytes** (25.5 GB resident): `gemv_q8.h`
  half-tiles are 5120 B per 16 x 256, not the container's 8704 B per 32 x 256. It also starts
  its accumulator at zero, so the biases need a seeding change.
- `dense.cores_for` gives this model ONE core (2880 shares no factor with 4096 / 512 at 64
  rows per band); the dense recipe's even split is the reason the layout deals bands
  unevenly instead.
- `pools.cpp std_perm_gguf` takes the source row stride from the pool's `in_dim`; q/k/v at
  K = 2880 into a 3072-wide pool need a separate source width first.
- A whole-layer design is ONE hardware context with two instruction streams (like
  `ax0`/`ax1`); two contexts per layer would pay ~2.5 ms per switch.
- The existing `ax1` main core is 12,032 of 16,384 bytes of program memory with one GEMV
  tile; the gpt-oss core carries two (q4_1 attention, MXFP4 experts). Measure on the first
  build (`llvm-size -A` on `final.prj/elfs_main_core_*`).
