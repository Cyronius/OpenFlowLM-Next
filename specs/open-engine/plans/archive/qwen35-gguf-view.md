# Plan: Qwen3.5-family GGUFs on the open kernels, through a q4nx view

**Status:** implemented on `feat/qwen35-gguf` (2026-10-17); validated on MiniCPM-V-4.7-1B on the NPU and through `oflm serve` / `oflm-test --llm`; archived.
**Spec impact:** two new requirements, `OPEN-GGUF-Q4NX-VIEW` and `OPEN-GGUF-QWEN35-SPEC`, and a
Result line under `OPEN-FAMILY-QWEN35`. No existing requirement changes: the dense families' f32
GGUF path (#180) and every q4nx container behave as before.

## Why

The qwen35 family (Qwen3.5 / 3.6 / 3.8 dense, MiniCPM-V-4.7, the 0.8B embedder and reranker, the
27B blend) had no GGUF route. #180's route for the dense families needs a second export of every
design with f32 scale planes and a nested manifest. The memory rule (no f32 in device memory)
retires that for new families: the kernels already narrow the scales to bf16, so narrowing at pack
time is the same rounding (`gguf-scale-width.md`).

## Design

The qwen35 kernel set keeps its one plan, written against the q4nx container. The engine serves the
GGUF to that plan as the container it expects (`GgufQ4nxView`), and the NumPy packer and the fp64
replica read the same view (`open_kernels/model/gguf_q4nx.py`). What the view does is in
`OPEN-GGUF-Q4NX-VIEW`. The block prefill route's host stages read through the same seam
(`WeightFile::chunk_bytes`, no `Q4nxFile` casts left in the pack ops).

Q4_1 and Q4_0 files pack exactly into the 5120-byte chunk (codes untouched, d / m narrowed once).
Q8_0 stays q8 (8704-byte chunks), streamed at q8 where the derived quant map says so and
re-quantized to q4_1 by `requant_q4_1_chunks` otherwise. Q4_K is transcoded (its per-32 scale and
min carried over, rounded to bf16, the codes kept). Everything else is re-quantized to q4_1 by
`requant_q4_1`'s law. The head is always the q8 head (the family's `lm_head_q8` kernel).

## The spec-hash finding

`ModelSpec.spec_hash()` covers `real_vocab`, and `oflm-add` links a kernel set by that hash.
MiniCPM-V-4.7-1B has 248094 ids; the Qwen3.5-0.8B set was derived with 248070. Every kernel is the
same. Taken here: a spec file of its own (`recipes/specs/minicpm-v-4.7-1b.json`), so the hash rule
and every existing link stay as they are. Leaving `real_vocab` out of the kernel-match hash would
let shape-identical fine-tunes link without a new export, but it changes linking for every family,
so it is the maintainer's call.

## Found at install and serve time

- `oflm-add`'s quant-name check refused the Q4_1 file over its Q6_K head (llama.cpp's default);
  for a view architecture the head is now outside that check.
- The Qwen3.5 adapter sized its sampler from `config.json`'s top-level `vocab_size`, which a VLM
  wrapper's config (MiniCPM-V's) keeps under `text_config`; the adapter now reads the nested value
  when the top level has none, as the omni adapter does.
- The kernel set is named after the directory `oflm-add` installs the model as
  (`MiniCPM-V-4.7-1B-gguf`); `extra` is not in the spec hash.

## Not in this plan

Vision (the mmproj GGUF and MiniCPM-V's own tower), the MoE line (`qwen35moe`: expert tensors),
Q5_K (unsloth's Qwen3.5 Q4_K_M files use it for qkv and out_proj), and the release export list
(the MiniCPM set is exported locally only).
