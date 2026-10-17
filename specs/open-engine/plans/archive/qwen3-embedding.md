# Qwen3-Embedding and Qwen3-Reranker 0.6B on the open engine, from GGUF

2026-10-16. Item 7 of the round-2 model push. Branch `feat/qwen3-embedding`, based on
upstream/main 7637e655. Design background: the 2026-10-13 evaluation
(`.claude/plans/qwen3-embedding-reranker.md` in the main checkout), checked against 7637e655 below.

## What changes

The open engine learns one new job: run a prompt and return the last token's final-norm hidden
state, without the lm_head. The server learns two uses of it:

- `/v1/embeddings` for Qwen3-Embedding-0.6B: last-token pooling, the instruction on queries,
  `dimensions` (Matryoshka) truncation;
- a new `POST /v1/rerank` for Qwen3-Reranker-0.6B: the card's prompt, the yes/no softmax.

Both models are the Qwen3-0.6B network with a 151669-row vocabulary. One kernel set, exported from
`open_kernels/recipes/specs/qwen3-embedding-0.6b.json`, serves both: every GGUF surveyed and both
models' `config.json` derive the same spec hash.

## The design claims, checked against 7637e655

- **Block prefill is still off for GGUF pools.** `recipes/dense.py programs()` emits no
  `gemm_block` for the `q4_1_f32` spec, so a GGUF prefills one token at a time. True today; the
  embedder therefore pays one decode step per prompt token.
- **What PR #193 would add.** Nothing for a GGUF model as it stands: the branch predates GGUF-direct,
  and its 8-bit route (`gemm: "dit"`) packs its bfp16 copy from q4nx chunks (`bfp16_dit` reads
  `q4_source`). But that copy is built from the weight values, not from the q4 pool's chunk law, so
  a GGUF reader for that one pack op (dequantize the file's rows, then `bfp16_dit_pack`) would give
  GGUF models a block prefill without the f32-scale GEMM twins. Not done here; it would sit behind
  #193's opt-in, since the 8-bit copy costs memory. (Read from the `perf/dense-prefill-gemm8`
  worktree; not built.)
- **`AutoEmbeddingModel` and `all_embedding_model.hpp`.** As described: a tag table naming a
  backend per tag, and one embedding model per server. There was no reranker anywhere.
- **Risk 3 (the 151669-row vocabulary), checked in Python.** The spec hash does not refuse the
  file: the GGUF and `config.json` derivations agree once the GGUF's float32 epsilon is read as the
  decimal it was written from (1e-06, not 9.99999997e-07). `derive_config` agrees with
  `hf_config_check`. The head pack does refuse: the plan packs 151680 rows (the vocabulary rounded
  up to a 64-row band) from a tied `token_embd` of 151669, and `embed_row` throws for row 151669.
  The fix taken: the engine opened for hidden states never packs or loads the head.

## Requirements

New:

- **OPEN-HIDDEN-STATE** (`specs/open-engine/spec.md`), manual. `Core::prefill_hidden` returns the
  last token's final-norm hidden state with no lm_head dispatch, from a reset state.
- **SERVER-EMBED-QWEN3** (`specs/server-api/spec.md`), test for the text, ids and `dimensions`
  rules, manual for the vectors.
- **SERVER-RERANK** (`specs/server-api/spec.md`), test for the body, prompt, score and order,
  manual for the scores.

Modified: none. `/v1/embeddings` gains two behaviours that reach the existing backends: a request
naming no task gets the backend's own default (unchanged, `task_query`, for every existing one),
and `dimensions` is refused by a backend that cannot shorten its vectors, where it used to be
ignored. Both are written into SERVER-EMBED-QWEN3.

Removed: none.

## Weights

Default: mradermacher's imatrix `i1-Q4_1` files of both models
(`mradermacher/Qwen3-Embedding-0.6B-i1-GGUF`, `mradermacher/Qwen3-Reranker-0.6B-i1-GGUF`). Every
matmul is Q4_1, which the engine packs exactly; `token_embd` is Q6_K, read exactly on the host;
the metadata is right (pooling last, `add_eos_token` true on the embedder).

Also tried: Qwen's own embedder file (`Qwen/Qwen3-Embedding-0.6B-GGUF`, Q8_0; its only other file
is F16) and ggml-org's reranker Q8_0 (llama.cpp's organisation; no official reranker GGUF exists).
Both load, but the engine requantizes Q8_0 to plain min/max q4_1, and on the embedder that moved the
model card's similarity matrix by up to 0.13 against 0.03 for the i1-Q4_1 file (both on the NPU).
`oflm-add` had refused the i1 files for their Q6_K `token_embd`; it now accepts them.
Several community files carry wrong metadata (`add_eos_token` false, no pooling type) and are
not usable: PeterAM4's and Mungert's.

## Results, 2026-10-17 (risk 2: how far the 4-bit weights sit from fp32)

The engine reproduces each file: NPU vectors against the same weights on the CPU have cosine
>= 0.99996 on every prompt (OPEN-HIDDEN-STATE). What separates the served model from fp32 is the
file. On a 50-query, 200-passage SciFact subset (passages cut to 128 tokens), with the fp32
transformers model as the yardstick:

- i1-Q4_1, served exactly: passage cosine to fp32 mean 0.979 (min 0.971); nDCG@10 0.881 against
  fp32's 0.891; top-10 overlap with fp32 0.89; same top-1 on 92% of queries.
- Qwen's Q8_0, requantized to plain q4_1 on load: cosine mean 0.944 (min 0.929); nDCG@10 0.901;
  top-10 overlap 0.82; same top-1 on 88%.

The two nDCG figures straddle fp32's by about 0.01 each way, which 50 queries cannot separate; by
every measure of closeness to fp32 the i1 file is ahead, hence the default. The card's similarity
matrix moves by 0.026 with i1-Q4_1, so the task's "within 0.02 of fp32" bar is missed by the
weights themselves (the CPU gives the same 0.026). Meeting it needs more than 4 bits on the
matmuls: a Q8_0 GGUF read as 8-bit (the engine has a q8 GEMV for the q4nx container, not yet for
GGUF), or a better requantizer than min/max for Q8_0 sources. Neither is done here.

The reranker is barely moved: on the card's four pairs every i1 score is within 0.001 of fp32's,
and the relevant document wins each query by a logit margin of more than 6.

## Server smoke, 2026-10-17

`oflm serve --embed 1 --embeddingmodel qwen3-embedding:0.6b --rerankmodel qwen3-reranker:0.6b`,
both from the i1-Q4_1 files: every SERVER-EMBED-QWEN3 and SERVER-RERANK check against a live server
holds except the embedder's "within 0.02 of fp32" bar, which the weights miss (0.026, as on the
CPU). `oflm-test --embedding --model qwen3-embedding:0.6b` passes its ten checks (E8 skips: its
bundled vectors are embeddinggemma's). Results are in the two requirements.

## The 4B sizes (2026-10-17)

Qwen3-Embedding-4B and Qwen3-Reranker-4B are the Qwen3-4B network (36 layers, hidden 2560, 32/8
heads at 128, ff 9728), both with a tied head. Unlike the 0.6B pair they differ in vocabulary: the
embedder's is 151665, the reranker's 151669, so each gets its own spec and kernel set
(`qwen3-embedding-4b.json` -> `Qwen3-Embedding-4B-NPU2`, `qwen3-reranker-4b.json` ->
`Qwen3-Reranker-4B-NPU2`). Each GGUF derives the same hash as its `config.json`. Sources:
mradermacher's `i1-Q4_1` files again (every matmul Q4_1, `token_embd` Q6_K, pooling metadata right);
Qwen's own 4B embedder GGUF is Q4_K_M, Q6_K, Q8_0 or F16, all requantized on load.

Requirements: no new ones. SERVER-EMBED-QWEN3 and SERVER-RERANK gain the tags
`qwen3-embedding:4b` and `qwen3-reranker:4b` (one line each in the tag table); OPEN-HIDDEN-STATE's
procedure is run once per kernel set. The 4B CPU references skip SciFact (about 20 minutes per
fp32 pass on this box) and keep the card, the ten fixed prompts and the reranker pairs.

The two specs build the same kernels. They differ only in `vocab` / `real_vocab`; both round the
head to 151680 rows, the 42 kernel sets (designs, build dirs, env) are identical, the generated
kernel sources are identical, and the MLIR of every design that reads the spec (`dx.py`, its f32
twin, `dx_attn.py`) is byte-identical under both. So the reranker's set was written by
`export_qwen36_kernels.py --no-build` from the embedder export's build dirs, in 14 s instead of
about 80 minutes. A plain export rebuilds it whenever wanted.

The fp32 references reproduce both cards: the embedder's similarity matrix to 1e-6, the
reranker's logit differences (6.32 and -14.42 for the first query) to within 0.12 of the printed
6.44 and -14.38. The i1-Q4_1 weights are closer to fp32 at 4B than at 0.6B: on the CPU the card
matrix moves by 0.0195, inside the 0.02 bar; NPU cosine to fp32 is 0.979 to 0.989.

On the NPU and through `oflm serve` the 4B pair passes the same checks as the 0.6B pair: the
embedder sits 0.0205 from fp32 on the card (0.0009 from the same weights on the CPU), the reranker
within 0.014; `oflm-test --embedding` and `--rerank` pass. Results are in the three requirements.
The 8B sizes were not started.
