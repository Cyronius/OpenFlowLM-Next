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

Embedder: `Qwen/Qwen3-Embedding-0.6B-GGUF`, `Qwen3-Embedding-0.6B-Q8_0.gguf` (Qwen's own file; its
only other file is F16). Q8_0 is requantized to q4_1 at load. Reranker: `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`
(llama.cpp's organisation; there is no official reranker GGUF), same treatment. The CPU references
also score mradermacher's imatrix `i1-Q4_1` files, which pack exactly; see the report for which is
better and why the Q8_0 files are the default.
