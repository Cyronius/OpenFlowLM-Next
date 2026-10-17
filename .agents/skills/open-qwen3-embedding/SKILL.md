---
name: open-qwen3-embedding
description: Serve Qwen3-Embedding-0.6B (/v1/embeddings) and Qwen3-Reranker-0.6B (/v1/rerank) on the open dense kernels, read straight from their GGUFs. Use when installing either model, rebuilding the Qwen3-Embedding-0.6B-NPU2 kernel set, adding the 4B/8B sizes, or debugging an LLM-backed embedder or reranker (OpenLlmEmbedding, Core::prefill_hidden).
---

# Qwen3-Embedding and Qwen3-Reranker on the open kernels

Both models are the Qwen3-0.6B network (28 layers, hidden 1024, 16/8 heads at 128, ff 3072) with a
151669-row vocabulary and a tied head. They need no kernel of their own: the job is a hidden state
instead of logits.

- Engine: `Core::prefill_hidden(ids)` (OPEN-HIDDEN-STATE) runs the prompt from position 0, then the
  tail up to the head (the final norm writes `hn`), and returns `hn` of the last token as f32.
  `CoreConfig::lm_head = false` never packs, loads or runs the head.
- Server: `OpenLlmEmbedding` (`src/include/AutoEmbeddingModel/open_llm_embedding.hpp`), two modes
  chosen by the tag table in `all_embedding_model.hpp`. The text, id, prompt and score rules are
  `llm_conventions.hpp`; the wire rules (`/v1/rerank`'s body, `dimensions`) are `openai_compat.hpp`.
- Specs: `specs/open-engine/spec.md` OPEN-HIDDEN-STATE, `specs/server-api/spec.md`
  SERVER-EMBED-QWEN3 and SERVER-RERANK; plan `specs/open-engine/plans/qwen3-embedding.md`.

## The kernel set

One set serves both models: every GGUF of either and both `config.json`s derive spec hash
`sha256:82824db3a742...` (`open_kernels/recipes/specs/qwen3-embedding-0.6b.json`). It is named
`Qwen3-Embedding-0.6B-NPU2`, not `Qwen3-0.6B-NPU2`: the chat model's `config.json` says vocabulary
151936, and a set under its name would make `qwen3:0.6b` load against the wrong head and fail.

```sh
# WSL, under dibs build; about 20 minutes, most of it the q4nx route's attention blocks
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/qwen3-embedding-0.6b.json
```

## Install and serve

```sh
python utilities/oflm-add/oflm-add.py Qwen/Qwen3-Embedding-0.6B-GGUF --open-kernels <set> --no-xclbin
python utilities/oflm-add/oflm-add.py ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF --tag qwen3-reranker:0.6b \
    --open-kernels <set> --no-xclbin
oflm serve --embed 1 --embeddingmodel qwen3-embedding:0.6b --rerankmodel qwen3-reranker:0.6b
```

`--tag` is needed for ggml-org's repo: `derive_tag` reads its `-Q8_0` (and mradermacher's `-i1`)
as part of the size and would register `qwen3-reranker:0.6b-q8`.

## Traps

- **The head pack refuses these GGUFs** on a plain `Core`: the plan packs 151680 rows (the
  vocabulary rounded up to a 64-row band) from a 151669-row tied `token_embd`, and `embed_row`
  throws for row 151669. `lm_head = false` is the fix for an embedder or reranker; a chat model
  with such a vocabulary would need the padded-row rule `feat/hy-mt2-1.8b` / `feat/gemma3-1b` add.
- **The GGUF's epsilon is float32.** Read raw it is 9.99999997e-07, and a spec derived from it hashes
  differently from the `config.json` one (1e-06). Read it as the shortest decimal that is the same
  float32 (`float(str(np.float32(v)))`) before `ModelSpec.from_gguf_metadata`.
- **EOS.** The embedder pools at the end-of-text id 151643, which its `tokenizer.json` appends
  through a post-processor. tokenizers-cpp's `Encode` does not apply it, so the backend appends the
  GGUF's `tokenizer.ggml.eos_token_id` when `tokenizer.ggml.add_eos_token` is set. Some community
  files (PeterAM4's, Mungert's) carry `add_eos_token` false or no pooling metadata: avoid them.
- **The reranker's "yes" / "no" rows** come from `token_embd` (the HF model ties its head), read
  exactly on the host. llama.cpp-converted rerankers also carry a 2-row `cls.output.weight`; it is
  not used.
- **Prefill is one token per step** for any GGUF kernel set (the block route is off for f32-scale
  pools), so a long passage costs a decode step per token.
- `oflm-add`'s GGUF check refuses a file whose `token_embd` is Q6_K beside Q4_1 matmuls (the
  imatrix `i1-Q4_1` files), which is why the Q8_0 files are the install default.
