---
name: open-qwen3-embedding
description: Serve Qwen3-Embedding (/v1/embeddings) and Qwen3-Reranker (/v1/rerank), 0.6B and 4B, on the open dense kernels, read straight from their GGUFs. Use when installing any of them, rebuilding their kernel sets, adding the 8B size, or debugging an LLM-backed embedder or reranker (OpenLlmEmbedding, Core::prefill_hidden).
---

# Qwen3-Embedding and Qwen3-Reranker on the open kernels

At 0.6B both models are the Qwen3-0.6B network (28 layers, hidden 1024, 16/8 heads at 128, ff
3072) with a 151669-row vocabulary and a tied head; at 4B, the Qwen3-4B network (below). They need
no kernel of their own: the job is a hidden state instead of logits.

- Engine: `Core::prefill_hidden(ids)` (OPEN-HIDDEN-STATE) runs the prompt from position 0, then the
  tail up to the head (the final norm writes `hn`), and returns `hn` of the last token as f32.
  `CoreConfig::lm_head = false` never packs, loads or runs the head.
- Server: `OpenLlmEmbedding` (`src/include/AutoEmbeddingModel/open_llm_embedding.hpp`), two modes
  chosen by the tag table in `all_embedding_model.hpp`. The text, id, prompt and score rules are
  `llm_conventions.hpp`; the wire rules (`/v1/rerank`'s body, `dimensions`) are `openai_compat.hpp`.
- Specs: `specs/open-engine/spec.md` OPEN-HIDDEN-STATE, `specs/server-api/spec.md`
  SERVER-EMBED-QWEN3 and SERVER-RERANK; plan (done) `specs/open-engine/plans/archive/qwen3-embedding.md`.

## The kernel set

One set serves both models: every GGUF of either and both `config.json`s derive spec hash
`sha256:82824db3a742...` (`open_kernels/recipes/specs/qwen3-embedding-0.6b.json`). It is named
`Qwen3-Embedding-0.6B-NPU2`, not `Qwen3-0.6B-NPU2`: the chat model's `config.json` says vocabulary
151936, and a set under its name would make `qwen3:0.6b` load against the wrong head and fail.

```sh
# WSL, under dibs build; up to an hour on a busy box (both layouts: q4nx and GGUF f32-scale)
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/qwen3-embedding-0.6b.json
```

At 4B the two models stop sharing a set: the embedder's vocabulary is 151665, the reranker's
151669, so each has its own spec (`qwen3-embedding-4b.json`, `qwen3-reranker-4b.json`) and set
(`Qwen3-Embedding-4B-NPU2`, `Qwen3-Reranker-4B-NPU2`). The network is Qwen3-4B's (36 layers,
hidden 2560, 32/8 heads, ff 9728, tied head), so no kernel point is new.

The two 4B specs compile the same kernels (both heads pad to 151680 rows; nothing else reads the
vocabulary), so after a full export of one, the other can be written from the same build dirs:

```sh
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/qwen3-embedding-4b.json   # ~80 min
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/qwen3-reranker-4b.json --no-build
```

Check it first (the 2026-10-17 run did): `F.builds()` and the generated kernel sources equal under
both specs, and the MLIR of `dense/dx.py` (also with `GEMV_SCALES_F32=1`) and `dense/dx_attn.py`
byte-identical (`CallableDesign.as_mlir()`).

## Install and serve

```sh
python utilities/oflm-add/oflm-add.py mradermacher/Qwen3-Embedding-0.6B-i1-GGUF --tag qwen3-embedding:0.6b \
    --open-kernels <set> --no-xclbin
python utilities/oflm-add/oflm-add.py mradermacher/Qwen3-Reranker-0.6B-i1-GGUF --tag qwen3-reranker:0.6b \
    --open-kernels <set> --no-xclbin
oflm serve --embed 1 --embeddingmodel qwen3-embedding:0.6b --rerankmodel qwen3-reranker:0.6b
oflm-test --embedding --model qwen3-embedding:0.6b
oflm-test --rerank --model qwen3-reranker:0.6b
```

The 4B pair installs the same way from `mradermacher/Qwen3-{Embedding,Reranker}-4B-i1-GGUF` with
`--tag qwen3-{embedding,reranker}:4b` and each model's own set. A new size needs its tag line in
`all_embedding_model.hpp` (the table routes by tag), so a rebuild of oflm.

These are imatrix `i1-Q4_1` files: every matmul packs exactly. Qwen's own embedder GGUF
(`Qwen/Qwen3-Embedding-0.6B-GGUF`, Q8_0 and F16 only) and ggml-org's reranker Q8_0 install and run
too, but the engine requantizes Q8_0 to plain min/max q4_1, and on the embedder that moved the
model card's similarity matrix by up to 0.13 (the i1-Q4_1 file: 0.03).

`--tag` is needed: `derive_tag` reads mradermacher's `-i1` (and ggml-org's `-Q8_0`) as part of the
size and would register `qwen3-embedding:0.6b-i1`.

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
- **The official embedder GGUF's README names `Qwen/Qwen3-0.6B-Base`**, whose `config.json` has the
  chat vocabulary (151936). `oflm-add` takes no `config.json` from the base chain for these two
  families; the engine derives the checked fields from the GGUF.
