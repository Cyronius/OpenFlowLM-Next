# qwen3-embed-ref

CPU references and NPU checks for Qwen3-Embedding and Qwen3-Reranker on the open engine. This is
the tool the manual procedures of OPEN-HIDDEN-STATE (`specs/open-engine/spec.md`) and
SERVER-EMBED-QWEN3 / SERVER-RERANK (`specs/server-api/spec.md`) run.

It needs `torch`, `transformers`, `gguf` and `numpy`, and the official safetensors of the model
(`Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B`, or the 4B ones) next to the GGUF the engine
loads. Run it with `python -P` so a stray file in the working directory cannot shadow a module.

```sh
U=utilities/qwen3-embed-ref/qwen3_ref.py
A="--embedder <hf-embedding> --reranker <hf-reranker> --scifact <beir-scifact> --ref <ref dir>"
python -P $U $A texts                                   # the card's texts, ten prompts of 5-500 tokens, a SciFact subset
python -P $U $A embed fp32                              # transformers fp32, last-token hidden state
python -P $U $A embed gguf:<model.gguf>                 # the same, through the weights the engine holds
python -P $U $A rerank fp32                             # the card's pairs: yes / no logits at the last position
python -P $U $A rerank gguf:<reranker.gguf>
python -P $U $A compare                                 # every GGUF reference against fp32 (card, prompts, SciFact)
python -P $U $A prompts <out>                           # the CLI's prompt files: <out>/embed, <out>/rerank
open_qwen36_cli --model <dir with model.gguf> --kernels <set> --no-lm-head --max-ctx 1024 \
    --prompts-file <out>/embed/prompts.txt --hidden <out>/embed/h --twice
python -P $U $A npu-embed <out>/embed --same <tag>      # tag: the GGUF's parent dir name; PASS at cosine >= 0.999
python -P $U $A npu-rerank <out>/rerank --same <tag> --gguf <reranker.gguf>
python -P $U spec <model.gguf> --name <Kernel-Set-NPU2> --write open_kernels/recipes/specs/<name>.json
```

"The weights the engine holds" means: Q4_0 and Q4_1 matmuls packed exactly, Q8_0 / Q4_K / Q6_K
matmuls requantized to q4_1 by min and max as `pools.cpp` does it, the scales narrowed to bf16 as
`gemv_q4` reads them, norms in bf16 and the embedding rows exact. The weights are written a few
thousand rows at a time, so a 4B fits in host memory. `embed` skips SciFact with
`--sets card,prompts10`; at 4B an fp32 SciFact pass takes close to an hour.

`spec` reads the GGUF's float32 epsilon as the shortest decimal that is the same float32 (1e-06,
not 9.99999997e-07); otherwise the derived spec hashes differently from the `config.json` one.
Unit tests: `python -m pytest utilities/qwen3-embed-ref/tests`.
