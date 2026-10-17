# quant-compare

`cmp_logits.py` scores how far one run's per-position logits are from another's (worst / median
correlation, argmax flips and how many were near-ties, top-5 agreement, KL divergence).
`head_nll.py` runs a dense GGUF through the fp64 replica and scores the next token with the head
as the engine packs it and as the file stores it.

## Result 2026-10-16: Gemma 3 1B's F16 head re-quantized to q4_1

Google's QAT Q4_0 GGUF keeps the 26 layers at Q4_0, which the pool holds exactly, and the tied
head at F16, which it cannot: the engine re-quantizes the head to q4_1 at load. Over the first
1024 tokens of the repo's `README.md`, both heads on the same hidden states:

| head | mean NLL | next token is the top-1 |
|---|---|---|
| as packed (q4_1) | 3.4845 nats | 438 / 1024 |
| as stored (F16) | 3.4860 nats | 437 / 1024 |

The 4-bit head costs nothing measurable in NLL (median difference +0.0001 nats, worst single
position +1.37). The two heads' argmax differs at 88 positions, 79 of them near-ties (top-2 margin
under 0.5 logits); KL(F16 || q4_1) is 0.020 nats on average, 0.23 at worst. One text, so it says
nothing about a long generation's drift.

    python utilities/quant-compare/head_nll.py --gguf gemma-3-1b-it-q4_0.gguf \
        --spec open_kernels/recipes/specs/gemma3-1b.json --text README.md \
        --tokenizer <the model's tokenizer.json> --tokens 1024

## Result 2026-10-06: the 35B's q8 container re-quantized to q4_1

`Atomic-Germ/Qwen3.6-35B-A3B-NPU2` (the same 23.24 GB file as `FastFlowLM/Qwen3.6-35B-A3B-NPU2`)
stores attention, DeltaNet and the DeltaNet out projection at q8. The default export keeps them at
q8, which leaves decode on two hardware contexts and prefill token by token (no block route).
`OPEN_KERNELS_FORCE_Q4_1=1` re-quantizes them to q4_1 at load, which turns on #116's one-context
decode and the block prefill route.

Same 656-token prompt (a summarize request over `open_kernels/README.md`), logits at every
position, HX 370:

| | prefill (logits at every position) | decode |
|---|---|---|
| q8 (default) | 84 s | 156-167 ms/token |
| q4_1 re-quantized, token by token | 51 s | 69-70 ms/token |
| q4_1 re-quantized, block route | 14 s | -- |

| against q8 | logits corr (median / worst) | argmax agree | top-5 identical | KL mean / max |
|---|---|---|---|---|
| q4_1, token by token | 0.9866 / 0.50 | 570 / 656 (86 flips, 43 near-ties) | 277 / 656 | 0.157 / 10.8 nats |
| q4_1, block route | 0.9864 / 0.44 | 570 / 656 (86 flips, 47 near-ties) | 271 / 656 | 0.155 / 10.1 nats |
| (block route vs q4_1 token by token) | 0.9993 / 0.55 | 633 / 656 | 538 / 656 | 0.031 / 8.4 nats |

143 of 656 positions fall below 0.98 correlation, spread through the text. **Quantizing q8
weights a second time, down to q4_1, moves this model far more than the route does.** It is not a
free speedup. A q4_1 (or Q4_K) container quantized once from the original weights has not been
measured.

    open_qwen36_cli --model <35B> --kernels <set> --ids-file ids.txt --max-tokens 1 \
        --prefill-logits --dump-logits <dir>/y [--gemm-block]
    python utilities/quant-compare/cmp_logits.py <q8 dir>/y <q4_1 dir>/y
