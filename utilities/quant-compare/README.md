# quant-compare

`cmp_logits.py` scores how far one run's per-position logits are from another's (worst / median
correlation, argmax flips and how many were near-ties, top-5 agreement, KL divergence).

`flores_chrf.py` scores a served translation model on FLORES-200 devtest with chrF, through any
OpenAI-compatible endpoint (`oflm serve`, `llama-server`), greedy, with the Hy-MT2 model card's
"Default Translation" prompt. `score` compares two runs on the sentences both finished, with a
paired bootstrap interval on the chrF difference.

## Result 2026-10-16: Hy-MT2-1.8B at q4_1 on the NPU

`tencent/Hy-MT2-1.8B-GGUF`'s Q8_0 file, requantized to q4_1 at load by the open engine
(`oflm serve`), against llama.cpp (build 10754, Vulkan, Radeon 890M) running the same Q8_0 file.
The first 100 devtest sentences; the German run stopped at 67 when its 15-minute budget ran out.

| pair | open engine, q4_1 | llama.cpp, Q8_0 | difference, 95% bootstrap | identical outputs |
|---|---|---|---|---|
| en -> zh (Chinese prompt) | 38.63 | 39.52 | -0.89 [-2.09, +0.37] | 17 / 100 |
| en -> de | 58.47 | 59.35 | -0.88 [-2.34, +0.67] | 6 / 67 |

The 4-bit engine scores about 0.9 chrF below the 8-bit reference on both pairs. At this sample
size that is within noise (both intervals include zero), and the loss is under about 2.3 chrF.

The Q4_K_M file, which oflm-add picks from this repo by default, requantized the same way:
en -> zh 39.07 on the same 100 sentences, -0.45 [-2.18, +1.26] against llama.cpp on Q8_0 and
+0.44 [-1.37, +2.35] against the Q8_0 file on the NPU. Only 1 and 2 of its 100 outputs match
those two runs exactly, against 17 between the two Q8_0 runs.

Not covered: other pairs, the reverse direction, German on Q4_K_M, the Q6_K file, COMET or any
human judgment.

    python flores_chrf.py gen --url http://127.0.0.1:52633 --model hy-mt2-flash:1.8b \
        --flores <flores200_dataset/devtest> --n 100 --out open.jsonl
    python flores_chrf.py gen --url http://127.0.0.1:52634 --model q8 --flores <...> --n 100 --out ref.jsonl
    python flores_chrf.py score --flores <...> --a open.jsonl --b ref.jsonl

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
