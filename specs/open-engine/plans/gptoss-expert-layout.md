# GPT-OSS 20B: the expert layout, MXFP4, and go / no-go

**Status:** design spike, 2026-10-16, branch `feat/gpt-oss`. No NPU was used. Plan item 5 of
`.claude/plans/top5-models-plan.md`.
**Spec impact:** new requirement `OPEN-GPTOSS-EXPERT-LAYOUT` (a host prototype of the recommended
layout, unit-tested; it records a proposal, not a decision); `OPEN-MOE-WIDE-FF` notes that the
proposal takes GPT-OSS out of qwen36moe's MoE block. `OPEN-FAMILY-GPTOSS` is unchanged: the family
still raises `NotImplementedError` until kernels exist.
**Evidence:** raw logs under `.claude/plans/gptoss-run/` in this worktree (gitignored); each claim
below names its file.
**Decided 2026-10-16:** the experts stay MXFP4 in the layout below, with the uneven deal, built
against unsloth's `gpt-oss-20b-Q4_1.gguf`. If the MXFP4 decode cannot keep pace on the NPU, the
fallback is the exact int8 form on the q8 GEMV, at 2.4 times the expert bytes (25.5 GB resident).
The build has not started.

## The recommendation

**Go, with the experts kept in MXFP4 and laid out for this model rather than squeezed into
qwen36moe's MoE block, on one condition the first NPU step has to meet.** MXFP4 values have to be
decoded inside the kernel, and the compiler says that decode costs about five times q4_1's work per
weight. On eight cores that is slower than the weight stream, and a second formulation plus a cost
probe both say the arithmetic itself is the cost, so the decode will most likely have to spread over
the 16 cores the layer leaves idle. That is new design work, and it is the first thing the build has to prove on the NPU. The
closed engine runs these same MXFP4 experts at its published speed, so it can be done.

1. **Do not re-quantize the experts to q4_1.** The GGUF ships them as MXFP4 (4-bit floats sharing
   one power-of-two scale per 32 values). Converting them to q4_1 at load moved the model's
   next-token distribution by a mean KL divergence of 0.18 to 0.20 nats and changed the top token
   at 13% (code and docs) to 42% (wikitext) of positions, measured with llama.cpp on this box. That
   is the same order as the 35B's q8-to-q4_1 re-quantization (0.157, measured another way: this
   engine's logits on one chat prompt, with attention and DeltaNet re-quantized), which this repo
   already declined to ship as a default. It would also stream 20% more expert bytes per token than
   the layout below (28% if the rows are padded too).
2. **The layout** (prototyped in `open_kernels/model/gptoss_layout.py`):
   - A pool tile holds 32 rows by 288 columns of MXFP4 in 5120 bytes. 2880 is exactly ten such
     tiles, so nothing is padded along K, and two tiles make the 10,240-byte weight element every
     q4_1 projection already streams in, so one weight fifo serves the whole layer. The 224 spare
     bytes of each tile carry the rows' output biases, which then cost nothing to stream.
   - Every expert projection is 45 bands of 64 rows, dealt 6,6,6,6,6,5,5,5 to the eight main cores.
     No row is padded.
   - Each expert is one contiguous block in the pool, so routing moves all of a slot's fills by
     one stride.
   - Per token, every routed slot's gate and up run first, then one barrier, then every slot's
     down. A core holds one activation table at a time.
3. **It passes the three checks that refuse qwen36moe's layout** (OPEN-MOE-WIDE-FF): the core
   assignment covers every row once with no zero divisor, the main core's data memory is 47 KB of
   its 60 KB budget, and the expert hidden arrives in three whole x elements. In a hand-written
   model of the per-token stream every fifo balances, the barrier and the activation offsets hold,
   and eight cores fit the shim's channels; on real weights the tile packer is exact and the
   layout's arithmetic matches the fp64 reference to 2-3e-3.
4. **What remains is about five weeks of build**, ordered so the kernel-rate question is answered
   first. The other three blockers in `NOT_IMPLEMENTED["gptoss"]` turn out to be small: the clamped
   SwiGLU is one kernel, the expert biases ride in the tiles, and the sliding window is a manifest
   field the dense recipe already uses.

What I need from you is at the end: four decisions. The first decides whether the kernel step
happens at all; the others can wait until the build reaches them.

## What the model is, and what the source file holds

gpt-oss-20b: 24 layers, hidden 2880, 64 query heads over 8 kv heads at head dim 64, a 128-row
sliding window on the even layers, 32 experts with 4 routed per token and no shared expert, expert
width 2880, vocabulary 201,088. Every projection has a bias, attention has a learned sink per head,
the experts use a clamped SwiGLU, positions use YaRN.

The measurements below used `ggml-org/gpt-oss-20b-GGUF`, file `gpt-oss-20b-MXFP4.gguf`
(12,109,566,624 bytes, sha256 `27cd6c43...35901`). The repo's `convert.log` says the experts were
carried over from OpenAI's own MXFP4 without re-quantizing, the attention projections, embedding and
head were converted from bf16 to Q8_0, and router, biases, norms and sinks are f32. The experts are
10.15 GB of the 12.1.

**Which GGUF to build against (survey, 2026-10-16).** Hugging Face lists 40 GGUF repos for
gpt-oss-20b. Reading the tensor headers of seven files across the most-downloaded ones:

- OpenAI released the experts in MXFP4 (`quant_method: mxfp4`; attention, router, embedding and
  head are bf16), and every base-model GGUF read keeps them in MXFP4: unsloth's Q4_0, Q4_1 and F16,
  bartowski's Q4_0 and ggml-org's file. The quant in a file's name applies only to the attention,
  embedding and head, which is why those files are 11.5 to 13.8 GB whatever the label.
- Non-MXFP4 experts appear only in fine-tunes saved in bf16 and re-quantized (MaziyarPanahi's
  Derestricted Q4_K_M: Q5_0 and Q8_0 experts, 15.8 GB). None is the base model.
- So the expert decision below does not depend on the file chosen, but the attention side does.
  **Build against `unsloth/gpt-oss-20b-GGUF`, file `gpt-oss-20b-Q4_1.gguf`** (11.58 GB): its
  attention projections and embedding are Q4_1, one rounding from OpenAI's bf16, and pack exactly
  into the q4 pool with no re-quantization by the engine. Its head is Q8_0. ggml-org's file would
  instead need its Q8_0 attention re-quantized (a second rounding).

## The expert layout: options and accounting

**What qwen36moe's MoE block assumes.** Its expert width is a quarter of its hidden (512 against
2048), and three things in the block depend on that ratio (OPEN-MOE-WIDE-FF): one 128-row stripe is
split across `8 / stripes` cores, which is a division by zero once there are more stripes than
cores (24 at 3072); the main core holds xm's activation table and h's side by side, 13,824 bytes
against a 6,912-byte reservation at 3072; and h is broadcast as one 4 KB element when it is three.
The block also assumes a shared expert, whose gate rides its header. Its expert pack laws are
q4_1-only and have no GGUF path at all (the 3-D expert pack ops are still open in
`.opencode/skill/open-gguf-direct`).

**The three checks, side by side** (qwen36moe padded to 3072/3072, against the chosen layout):

- *Core assignment.* qwen36moe: 24 stripes of 128 rows per projection over 8 cores, and its offset
  formula needs the stripe count to divide the core count (`cps = 8 // 24 = 0`). Chosen: 45 bands
  of 64 rows dealt 6 or 5 to a core; each core streams one contiguous range, nothing is shared.
- *Core scratch.* qwen36moe: two K = 3072 tables, 13,824 bytes against a 6,912-byte reservation;
  with the reservation raised the core comes to 56,960 bytes, but only after dropping the 5,120-byte
  DeltaNet scratch it carries everywhere (62,080 with it, 640 over). Chosen: one K = 2880 table,
  6,480 bytes, 47,056 in all.
- *Element counts.* qwen36moe broadcasts h as one 4 KB element (good for an expert width up to
  1024) and acquires xm as one. Here h is 11,520 bytes, three elements, and xm 5,760, two. Chosen:
  xm's two elements are prepared and released, h's three are streamed one at a time and prepared
  by element index, as the dense FFN tail already does (`dense_prep_f32`).
- *Prep width.* Both need a new prep point: K = 3072 for qwen36moe, K = 2880 (90 blocks) here.

**Why the width is awkward.** 2880 is 45 bands of 64 rows. Three or five cores would divide that,
but the q projection's 64 bands and the kv projections' 8 rule them out, so `dense.cores_for`
gives this model one core. It is also 11.25 of the q4_1 chunk's 256 columns. A q4_1 pool
therefore pads K to 3072 whatever is done with the rows. An MXFP4 tile is a new kernel anyway, so
its width can be chosen to divide 2880.

The options, with the experts' per-token weight stream (24 layers x 4 experts x 3 projections) and
their resident size. `python -I open_kernels/model/gptoss_layout.py` prints both columns;
`.claude/plans/gptoss-run/slice_recheck_l0.log` has them.

| Layout | Format | Rows | K | Stream / token | Resident | New kernel | Fits the core? |
|---|---|---|---|---|---|---|---|
| qwen36moe block, padded | q4_1 (re-quantized) | 3072, stripes | 3072 | 1.699 GB | 13.59 GB | stripe walk, h in 3 elements | 56,960 B only without the DeltaNet scratch |
| Dense-tail bands, padded | q4_1 (re-quantized) | 3072 (6 per core) | 3072 | 1.699 GB | 13.59 GB | none | yes |
| Dense-tail bands, uneven | q4_1 (re-quantized) | 2880 (6,6,6,6,6,5,5,5) | 3072 | 1.593 GB | 12.74 GB | none | yes |
| MXFP4, 256-column tile | MXFP4 | either | 3072 | 1.593-1.699 GB | as q4_1 | MXFP4 tile | yes |
| **MXFP4, 288-column tile, uneven** | **MXFP4** | **2880** | **2880** | **1.327 GB** | **10.62 GB** | **MXFP4 tile** | **47,056 B** |
| MXFP4, 288-column tile, padded | MXFP4 | 3072 (6 per core) | 2880 | 1.416 GB | 11.32 GB | MXFP4 tile | yes |
| q8 GEMV, exact (fallback) | int8 | 2880 (6,6,6,6,6,5,5,5) | 3072 | 3.185 GB | 25.48 GB | bias seeding in `gemv_q8.h` | yes |

For scale, the GGUF's own MXFP4 bytes would be 1.269 GB per token: the chosen tile spends 4.6% on
padding to keep the 10,240-byte element. The closed engine's container pads every expert
projection to 2944 x 2944 in 2560-byte chunks of 32 x 128 (OPEN-PACK-EXPERT-ORDER), which is 1.560
GB per token, 18% more than the chosen layout. The 256-column MXFP4 tile gains nothing over q4_1 in
bytes, because a 32 x 256 MXFP4 tile is 4352 bytes and has to be padded to 5120 to share the
element, which is exactly q4_1's size.

If the decode spreads over 24 cores (below, "MXFP4: keep it, measured"), the same 45 bands deal 2
or 1 per core and the core's data memory still fits, but the streams then need a memory-tile split
that this prototype does not model, and the split constrains the deal (below).

**Why uneven rows rather than padded.** Padding to 3072 makes every core's program identical,
which is simpler, and costs 6.7% more stream on the experts (1.416 against 1.327 GB) and 0.71 GB
of NPU memory. The uneven deal costs nothing at run time: each core is its own `Worker` with its own
compiled loop counts (the attention cores already work this way, `dx.py make_attn_body`), and the
host issues per-core fill sizes. Decode is weight-stream bound on this box, so the 6.7% would be
6.7% of the experts' share of every token. If the uneven deal causes trouble in the design, the
padded deal is the fallback and changes nothing else.

**Why two phases.** qwen36moe runs each slot's gate, up and down in turn, so a core needs xm's table
and h's at once. Running every slot's gate and up first, then every slot's down, needs one table at
a time and one barrier per token instead of one per slot. The cost is that h for all four slots
sits in DDR between the phases (46 KB per layer), which it does anyway.

**The core's data memory**, from `layout()`: the activation table at K = 2880 (6,480 B), the MoE
scratch `ms` (router record, residual rows, accumulator rows, one gate and one up band, the down
rows: 1,312 floats), two weight elements, two x elements, two y elements and the 6 KB stack:
47,056 bytes of the 61,440 budget. If the same core also runs the attention projections, the table
grows to o_proj's K = 4096 and the total is 49,792. The DeltaNet scratch qwen36moe keeps on every
main core is not needed: this family has no linear-attention layers.

**What the prototype proves, and how** (`open_kernels/model/gptoss_layout.py`,
`specs/open-engine/tests/test_gptoss_layout.py`, 22 tests):

- `core_program()` and `host_sequence()` are hand-written models of the per-core fifo operations
  and the runtime sequence's DMA tasks for one token, in the order an IRON design would issue
  them. `check()` holds them against each other: the bytes each core's program takes from its
  weight and result fifos equal the bytes the host moves; the x broadcast matches; every slot's
  fills read exactly its expert's block once; slot s is slot 0 moved by s blocks; the barrier sits
  after the last h drain and before the first h fill; every slot's h is drained whole before it
  and read back after it from that slot's own region, in slot order; each core's header fills
  read the routing record and its own residual rows, its weight fills come in its program's
  order, and its drains land on the rows its fills compute; no fifo is held deeper than its depth;
  a drain the throttle waits on already has its weights issued; and the streams fit the shim's 16
  channels each way. Eleven broken sequences each fail it, and the tests pin them: one slot's
  fills shifted by a band, one core's fill a band short, an oversized table, an xm wider than the
  x fifo, the barrier deleted, the h fills issued ahead of it, a slot reading another slot's h,
  the outputs all drained at row 0, two cores' outputs swapped, the residual headers all reading
  core 0's rows, and two slots' gate | up fills issued in swapped order.
- This is not the stub-IRON tracer. That tool runs a design's own IRON code, and no design exists
  yet. The two functions here were written together, so `check()` catches a mismatch between them,
  not an assumption both share. It does not model DMA timing either, so a deadlock that depends on
  timing would still show up first on hardware. What it does catch is count, order and offset
  mistakes, which is what has hung designs here before.
- On the real GGUF (`--gguf`), one layer's four routed experts are packed into tiles and read back:
  exact, values and biases, at layers 0, 11 and 23. The layout's arithmetic (activations quantized
  the way `gemv_tab.h` does, h rounded to bf16 by its table prep, f32 accumulation per core band)
  matches `replica_gptoss` in fp64 to a relative error of 2.1e-3, 2.6e-3 and 3.1e-3 (correlation
  0.999998, 0.999998, 0.999995). Logs: `slice_check_l0.log`, `slice_check_l11.log`,
  `slice_check_l23.log`. The input is a token embedding passed through the post-attention norm, not
  a real hidden state, so this checks the arithmetic, not the model.

## MXFP4: keep it, measured

**The measurement** (llama.cpp b10754, Vulkan on the iGPU, no NPU). I made a copy of the GGUF with
only the 72 expert tensors re-quantized to Q4_1 (`llama-quantize --allow-requantize --pure
--tensor-type ffn_{gate,up,down}_exps=q4_1 ... Q8_0`) and confirmed with
`utilities/quant-compare/mxfp4_requant.py diff` that the other 387 tensors are byte-identical
(`gguf_diff.log`). llama.cpp's Q4_1 is the same min/max law the engine's GGUF packer uses
(`pools.cpp std_perm_gguf`), with fp16 rather than bf16 scales, which flatters it slightly.
Then KL divergence against the original over 20 windows of 512 tokens, two texts:

| Text | Mean KLD | Median KLD | 99th pct | Same top token | PPL, MXFP4 -> Q4_1 |
|---|---|---|---|---|---|
| wikitext-2 test | 0.200 | 0.122 | 1.26 | 57.6% | 393.5 -> 358.2 |
| code and docs (CPython argparse.py, the README, pools.cpp) | 0.182 | 0.049 | 2.01 | 87.1% | 3.68 -> 3.30 |

Logs: `ppl_mxfp4.log`, `kld_q4_1.log`, `code_ppl_mxfp4.log`, `code_kld_q4_1.log`.

**Perplexity is not the measure here.** The re-quantized model has *lower* perplexity on both
texts, on the Vulkan backend and again on the CPU backend (4 windows: 6.64 -> 5.62,
`code_cpu_*.log`, which do not record the command line), so it is probably not a backend bug. Why
it drops was not measured. One guess is that gpt-oss, trained on its chat format, is poorly
calibrated on raw text and noise in the experts flattens its predictions; but the probability it
gives the actual next token rose by 1.5 points on average on code and fell by 0.3 on wikitext
(`code_kld_q4_1.log`, `kld_q4_1.log`), which that guess alone does not explain. Either way, the KL
divergence says how far the re-quantized model is from the one OpenAI shipped, and that is the
question the format rule asks.

**Why it moves so much.** MXFP4's sixteen levels are not evenly spaced (0, ±0.5, ±1, ±1.5, ±2, ±3,
±4, ±6 times the block scale) and q4_1's are, so no q4_1 block can hold an MXFP4 block exactly: 0.1%
to 0.6% of blocks survive, and the weight error is 9 to 10.5% RMS on every expert matrix sampled
(`weight_err.log`, 27 matrices). For comparison, quantizing Gaussian weights to q4_1 once costs
about 8% (an estimate: a quarter of a step's width over the spread of 32 samples). A smarter
re-quantization that searches each block's clip range instead of taking its min and max gets the
error down to about 7.5% on three matrices (`best_q4_1.log`); its KL was not measured, and it would
still be a second rounding of the shipped model.

**What native costs instead** is the decode inside the kernel. q4_1's nibbles feed the matrix unit
directly; MXFP4 codes have to become their integer values first (`2 x e2m1` fits int8, and
`lm_head_q8.h` already runs the int16 x int8 matrix mode this needs). The decode is a few vector
operations per 64 codes: mask, two max-and-subtract steps for the ladder, the sign, and a 16-bit
shift to bring the high nibbles down, because an 8-bit shift does not legalize on this core.
Counted by hand, about twelve vector operations per 64 codes for each nibble half, against q4_1's
one mask.

**The compiler's view, without hardware.** `utilities/aie-probes/gemv_mx4_probe.cc` is the chosen
tile's GEMV body: q4_1's walk with the decode in registers, the E8M0 scale built from its byte, the
bias seeding the accumulator. `open_kernels/kernel_remarks.py` compiled it and the shipped q4_1 tile
the way IRON does (`mx4_remarks.log`, `mx4_probe/`; the command is in the probe's first lines, and
re-running it under WSL gives the same numbers, `probe_command_recheck.log`):

- q4_1: 59 bundles (cycles) per 32-column block of a 32-row chunk, no software pipeline.
- MXFP4: 289 cycles per block, bound by the vector ALU slot (the pipeliner's resource floor is 266).
  The body is 1,760 bytes of program memory against q4_1's 1,280.

That is 2.0 bytes of tile per cycle per core against q4_1's 10.8. Eight cores keep pace with a 35-40
GB/s weight stream only if a block takes at most about 121 cycles at a 1 GHz core clock, 182 at 1.5,
218 at 1.8. So the first cut is compute-bound on eight cores at any plausible clock, while q4_1 has
room to spare. Two ways back, which step 3 measures before anything else:

- **A cheaper decode.** The floor is about ten vector operations per 64 codes per nibble half; the
  first cut spends roughly twice that. A second formulation (codes sorted by value at pack time,
  min and max only, the offset of 12 taken back through the block sums the way q4_1's minimum is)
  is in the same file behind `MX4_DECODE=2`. It came out worse, 333 cycles (`mx4_variants.log`).
  Dropping the 16-bit shift altogether (`MX4_NO_SHIFT`, wrong values, a cost probe only) saves 24
  cycles, 289 to 265 (`mx4_noshift.log`). So the decode arithmetic itself is most of the cost, and a
  better formulation is unlikely to find the factor of 1.5 to 2.4 that eight cores need.
- **More cores, which is the realistic route.** In the MoE half of the layer, rows 4 and 5 of the
  array are idle: 16 more cores. Splitting each column's weight stream three ways through its memory
  tile gives each core a third of the decode, and the first cut then fits even at 1 GHz (289 against
  a budget of about 360). Part of the host accounting holds there: `layout(n_cores=24)` deals the
  45 bands 2 or 1 per core and a core's data memory falls to 43,984 bytes. The stream as modeled
  does not: it gives every core its own shim stream, 25 fill streams at 24 cores (17 already at
  16) against the shim's 16 channels, and `check()` refuses it (`layout_cores_recheck.log`). The
  fix is the memory-tile split and join that `gemm_q4`'s fan-out mode and `moe_batch` already use:
  one shim stream per column, split at the column's memory tile, results joined there. The
  prototype does not model it, and it constrains the deal: a split hands every core of a column
  the same number of elements, so a column's cores need equal band counts. At 24 cores the 45
  bands can still be dealt that way without padding (two per core in seven columns, one per core
  in the eighth); at 16 cores they cannot, since 45 is odd, so one band would be padded. Using the
  split inside a whole-layer decode design is the new work here, and it carries its own risk.

The closed engine's container is MXFP4 too, and at the decode rate FastFlowLM publishes for this
model (`docs/docs/benchmarks/gpt-oss_results.md`) it reads about 28 GB/s of expert data, so a decode
that keeps pace exists on this hardware.

**A lossless fallback, not the plan**: every MXFP4 value is an int8 times its power-of-two block
scale, so the engine's q8 GEMV could hold the experts exactly with no new decode
(`weight_err.log` confirms every sampled scale is a normal bf16). `gemv_q8.h` reads 16-row
half-tiles of 16 x 256 values in 5,120 bytes, 1.25 bytes a value, and pads K to 3072, so the
expert stream becomes 3.19 GB per token and the resident experts 25.5 GB: 2.4 times the chosen
layout, and more than the whole 35B keeps resident (about 21.6 GB). `python -I
open_kernels/model/gptoss_layout.py` prints both. The kernel also starts its accumulator at zero
and reads no bias, so the expert biases need a small change to it (seed the accumulator from the
half-tile's spare bytes, as the MXFP4 tile does). It is there if the decode cannot be made to keep
pace and q4_1's quality cost is not acceptable.

## Go / no-go: what remains, in order

Sizes are rough: S is up to a day, M two to three days, L about a week. "Needs the NPU" means short
harness or model runs, not a long campaign.

1. **GGUF reading for gpt-oss** (S-M, host only). Add `gpt-oss` to `GGUF_FAMILIES` (the HF-config
   derivation exists; the GGUF keys are in the convert log), map the tensor names (`attn_sinks`,
   `ffn_{gate,up,down}_exps.weight` and `.bias`, `ffn_gate_inp.weight` and `.bias`, the attention
   biases), teach `GgufFile` MXFP4 (type 39, 17-byte blocks) as raw blocks, and let `oflm-add`
   install an MXFP4 GGUF for this family (`GGUF_QUANT_PREFERENCE` and `GGUF_ENGINE_TYPES` in
   `oflm_add/__init__.py` do not list MXFP4, so it would be refused today).
2. **The pack op `expert_mx4`** (S-M, host only): `gptoss_layout.pack_expert` as a `pack.py` op with a
   byte-identical `pools.cpp` twin, pinned by an FNV hash on both sides the way `std_fuse` is. Also
   give `std_perm_gguf` a source width: it uses the pool's `in_dim` as the source row stride, so q,
   k and v at K = 2880 cannot pack into a 3072-wide pool today.
3. **The MXFP4 GEMV kernel, and its rate, first on the NPU** (M, needs the NPU; L if it has to
   spread over more cores). `gemv_mx4.h` from `utilities/aie-probes/gemv_mx4_probe.cc`: the band
   walk, the bias seeding the accumulator, the entry points. Harness test against a numpy reference,
   then the `GEMV_NULL` ablation against the q4_1 kernel on the same stream, which says whether the
   tile or the stream sets the pace. **This is the gate.** The compiler puts the first cut at 289
   cycles per block where eight cores need roughly 120 to 220, so expect to spread the decode over
   rows 4 and 5 (a memory-tile split and join as `gemm_q4` and `moe_batch` have, new inside a
   whole-layer decode design). If that cannot get the expert stage within
   about 20% of the stream (the bytes MXFP4 saves over q4_1), stop and bring it back: the fallbacks
   are q4_1 with its measured quality cost or the exact int8 form at 2.4 times the bytes, and both
   are your call.
4. **The clamped SwiGLU** (S): `vsigmoidN` exists; the kernel clips gate above at 7, up both ways,
   applies 1.702 inside the sigmoid and the +1 on up. Checked against `replica_gptoss.clamped_swiglu`
   on the harness.
5. **The router** (S-M): qwen36moe's router cores (`router_acc`, `router_fin`) on the dense norm
   helper, with the bias added before the top-4, at 32 experts. The weights are 32 rows of 2880,
   exactly 32 of the helper's 5760-byte elements.
6. **The attention half** (M-L, needs the NPU): the dense recipe's attention with the sink
   (OPEN-ATTN-SINK steps 2-5; the kernel guard is in), the q/k/v bias (exists, Qwen2), an o_proj bias
   (new: a norm-helper variant that adds it, or the accumulator seeded the way the expert tiles do),
   the window (a manifest field Gemma 3 already uses), YaRN (the spec derives the table; check the
   engine applies the attention factor), the attention point (64, 64, 8, 64) in the catalogue, the
   norm at 2880, and o_proj's 2880 output rows (dealt unevenly, or padded to 3072).
7. **The layer design and the recipe** (L, needs the NPU): one hardware context with two instruction
   streams per layer as `ax0`/`ax1` have (a context switch costs about 2.5 ms, so two contexts per
   layer is not an option): part 0 is the attention half and the router, `moeroute2` patches part 1,
   part 1 is the two-phase MoE. `recipes/gptoss.py`, `gen_kernels`, the manifest, both layer types
   sharing the MoE program with only the attention kernel's window differing, and the export. The
   main core then carries the q4_1 tile (attention) and the MXFP4 tile (experts); the existing `ax1`
   main core is 12,032 of 16,384 bytes of program memory with one tile, so this is the second thing
   to measure, on the first build (`existing_core_sizes.log`).
8. **The engine** (S): `moeroute2` already patches one block per expert if the manifest says so.
   `stream_patch.hpp moe2_table` reads any fill below `pool_down` as expert `off / (2 * up_bytes)`
   plus a remainder, so setting `up_bytes` to half an expert block and `pool_down` past the last
   expert gives exactly this layout's patch. What is left is the manifest's `moe` block for this
   family and a test that a slot's fills move by one block.
9. **Validation** (M, needs the NPU): a layer slice against `replica_gptoss`, the full model, one
   `oflm-test` category through `oflm serve`, per the five-model rules.
10. **Later: block prefill** (L). A GGUF kernel set prefills token by token today (the GEMM route has
    no f32-scale twin), so a 1000-token prompt pays a thousand decode steps. The closed engine
    prefills at 221 to 477 tokens per second on its own page. This is what makes the model pleasant
    rather than possible, and Mellum (feasibility Part 1) needs the same routed-expert prefill.

Order: 1 and 2 on the host while 3 runs on the NPU; then 4 and 5; then 6; then 7 and 8 together; then
9. About five weeks of focused work to a model that decodes through `oflm serve` (four if eight cores
turn out to keep pace after all), not counting 10.

**If step 3 stalls, the rest need not wait.** The same layout runs exactly on the q8 GEMV: the
experts as int8 (exact, above) in its 16-row half-tiles with K padded to 3072, the biases in the
768 bytes each half-tile leaves free once the kernel seeds its accumulator from them (a small
change, above), the same bands, blocks and two phases. That brings the whole layer up with exact
numerics at 2.4 times the expert bytes (25.5 GB resident, more than the 35B's 21.6 GB, so it needs
at least the same care with free memory on this box), and the MXFP4 tile then replaces it as a
speed and memory step rather than a correctness one.

**Spec impact of the build**, for its own plan (names are proposals):
- New: `OPEN-GPTOSS-GGUF` (the spec and tensor names from the GGUF), `OPEN-PACK-EXPERT-MX4` (the pack
  op, NumPy and C++ byte-identical), `OPEN-GEMV-MX4` (the tile against a reference, and the rate
  gate), `OPEN-GPTOSS-ACT` (the clamped SwiGLU kernel), `OPEN-GPTOSS-ROUTER` (the router with its
  bias), `OPEN-ATTN-O-BIAS`.
- Modified: `OPEN-FAMILY-GPTOSS` (a recipe exists, the `NotImplementedError` goes),
  `OPEN-ATTN-SINK` (steps 2-5), `OPEN-GPTOSS-EXPERT-LAYOUT` (from a host prototype of a proposal
  to the design, in the format and deal you choose, with the memory-tile split if the decode
  spreads), the `oflm-add` requirements (an MXFP4 install).
- Removed: none.

**Risks, beyond the kernel rate:**
- **Program memory on the main core** (step 7). Two GEMV bodies plus both halves' loops; my estimate
  from the measured `ax1` core is 14-15 KB of 16. It is an estimate, not a measurement: no build
  links both tiles yet. If it does not fit: `-Oz` on selected units (about 300 bytes on the q8
  35B), dropping what this family does not use (`moe_hdr2`'s shared-expert gate), and as a last
  step one tile walk for both formats with a runtime format branch, the kind of fold that made the
  mixed q8/q4_1 cores fit (`gemv_q4_gyms`).
- **The attention projections are Q8_0 in ggml-org's file**, and the weights rule re-quantizes Q8_0
  to q4_1. On the 35B that cost a KL of 0.157 on its own. Building against unsloth's Q4_1 file
  (above) avoids it: its attention and embedding are already Q4_1 and pack exactly.
- **Prefill speed** (step 10), as above.
- **Memory**: about 11.5 to 12 GB of NPU buffers resident (10.62 experts, 0.41 attention, 0.39 to
  0.70 for the head at q4_1 or q8, the KV cache), against 21.6 GB for the 35B. FLM's page asks for
  more than 32 GB of system RAM for this model.

## What this spike did not do

- **No NPU at all**, by brief. Nothing here says how fast anything runs on the NPU.
- **No IRON design was built**, so the stub-IRON tracer had nothing to run. A hand-written model
  of the stream (`check()`) and the arithmetic stand in for it. A dataflow timing simulation (the
  threaded mock in the `aie-dataflow-presim` skill) was not written.
- **The kernel numbers are the compiler's schedule, not the NPU's.** Neither tile's loop was
  software-pipelined, and the core clock was not established here, so the 289 against 59 is a ratio
  to trust and the cycle budgets at 1, 1.5 and 1.8 GHz are bounds to check on the NPU.
- **The KL measurement was trimmed** to 20 windows of 512 tokens per text, two texts, one reference
  route, about 6 minutes of iGPU time in all. It does not cover chat-formatted text or task accuracy,
  and the absolute perplexities mean little for this model (above).
- **The weight error was sampled** on 27 of 2304 expert matrices (layers 0, 11, 23; experts 0, 17,
  31), which all agree to within 1.5 points.
- **The slice used one token per layer** and a stand-in hidden state.
- **The re-quantized copy was deleted** after the measurement; the MXFP4 GGUF was kept for the
  next session.

## Decisions

Items 1 to 3 were decided on 2026-10-16 as recommended. Item 4 is open; the build has not started.

1. **Native MXFP4 for the experts** (recommended, gated on the kernel keeping pace with the
   stream), or q4_1 with the measured KL of about 0.19 and the existing kernel, or the exact int8
   form on the q8 GEMV at 2.4 times the bytes (25.5 GB resident). The weights rule lists Q8_0, Q4_K
   and Q6_K as re-quantized and does not mention MXFP4; this is the first case where the format the
   GGUF ships is not one the pools hold.
2. **Uneven deal or padded rows.** I recommend uneven (6.7% less expert stream). Padded is simpler
   and is the fallback.
3. **The source file.** I recommend unsloth's `gpt-oss-20b-Q4_1.gguf`: the experts are the same
   MXFP4 as every base-model GGUF, and the attention and embedding arrive as Q4_1, so nothing outside
   the experts is re-quantized. The alternative is ggml-org's file, whose Q8_0 attention the engine
   would re-quantize to q4_1 (unmeasured here; the 35B's number suggests it is not free).
4. **Whether to start the build now or after Gemma 4.** The plan put gpt-oss last of the five, and
   the work above is about five weeks.
