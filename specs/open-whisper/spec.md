# The open Whisper engine

Whisper is the last model that runs only on the closed engine
([#72](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/72)). It does not fit
`causal_lm`: it encodes a 30 s window once, then decodes one token at a time while
attending to that window, so it has two halves rather than one. This file is what the open
replacement must do, and what has been measured.

Directory name gives the prefix: `OPEN-WHISPER`.

The model is `openai/whisper-large-v3-turbo` (MIT): d_model 1280, 32 encoder layers of 20
heads x 64 with FFN 5120, 4 decoder layers, 128 mels, 1500 audio frames, 448 text
positions, vocab 51866, pre-LN with biases, exact-erf GELU, learned absolute positions, no
RoPE. Weights ship fp16.

What stays as it is: the FFmpeg decode, the FFTW log-mel, the token loop, language
detection, timestamps and the 30 s chunking in `src/common/whisper/`. They are already
open, and the seam below is the only place the engine is reached.

**The reference is transformers in float64**, not the closed engine. The closed engine is
Q4_1 and its transcripts differ from the model's: a Chinese clip comes back translated into
English, a Korean one as a different sentence, and a quiet window as the well-known Russian
subtitle hallucination. Matching it would mean matching a lossy reference.

## Requirements

### OPEN-WHISPER-REF: the oracle, the replica and the ceiling they establish
**Applies to:** `open_kernels/model/whisper_goldens.py`, `replica_whisper.py`, `whisper_decode_check.py`
**Test category:** offline (torch + transformers; no NPU)
**Acceptance criteria:**
- The goldens are produced by `WhisperForConditionalGeneration` in float64 and carry, per
  clip's first 30 s window: mel, conv stem, all 32 layer inputs, the pre-LN last output,
  `enc.out`, each decoder layer's cross K and V, and the greedy token path with its logits
  under both prompt protocols (transformers', and the host's, which never feeds the
  detected language token).
- Tensors are stored contiguous. `safetensors.numpy` writes a non-contiguous array in
  memory order, silently, and every hidden state of this model is non-contiguous
  (transformers keeps the conv stem's permuted layout through the residual adds).
- `replica_whisper.py --exact` reproduces the goldens to fp32 roundoff, which is what
  makes the decomposition (tap-major im2col, fused QKV with a zero K bias, the fused cross
  K|V operand) a checked claim rather than an intention.
- The bf16-operand replica is the **ceiling** any engine built on the bf16 GEMM can reach.
  Gates are calibrated from it, never set in advance.

**Measured 2026-09-22**, on the corrected audio (float64 cosine over the 1500 real rows,
six clips): `enc.out` **0.99600-0.99943**, cross K/V min 0.99485-0.99922, conv stem
0.9999979-0.9999999. That reproduces whisper-xdna's independent finding that 0.999 on
`enc.out` is not reachable at 32-layer depth in bf16. The final LayerNorm amplifies the
relative error about 3x.

**It costs a few tokens, and the number depends on the implementation.** With the exact
decoder on the numpy replica's encoder output, 8 of 12 (clip, protocol) pairs are
token-identical to float64 and 4 diverge. The earlier claim here -- "it costs no tokens",
1 of 12 -- was measured against goldens built from **louder audio** (trap 28) and is
withdrawn.

**And the two bf16 implementations do not diverge on the same pairs.** The NPU engine
matches float64 on 11 of 12, including three pairs where the numpy replica does not; the
replica's fourth is the one the engine also takes. Both are bf16-operand paths, and they
differ in accumulation order and in how attention is computed, so on a clip where a token
is marginal they land on opposite sides of it. A bf16 "ceiling" is therefore a property of
a datapath, not a single number two implementations must agree on.

### OPEN-WHISPER-SEAM: one engine interface, and no silent fallback between engines
**Applies to:** `src/include/whisper/whisper_engine.hpp`, `src/common/whisper/whisper_engine_closed.cpp`, `whisper_engine_select.cpp`
**Test category:** integration (through `oflm serve --asr 1`)
**Acceptance criteria:**
- `Whisper` holds a `whisper_engine`, never a concrete engine. The closed `whisper_npu` is
  a prebuilt class and is wrapped, not derived from.
- The load log names the engine that was selected.
- `OFLM_WHISPER_ENGINE` selects one explicitly; a value this build cannot provide is an
  error naming it, never a fallback to the other engine. Both engines return a transcript,
  so a fallback would be invisible.
- `--asrmodel TAG` chooses the model. A tag that does not resolve to a Whisper model is
  refused **before** anything is downloaded: `model_list::get_model_info()` answers an
  unknown tag with `llama3.2:1b`.
- The open engine is ready when its weights (`model.open.safetensors`, `model.safetensors`
  or `model.gguf`, OPEN-WHISPER-STOCK) and a kernel set are both found. Kernels belong to
  the model's shape, not its name: when no set sits under the model's own name, a complete
  set whose `hf_config_check` matches the model's `config.json`, under any xclbins root, is
  used. The log names the weight file and the rule that found the kernels.

**Measured 2026-09-19:** six clips x three requests through `/v1/audio/transcriptions`,
byte-identical to the pre-seam build (18/18). `OFLM_WHISPER_ENGINE=open` and
`--asrmodel llama3.2:1b` are both refused, exit 1.

### OPEN-WHISPER-CONTAINER: the weights the open engine reads
**Applies to:** `utilities/q4nx-build/q4nx/open_whisper.py`
**Test category:** unit (`utilities/q4nx-build/tests/test_open_whisper.py`)
**Acceptance criteria:**
- Built from the HuggingFace checkpoint, into `model.open.safetensors` -- deliberately not
  `model.q4nx`, so the closed reader is never handed it.
- Encoder GEMM operands are `B = W^T` row-major `[K, N]` bf16, **not** tiled: the tile
  tuple belongs to the kernel set, and a container pre-tiled for one tuple is silently
  wrong for another. The engine tiles at load.
- The fusions the streams expect are done here: `Q|K|V` with a zero K bias (k_proj has
  none), the four decoder layers' cross `K|V` as one operand with bias `0|bv0|...`, and
  the conv stem as tap-major im2col operands.
- Biases, norms and position tables are f32. The decoder keeps transformers' names in
  `[out, in]` bf16, with `embed_tokens` as the output head.
- `tokenizer_config.json` gains `bos_token_id` and `eos_token_id`: the host reads both and
  HuggingFace's file carries neither. They are the only two fields the shipped closed
  container adds, diffed to confirm.
- The output is byte-reproducible, and a geometry other than turbo's is refused.

**Measured 2026-09-19:** 8 unit tests pass, including that the stored im2col operand
computes a direct conv1d at strides 1 and 2; mutating the conv tap order or the cross bias
order each fails the test that names it. The real container is 1,624,143,560 B, 481
tensors.

### OPEN-WHISPER-STOCK: the engine reads OpenAI's checkpoint as it ships
**Applies to:** `src/open_whisper/weight_source.cpp`, `weights.cpp`, `decoder.cpp`
**Verification:** test
**External tests:** `src/open_whisper/weight_source_test.cpp` (built and run by `src/open_whisper/build.cmd`)
**Acceptance criteria:**
- A model directory holding OpenAI's `model.safetensors` (transformers' names; F16, BF16 or
  F32) loads on the open engine with no conversion step. At load, its tensors are arranged
  into the container's (OPEN-WHISPER-CONTAINER): the same names, fusions, f32/bf16 split and
  round-to-nearest-even, so every container tensor is bit-identical to the one q4nx-build's
  `whisper_tensors()` writes from the same checkpoint. The encoder, decoder and tiling are
  unchanged.
- A tensor that is missing, or has another shape, is refused at load, naming it; so is a
  file cut short.
- When a directory holds more than one weight file, the engine reads `model.open.safetensors`,
  then `model.safetensors`, then `model.gguf`, and the load log names the one it read and the
  ones it ignored.

**Test (2026-10-16):** a tiny Whisper (d_model 32, one encoder and two decoder layers, FFN
64) written as a checkpoint: all 67 container tensors match the converter's output for the
same values, by dtype, shape and bytes (fixtures and expected hashes from
`src/open_whisper/testdata/gen_weight_fixtures.py`, which runs the converter's own code).
Swapping q and v in the fused operand, or moving the cross-attention bias, fails it. The
same checkpoint cut short by 64 bytes is refused.

**Verification (manual), 2026-10-16:** `weight_source_test --compare <model.open.safetensors>
<model.safetensors>` on openai/whisper-large-v3-turbo against the container built from it:
481 of 481 tensors, 808,924,160 values, identical.

**Verification (manual), hardware.** Procedure: install with `oflm-add` into a user registry,
put the exported kernel set under any xclbins root, run `oflm serve --asr 1 --asrmodel <tag>`
with `OFLM_WHISPER_ENGINE` unset (the load log must say the open engine was chosen and why),
post a 30 s clip to `/v1/audio/transcriptions`, then serve the container built from the same
weights the same way. Pass: the two texts are identical. A transcript from a build older
than #111 is compared with today's speed defaults switched off: `OFLM_WHISPER_PROTOCOL=legacy
OW_ATTN=host OW_HOST_FAST=0 OW_DEC_W=bf16 OW_DEC_HEAD=bf16 OW_DEC_XKV=fp32`.
2026-10-16: `whisper-v3:turbo-safetensors` matched the checkpoint's container with NPU
attention and with `OW_ATTN=host`; with the speed defaults switched off it reproduced, word for
word, the transcript a September build made from that container. One clip.

### OPEN-WHISPER-GGUF: the engine reads a GGUF of the model
**Applies to:** `src/open_whisper/weight_source.cpp`, `src/open_qwen36/gguf_file.cpp`
**Verification:** test
**External tests:** `src/open_whisper/weight_source_test.cpp`
**Acceptance criteria:**
- A model directory holding `model.gguf` in the `stt.whisper` layout (handy-computer's
  whisper-large-v3-turbo GGUFs: `enc.blocks.N.attn.q.weight` and so on) loads with no
  conversion step, through the same `GgufFile` reader the LLM engine uses. Its tensors are
  arranged into the container's as for OPEN-WHISPER-STOCK.
- An F16 GGUF gives every container tensor bit-identical to the container built from the
  checkpoint it was made from. A quantized one (Q8_0, Q4_K, Q5_K, Q6_K) is dequantized with
  llama.cpp's loops and gives the container the converter writes from those values.
- Refused at load, naming the reason: `general.architecture` other than `whisper`; any
  `stt.whisper.*` geometry key (layers, widths, heads, mels, positions, vocabulary) that is
  not the engine's; a scaled or untied token embedding; a missing tensor; a tensor of
  another shape; a type the reader does not dequantize; a file cut short, even by less
  than its header (tensor offsets count from the end of the header).
- Every tensor is widened to bf16 at load whatever the file holds, so the quant changes
  the download and adds its own rounding, not the 1.2 GB of resident encoder operands,
  which stay bf16. The decoder then requantizes its linear layers and head to int8 by
  default (`OW_DEC_W`, `OW_DEC_HEAD`), as it does for the container. An 8-bit encoder
  stream is separate work.

**Test (2026-10-16):** the same tiny Whisper as an F16 and a Q8_0 GGUF, all 67 tensors
against the converter run on the values the gguf package reads from each file; Q4_K, Q5_K
and Q6_K rows against the gguf package's dequant of the same bytes, bit for bit; the name
map against the real turbo GGUF's tensor list (all 587 tensors but the two frontend ones
used, with their shapes); a GGUF of another geometry, or without the `stt.whisper` keys,
refused; the F16 GGUF cut short by 64 bytes refused (it used to load, the missing bytes
read as zeros).

**Verification (manual), 2026-10-16,** handy-computer/whisper-large-v3-turbo-gguf with
`weight_source_test --compare`: the F16 GGUF against the container built from OpenAI's
checkpoint, 481 of 481 tensors identical; the Q8_0 GGUF against the container the old
converter built from the same Q8_0 file, 481 of 481 identical; the Q4_K_M and Q5_K_M files
(Q4_K or Q5_K, Q6_K and Q8_0 tensors) against the converter run on the gguf package's
dequant of the same file, 481 of 481 identical each. The all-Q6_K file was not run; its
type is covered by the token embedding in both K files.

**Verification (manual), hardware,** the OPEN-WHISPER-STOCK procedure. 2026-10-16: the Q8_0
GGUF installed as `whisper-v3:turbo-q8_0` (directory `whisper-gguf-Q8_0`, kernels found by
geometry under `xclbins/Whisper-V3-Turbo-NPU2`) transcribed the clip exactly as the Q8_0
container did with the same build, and the default rule chose the open engine with
`OFLM_WHISPER_ENGINE` unset. With the speed defaults switched off it reproduced, word for word,
the transcript a September build made from that container. With them on, the GGUF and the
container alike differ from that older transcript in two marginal words. `oflm-test --audio
--model whisper-v3:turbo-q8_0`: transcription 3 of 3 PASS. One clip; F16, Q4_K_M and Q5_K_M
were checked at the weight level only.

### OPEN-WHISPER-INSTALL: oflm-add installs a GGUF or OpenAI's checkpoint
**Applies to:** `utilities/oflm-add`
**Verification:** test
**External tests:** `utilities/oflm-add/tests/test_whisper_install.py`
**Acceptance criteria:**
- `oflm-add handy-computer/whisper-large-v3-turbo-gguf` installs one of the repo's GGUFs as
  `model.gguf` (`--quant`, default Q8_0) and registers `whisper-v3:turbo-<quant>`;
  `oflm-add openai/whisper-large-v3-turbo` installs the checkpoint and registers
  `whisper-v3:turbo-safetensors`. Both are ASR entries (`asr: true`, family `whisper-v3`,
  `details.weights` `gguf` or `safetensors`) in the user registry, with no download URL, so
  nothing can be re-fetched over them.
- Each GGUF quant installs into its own directory (`<repo directory>-<quant>`), so a second
  `--quant` install never serves the first one's `model.gguf` under its own tag. `--force`
  replaces files already in that directory, downloaded ones included.
- `--tag` overrides the tag within `whisper-v3:<name>`: `oflm serve` refuses an `--asrmodel`
  tag that does not start with `whisper`, and keeps only `whisper-v3` out of the chat model
  list. A tag the shipped registry already holds (`whisper-v3:turbo`) is refused: `oflm`
  would size the files against that model's download records, call them missing and
  download it again. Both refusals come before anything is copied.
- A GGUF carries no config or tokenizer files. `config.json`, `tokenizer.json`,
  `tokenizer_config.json` and `generation_config.json` come from the checkpoint the GGUF
  names in `general.repo_url` (else its README's `base_model`), and the install is refused
  unless they agree with the GGUF's own metadata: the geometry, every vocabulary id, the BPE
  merges, the special-token ids and both suppress lists.
- `tokenizer_config.json` gains `bos_token_id` and `eos_token_id` from `config.json`, the
  two fields the host reads that OpenAI's file lacks (the container build added the same two).
- Only whisper-large-v3-turbo's geometry installs; another is refused, since no kernel set
  exists for it.
- No kernel link is made. The engine finds the Whisper kernel set by geometry
  (OPEN-WHISPER-SEAM); the installer reports the set it will use, or how to build one.

**Why the files are fetched, not derived from the GGUF:** the GGUF carries the vocabulary,
merges, special ids and suppress lists, but not every value the host reads: generation's
`max_initial_timestamp_index`, and which added tokens are special (they decide how
`<|0.00|>` tokenizes and what a decode prints). A file rebuilt from the metadata would rest
on conventions for those, and would differ from OpenAI's without failing anywhere. The
fetched files are OpenAI's own, checked against the metadata.

**Test (2026-10-16):** 10 unit tests: quant choice, the GGUF checks, every sidecar check
(merges, vocabulary, a special id, a suppress list), the tokenizer_config fields, kernel
discovery by geometry under another name, another Whisper size refused before anything is
copied, a local GGUF install end to end with the fetch stubbed, two quants installed in
turn (then one again with `--force`) each serving its own file, and the tags the server
cannot serve refused before anything is copied. The two-quant test fails on the earlier
one-directory layout.

**Verification (manual), 2026-10-16:** installing the Q8_0 GGUF from a local directory into
a scratch registry fetched the four files from openai/whisper-large-v3-turbo; config.json,
tokenizer.json and generation_config.json are byte-identical to OpenAI's, and
tokenizer_config.json to the one in the container directory (OpenAI's plus the two ids).
Reinstalled into `whisper-gguf-Q8_0` after the directory change, every file came out
identical to the first install's. With the network stubbed at the download call, `--force`
on a remote install replaced a `model.gguf` downloaded earlier; before the fix it kept it.

### OPEN-WHISPER-KERNELS: one xclbin, one instruction stream per encoder shape
**Applies to:** `open_kernels/designs/whisper_gemm/`, `open_kernels/export_whisper_kernels.py`
**Test category:** hardware (`open_kernels/harness/run_kernel`)
**Acceptance criteria:**
- Every encoder matrix product is a stream over the pre-tiled bf16 whole-array GEMM at
  tile (64, 64, 32) on 8 columns, `rtp=True`, `tg_depth=2`: conv1 3072x384x1280, conv2
  1536x3840x1280, qkv 1536x1280x3840, o 1536x1280x1280, fc1 1536x1280x5120, fc2
  1536x5120x1280, xkv 1536x1280x10240 (1500 frames padded to 1536).
- The export **refuses the set** unless every stream's `final.xclbin` is the same static
  configuration. fc1 and xkv drain C one row block at a time and conv2 has K = 3840, so
  they are the streams that could diverge; a divergence means they cannot share one
  hardware context.
- The set records `design.json` (npu::Design's schema, with `b_layout_hash`, `tg_depth`
  and `tb_max_n_rows`), `toolchain.json` and a `whisper_kernels.json` marker carrying the
  geometry it was built for.
- Each stream is within rel_fro 5e-3 of a float64 reference, from the **exported** set
  rather than a private build.

**Measured 2026-09-19** (mlir-aie 1.4.2.dev16+g7e00b57, Peano 21.0.0.2026080301, built
natively on Windows): all seven xclbins identical up to 70-78 bytes in 11-14 short runs
(UUID and metadata); rel_fro 1.4e-07 to 7.1e-07, per-row cosine 1.000000000.

### OPEN-WHISPER-RELEASE: release builds carry the Whisper kernel set
**Applies to:** `utilities/export-kernels.py`, `open_kernels/export_whisper_kernels.py`,
`utilities/release/stage-prebuilts.sh`, `utilities/release/verify-package.sh`,
`nix/open-kernels.nix`
**Verification:** test (the stream cache: manual, below)
**Acceptance criteria:**
- `utilities/export-kernels.py`, which the CMake `export_kernels` target and
  `stage-prebuilts.sh` run, runs `open_kernels/export_whisper_kernels.py` once, with the
  venv's Python and `--build-dir open_kernels/designs/whisper_gemm/build-export`, after every
  open_kernels spec and BERT command, adding `--force` when it was given `--force`. Every
  command before it is the one the script runs with `--skip-whisper`.
- The set lands in `src/xclbins/Whisper-V3-Turbo-NPU2/open_kernels/`, which staging collects
  (`src/xclbins/*/open_kernels*/` in `.gitignore`) and where the engine looks
  (OPEN-WHISPER-SEAM). The build cache stays out of `src/xclbins`, because everything under it
  ships, and git ignores it.
- `--skip-whisper` and `--bert-only` leave the set out. So does `nix/open-kernels.nix` while its
  `skipWhisper` is true, the default: the set has not been built on that file's pinned
  toolchain. `--specs`, `--skip-specs` and `--skip-bert` name the other two families and do
  not. A failed Whisper export fails the run.
- Without `--force`, as the CMake target runs it, the exporter keeps a stream's build only
  when the build's stamp matches the stream's shape, the bytes of `whisper_gemm.py`,
  `gemm_pretiled.py`, `npue.py` and `build_design.py`, and the mlir-aie and Peano versions,
  the way it already keeps `fa`. Anything else rebuilds the stream.
- `stage-prebuilts.sh` checks a Whisper set by its own layout: a `whisper_kernels.json` that
  says complete, `final.xclbin`, `insts.bin`, every stream file `design.json` names, and
  `fa/air.xclbin`, `fa/air.insts.bin` and `fa/fa.json` when the marker says `fa`. It refuses a
  set that carries `build/`. It also refuses kernel files or a `design.json` directly in an
  `open_kernels/` that has no marker: the exporter deletes the marker before it replaces
  anything, so that is what an export cut short leaves. Every other directory gets the same
  check as before.
- `verify-package.sh` fails a package without `xclbins/<model>/open_kernels/whisper_kernels.json`,
  unless `--allow-missing-kernels`.

**Tests** (`specs/open-whisper/tests/`): `test_release_export.py` drives `export-kernels.py`
with `subprocess.run` and the toolchain checks stubbed: the staging and CMake command lines,
the commands with and without `--skip-whisper` across six flag sets (the Nix one included),
which flags leave the set out, a failed Whisper export, and the build cache's place.
`test_release_staging.py` runs the shape-check block of `stage-prebuilts.sh`, taken from the
script, on built trees: a complete set beside another family, an export cut short, an `--only`
export, a missing stream, a missing `fa/`, a `build/` inside, and the other family's own rule.
`test_release_package.py` runs `verify-package.sh` on a built TGZ with and without the marker,
and with `--allow-missing-kernels`; it needs bash and tar, and is skipped without bash.

**Verification (manual):** the stream cache. The exporter imports IRON (through
`whisper_gemm.py`), so this needs the toolchain's venv, and a real build of the seven streams
takes minutes. In a scratch copy of the exporter and the files it imports, load the exporter
in the release venv, replace the `build_design.py` run with a stub that writes `final.xclbin`
and `insts.bin`, and call `build_stream` for every stream without `--force`: on an empty
cache; again; after writing a shape-only stamp (the format before 2026-10-17) into one
stream; after writing another mlir-aie version into one stream's stamp; after appending a
comment to `gemm_pretiled.py`; after restoring it.

2026-10-17, mlir-aie 1.4.2 and Peano 21.0.0.2026080301: all seven built; none (all kept);
that one stream; that one stream; all seven; all seven again (their stamps named the edited
file). A real export with the new stamp has not been run yet; only the stamp changed, and the
build it guards is the one the proof below ran.

**Proof that the other kernel sets did not change (2026-10-16, one-off, for the release
commit).** The scratch tree held the commit's `open_kernels/`, `npu_offload/`,
`utilities/export-kernels.py` and `.gitignore`, and a `src/xclbins` with the tracked closed
sets plus 35 exported open_kernels sets (1371 files; the rest of the repository was not
copied). The `--force` run was cut by a timeout during `fa` and finished without `--force`,
keeping the seven streams it had built. The step added the set's 16 files, its build cache and
six `__pycache__` files, and changed or removed nothing; all 1001 other files under
`src/xclbins` kept their hashes, and staging collects the same files for every other family.
The old shape check refuses the Whisper set (`fa/` has no `final.xclbin`); the new one passes
it and gives the other 301 kernel directories the same result as the old, and so does the
check with the rule for an export cut short (2026-10-17, same lists). The old and new
`verify-package.sh` agree on every other check; the new one adds the Whisper line. Against the
set exported from upstream/main the same day, every instruction stream is byte-identical and
both xclbins differ only in build stamps.

### OPEN-WHISPER-ENCODER: the encoder on the NPU
**Applies to:** `src/open_whisper/`
**Test category:** hardware (`open_whisper_cli --golden`)
**Acceptance criteria:**
- The GEMMs run as instruction streams over one hardware context, with every B staged once
  at load.
- LayerNorm (biased variance, eps 1e-5), bias, exact-erf GELU, the bidirectional attention
  and the softmax run on the host in fp32.
- Padded rows (1500 -> 1536, 3000 -> 3072) are zero and never enter attention as keys.
- Per layer, chained and teacher-forced, the float64 cosine against the golden is at least
  the bf16 replica's for that clip and layer, less a stated margin.
- The kernel set is refused when its marker, its geometry check or its B layout disagrees
  with the container.
- **No host code writes into a buffer the device writes.** A GEMM's C buffer is mapped
  from the device; a host write there leaves dirty cache lines whose write-back lands on
  top of a later dispatch's result.
- Two runs of the same clip give the same numbers.

**Measured 2026-09-22** on the corrected audio (`open_whisper_cli --golden ... --forced`),
both clips at the replica's bf16 ceiling:

| | Demos_sample-data_journal (replica) | nvidia (replica) |
|---|---|---|
| conv1 / conv2 | 0.99999823 / 0.99999993 (same) | 0.99999860 / 0.99999993 (same) |
| chained `enc.hidden.1` | 0.99999813 | 0.99999812 |
| chained `enc.hidden.32` | 0.99988115 | 0.99992415 |
| `enc.out` | **0.99810825** (0.99799226) | **0.99943178** (0.99922097) |
| cross K/V worst | 0.99759194 | 0.99937406 |
| teacher-forced, all 32 layers | >= 0.99999829 | >= 0.99999827 |

Host wall clock, labelled as such and not an NPU figure: 4.4-4.8 s per 30 s window,
attention ~4.3 s of it, NPU submit+wait ~1.6 s.

**The C-buffer rule is in this list because breaking it is silent.** In-place bias on the
C buffer produced single wrong ROWS (one row at cosine 0.78 inside a matrix reading
0.9998), at an unpredictable layer, in whole 64-byte-aligned runs -- and serialising the
host made it disappear without fixing it. Three probes separate the layers and are kept in
the engine: `--stress N` (512 dispatches cycling four streams and 32 weight slots:
identical), `OW_VERIFY_A=1` (the operand read back off the device: always correct) and
`OW_DOUBLE_CHECK=1` (the same dispatch twice: caught it).

### OPEN-WHISPER-CROSSKV: the encoded window survives clear_context
**Applies to:** `src/open_whisper/`
**Test category:** unit + integration
**Acceptance criteria:** `clear_context()` resets the decoder's self-attention state only.
The cross K/V computed by the last `encode_audio()` stay, because the host calls
`clear_context()` **after** encoding and before the first token.

**Status:** not started (needs the decoder).

### OPEN-WHISPER-DECODER, -E2E: the transcript
**Applies to:** `src/open_whisper/decoder.*`, `src/common/whisper/`
**Test category:** hardware, end to end
**Acceptance criteria:**
- The logits have the padded vocabulary width the host's sampler expects, with the pad tail
  never able to win a sample.
- Against the goldens' forced token paths, argmax agrees at every step.
- Free-running greedy under the host's own protocol reproduces the float64 transcript on
  the golden clips. Differences from the **closed** engine are reported, not treated as
  failures.
**Measured 2026-09-22** on the corrected audio: `open_whisper_cli --decode hf|host
--baseline <file>` over six clips x two protocols, **12/12 pass**. Eleven reproduce
transformers' float64 token path exactly, with teacher-forced argmax agreement 5/5 to
114/114. The twelfth, `Demos_sample-data_journal` under the transformers protocol,
diverges at token 5 and **matches the recorded bf16 path exactly** -- so that is the
datapath, not this engine. `whisper_decode_check.py --write-baseline` records that path
and the gate accepts it while still printing the float64 difference: a gate that cannot
pass is one its reader learns to skip (T64).

Decode cost, host wall clock and not an NPU figure: 12.5-14.6 ms/token, 69-80 tok/s.

### OPEN-WHISPER-ENDPOINT: something tests /v1/audio/transcriptions
**Applies to:** `src/server/server.cpp`, `src/server/rest_handler.cpp`, `utilities/oflm-test`
**Test category:** integration (through `oflm serve --asr 1`)
**Acceptance criteria:**
- A clip of clear speech comes back as text containing what is said in it.
- A request with a missing or empty `file`, or a missing `model`, is refused with **400**
  and an OpenAI-shaped error naming the field. This is the client's error; a 5xx here means
  the request reached something that threw instead of being validated
  (`SERVER-REQUEST-VALIDATION`).
- The response names a model.

**Measured 2026-09-20:** the suite exists (`oflm-test`'s transcription task, run as part of
`--audio`) and passes 3/3 against the closed engine. It was written because the audio suite
posts `chat.completions` with `input_audio`, which Whisper refuses as a non-chat model, so
**nothing exercised the endpoint at all**.

It found one defect immediately: the route built its JSON with `parts["file"]`, and
`std::map::operator[]` default-constructs a missing part, so an absent file read as a
present string of the right type and died in the audio decoder as **HTTP 500**. Fixed by
passing on only the parts that are there, and refusing an empty file. Verified both ways:
the pre-fix binary answers 500 (which T2 now judges FAIL), the fixed one answers 400
`missing_required_parameter`.

### OPEN-WHISPER-HOST-PROTOCOL: the language token the host never feeds
**Applies to:** `src/common/whisper/modeling_whisper.cpp`
**Test category:** offline (goldens under both protocols)
**Acceptance criteria:** the decoder prompt is stated and measured, not assumed. Today the
host samples the language after SOT and then feeds `transcribe`, so the model sees
`[SOT, transcribe, ...]` where transformers sends `[SOT, <lang>, transcribe, ...]`.

**Measured 2026-09-19:** on the six golden clips' first windows, in float64, the two
protocols give **identical text on 6/6**; the only difference is one closing timestamp
(`<|3.24|>` vs `<|1.48|>` on a 3.24 s clip). A wrong closing timestamp does move the next
window's start, because the host chunks from the last timestamp. Any change here is a
separate PR and must be measured on both engines.

### OPEN-WHISPER-PERF: what may be called a performance number
**Applies to:** all of the above
**Acceptance criteria:** NPU figures come from hardware traces or static instruction
counts. Harness and engine wall clock is a host-side observation and is labelled as such,
never quoted as array time. End-to-end wall clock is quoted only against a paired run of
the other engine, on a quiet machine.

### OPEN-WHISPER-ATTN-NPU: encoder attention stays on the host, and why

**Applies to:** the encoder's 32 x 20 attention heads at seq 1500 (1536 padded)
**Acceptance criteria:** the two attention GEMMs move to the array only if a measurement
says the move pays. Whatever is built must keep the single xclbin -- an attention shape
that needs its own design costs a context switch per layer and is refused on that ground
alone.

**Measured 2026-09-22 -- NOT BUILT, and the measurement is the reason.**

The host's three attention phases, instrumented behind `OW_ATTN_PHASES=1` (CPU seconds
summed across threads, so the shares are what matter, not the totals):

| phase | share of attention |
|---|---:|
| Q.K^T | 49.9% |
| row softmax | 12.4% |
| P.V | 37.7% |

The softmax stays on the host in any split design, so the ceiling for moving both GEMMs
-- crediting them as *free* -- is **1.66x** on the encoder as a whole.

Both shapes were built and run. `Q.K^T` tiles per head as 1536 x 64 x 1536; `P.V` has
N = 64 and does not tile, so four heads are grouped block-diagonally into
1536 x 6144 x 256, which costs 4x the arithmetic. Both produce a `final.xclbin` in the
**same equivalence class as the seven production streams** -- same size, differences
confined to the same UUID and name runs -- so they would be two more instruction streams
in the existing context, not a switch. Both are numerically exact against an fp64
reference (rel_fro 5.6e-08 and 8.0e-07).

| shape | GFLOP | per dispatch |
|---|---:|---:|
| `s_h1` 1536 x 64 x 1536 (Q.K^T, one head) | 0.302 | 0.437 ms |
| `pv_g4` 1536 x 6144 x 256 (P.V, four heads) | 4.83 | 1.914 ms |

That is 20 + 5 dispatches per layer, **586 ms of array time per 30 s window** -- against
824 ms for the entire encoder as it ships today.

**The transport is what decides it.** A split design materialises the scores matrix in
host memory: 6.04 GB read back as fp32 C, 3.02 GB written back as bf16 A for the value
GEMM, plus 0.88 GB of operands and output, **9.94 GB per window**. At 20-60 GB/s of host
memory that is 166-497 ms, and it buys nothing -- it is pure overhead that the host path
does not pay, because the host never leaves L1 with a row of scores.

Against a measured 5052 ms encode with 2296 ms of attention:

| | attention | total | speedup |
|---|---:|---:|---:|
| today | 2296 ms | 5052 ms | 1.00x |
| ceiling (GEMMs free) | 285 ms | 3042 ms | **1.66x** |
| built, transport at 60 GB/s | 1036 ms | 3793 ms | 1.33x |
| built, transport at 40 GB/s | 1119 ms | 3876 ms | 1.30x |
| built, transport at 20 GB/s | 1368 ms | 4125 ms | 1.22x |

**Decision: not built.** A split design collects at most 1.33x of a 1.66x ceiling, and
half of what it gives back is a cost that exists only because the design splits. The form
that avoids it is the chained one -- Q.K^T, softmax and P.V on the array with the scores
never leaving it -- which needs a row-softmax kernel, the one genuinely new kernel in this
whole effort. whisper-xdna (MIT) measured 1.73x for exactly that shape at ctx 1500 on
XDNA1. That is the version worth building, and it is not a variant of this one.

## Not settled

- **The shipped `whisper-v3:turbo` entry still points at the closed container.** The open
  engine no longer needs a converted repository (OPEN-WHISPER-STOCK, -GGUF): `oflm-add`
  installs OpenAI's checkpoint or a GGUF under another tag (OPEN-WHISPER-INSTALL). Whether
  the default tag should move to one of those is the maintainer's call.
- **The decoder** runs on the host first. Moving it to the NPU needs an `attn.h` variant
  with no RoPE and no cache append, plus a decoder layer design with LayerNorm, biases and
  a non-gated GELU.
