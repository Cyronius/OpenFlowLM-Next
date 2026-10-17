# Skill: open-whisper-weights

How the open Whisper engine (`src/open_whisper`) gets its weights, and how to install
whisper-large-v3-turbo for it from OpenAI's checkpoint or a GGUF. Use when adding another
Whisper file source, debugging a Whisper model that refuses to load, installing Whisper
with `oflm-add`, or touching `GgufFile`'s whole-tensor reads.

## What the engine reads (2026-10-16)

`weights_present()` in `src/open_whisper/weight_source.cpp` checks the model directory for,
in this order, and the engine reads the first:

1. `model.open.safetensors` + `weights_manifest.json`: the container q4nx-build writes
   (`q4nx-build --open-whisper`, OPEN-WHISPER-CONTAINER). Read as it is.
2. `model.safetensors`: OpenAI's checkpoint, transformers' names, F16 (BF16/F32 accepted).
3. `model.gguf`: the `stt.whisper` GGUF layout (handy-computer/whisper-large-v3-turbo-gguf:
   `enc.blocks.N.attn.q.weight`, `dec.blocks.N.cross_attn.k.weight`, ...), read through the
   same `open_qwen36::GgufFile` the LLM engine uses.

For 2 and 3 a `DerivedSource` builds each container tensor at load the way
`whisper_tensors()` in `utilities/q4nx-build/q4nx/open_whisper.py` does (Q|K|V fused with a
zero K bias, the decoder's cross K|V as one operand with bias 0|bv0|0|bv1..., conv stems as
tap-major im2col, encoder operands as W^T, decoder [out, in], f32 biases/norms/positions,
bf16 by round-to-nearest-even). `Weights` and `Decoder` only ever see container tensors, so
tiling, encoder and decoder code are shared by all three.

GGUF types the reader widens: F32, F16, BF16, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K (llama.cpp's
dequant loops in `gguf_file.cpp dequant_row`). Every tensor is widened to bf16 at load
whatever the file, so a quant saves download and disk, not memory. After that the decoder
requantizes its linear layers and head to int8 by default (`OW_DEC_W`, `OW_DEC_HEAD`; `bf16`
restores the exact path), whichever file it came from.

## Install

```
oflm-add handy-computer/whisper-large-v3-turbo-gguf [--quant Q8_0|F16|Q6_K|Q5_K_M|Q4_K_M]
oflm-add openai/whisper-large-v3-turbo
oflm serve --asr 1 --asrmodel whisper-v3:turbo-q8_0      # or whisper-v3:turbo-safetensors
```

`install_whisper()` in `utilities/oflm-add/oflm_add/__init__.py`:

- A GGUF has no config/tokenizer files. config.json, tokenizer.json, tokenizer_config.json and
  generation_config.json come from the checkpoint the GGUF names in `general.repo_url` (else
  README `base_model`), then are checked against the GGUF metadata (geometry, every vocab id,
  merges, special ids, suppress lists). They are not rebuilt from the metadata: it lacks
  `max_initial_timestamp_index` and which added tokens are special.
- tokenizer_config.json gains `bos_token_id` and `eos_token_id` (the host reads both; OpenAI's
  file has neither). The result is byte-identical to the container directory's file.
- Each GGUF quant gets its own directory, `<repo directory>-<quant>` (e.g.
  `whisper-large-v3-turbo-gguf-Q8_0`): with one shared directory, a second `--quant` install
  registered its tag over the first quant's `model.gguf`. `--force` re-downloads what the HF
  cache did not supply.
- Tags live in the `whisper-v3` bucket and `--tag` is refused outside it: `rest_handler`/`runner`
  refuse an `--asrmodel` tag that does not start with `whisper`, and `model_list.hpp` hides
  only that bucket from `/v1/models`. `whisper-v3:turbo` itself is refused (see Traps).
- No kernel link: the engine finds the kernel set by geometry (below).

## Kernels belong to the shape

`find_open_kernels()` in `src/common/whisper/whisper_engine_select.cpp`:
`OFLM_WHISPER_KERNELS_DIR`, then `<model>/open_kernels`, then
`<root>/xclbins/<model dir name>/open_kernels`, then any complete set under any xclbins root
whose `whisper_kernels.json` `hf_config_check` matches the model's config.json. The export
(`open_kernels/export_whisper_kernels.py`) writes `src/xclbins/Whisper-V3-Turbo-NPU2/open_kernels`;
a GGUF installed as `whisper-gguf-Q8_0` finds it by the last rule. The load line names the rule.

Releases do not ship this set yet: `utilities/export-kernels.py` (the CMake `export_kernels`
target and `utilities/release/stage-prebuilts.sh`) builds the recipe specs and the BERT sets,
not Whisper. Until it does, the open engine runs only where someone ran the export by hand.

## Tests

- `src/open_whisper/weight_source_test.cpp`, built and run by `src/open_whisper/build.cmd`:
  a tiny Whisper (d 32, 1 encoder + 2 decoder layers) as checkpoint, F16 GGUF and Q8_0 GGUF
  against hashes of the converter's own output; K-quant rows against the gguf package; the
  name map against the real turbo GGUF's tensor list (`testdata/whisper_turbo_gguf_tensors.json`);
  a GGUF and a checkpoint cut short by 64 bytes, both refused.
  Fixtures: `python src/open_whisper/testdata/gen_weight_fixtures.py` (needs numpy, gguf;
  imports q4nx-build's converter). `.gitignore` has a `!` rule so the tiny `.gguf` fixtures
  are tracked.
- Real files: `weight_source_test --compare <model.open.safetensors> <model.gguf|model.safetensors>`
  compares all 481 tensors (about 30 s each).
- `utilities/oflm-add/tests/test_whisper_install.py`; `oflm-test --audio --model <whisper tag>`
  now runs only the transcription checks under that tag.

## Results on this box (2026-10-16)

- `--compare`: stock checkpoint and F16 GGUF vs the checkpoint's container, Q8_0 GGUF vs
  the converter's Q8_0 container, and Q4_K_M / Q5_K_M vs the converter run on the gguf
  package's dequant of the same file: 481/481 tensors, 808,924,160 values identical each.
- Hardware smoke (30 s clip, `oflm serve --asr 1` on port 52631): the Q8_0 GGUF and the stock
  checkpoint transcribe exactly as the containers built from the same weights, served by the
  same build, and with the speed defaults off they reproduce the September transcripts word
  for word. The default rule picks the open engine with `OFLM_WHISPER_ENGINE` unset.
  `oflm-test --audio --model whisper-v3:turbo-q8_0` 3/3. Details in
  specs/open-whisper/spec.md (OPEN-WHISPER-GGUF, -STOCK).

## Traps

- A Whisper registered as `whisper-v3:turbo` inherits that tag's `model_info.json` records,
  so `oflm serve` sizes config.json against the closed container's, calls it missing and
  downloads `model.q4nx` (seen in the first smoke). `oflm-add` now refuses that tag; OpenAI's
  config.json has no `oflm_version`, so no version re-pull either.
- `GgufFile::f32()` used to read only `dims[1]` rows, truncating 3-D tensors (the conv
  kernels). It now counts every dim past the first.
- `GgufFile` checked each tensor's end against the file size before it knew where the data
  section starts, so a GGUF cut short by less than its header loaded with zeros for the
  missing bytes. It now checks after. `Q4nxFile` refuses a cut checkpoint but keeps the file
  mapped when its constructor throws, so the test ignores a failed delete of its cut copy.
- When rebuilding `weight_source_test` after a mutation check, rebuild before `--compare`:
  a stale binary reports real differences that are not there.
- Compare a transcript against a container served by the same build, not against an old
  transcript. Since #111 (2026-09-25) the defaults are the `hf` protocol, int8 decoder weights
  and head, bf16 cross K/V, fused host ops and NPU FlashAttention; together they move two
  marginal words on the 30 s audiobook clip ("Jessop"/"Jessup", "seem"/"seemed"), for the
  container itself too. Host attention alone does not bring the old text back for the
  checkpoint. To compare with a pre-#111 transcript, switch them all off:
  `OFLM_WHISPER_PROTOCOL=legacy OW_ATTN=host OW_HOST_FAST=0 OW_DEC_W=bf16 OW_DEC_HEAD=bf16
  OW_DEC_XKV=fp32`. With that, the Q8_0 GGUF and the checkpoint each reproduced their September
  transcript word for word (2026-10-16). Which default moves which word was not isolated.
