# The open text-to-speech engine

Kokoro-82M (`hexgrad/Kokoro-82M`, Apache-2.0, v1.0) on the host CPU, in C++, behind
`POST /v1/audio/speech`. This file is what the engine must do and what has been measured.
The server's half is `SERVER-SPEECH-*` in `specs/server-api/spec.md`.

Directory name gives the prefix: `OPEN-TTS`.

The model is StyleTTS 2 with an ISTFTNet vocoder and no diffusion: a PL-BERT text encoder
(ALBERT, one layer shared 12 times, hidden 768), a text encoder (three convs and a BiLSTM), a
prosody predictor (BiLSTMs with adaptive layer norm for durations; AdaIN residual blocks for F0
and energy), and a decoder that turns the frame-aligned features into a waveform through two
transposed convs, Snake resblocks and an inverse STFT, driven by a harmonic-plus-noise source
built from the F0 curve. 82M parameters, 24 kHz mono. Voices are tables of 510 style vectors.

Version 1 is English only (the American `a*` and British `b*` voices), WAV and raw PCM out, no
NPU. The code is `src/open_tts/`; the reference is `utilities/kokoro-ref/ref_kokoro.py`.

**The reference is hexgrad's `kokoro` 0.9.4 package** on the CPU in float32, with the one random
input the forward pass has replaced by a seeded stream (OPEN-TTS-NOISE). It is the model as its
author ships it, not a re-derivation.

## Requirements

### OPEN-TTS-WEIGHTS: the checkpoint is read as stored, weight norm folded at load
**Applies to:** openflowlm-next (`src/open_tts/kokoro.cpp`, `src/open_tts/safetensors.cpp`)
**Verification:** test (the loader, on fixtures) and manual (the real file)
**Test:** `src/open_tts/ops_test.cpp` (CTest `open_tts_ops`)

There is no GGUF from the author, and the author's own checkpoint is a PyTorch pickle
(`kokoro-v1_0.pth`) that a C++ engine would need a pickle reader for. The engine reads a
safetensors conversion instead: `kokoro-v1_0.safetensors` or `model.safetensors` beside
`config.json`, in F32, F16 or BF16. Tensors are found by their PyTorch module path
(`decoder.generator.ups.0.weight_v`); mlx-audio's renames (`forward_lstm.Wx`, LayerNorm
`weight`/`bias` for `gamma`/`beta`) and MLX's channels-last conv layout are accepted too.
`torch.nn.utils.weight_norm` (dim 0) is folded at load, `w[i] = g[i] * v[i] / ||v[i]||` over
row `i` of dim 0, which for a transposed conv is the *input* channel. A `config.json` that is not
v1.0's geometry is refused, because the forward pass is written for that geometry.

The matrix weights (every conv, transposed conv, linear and LSTM matrix: 99.7% of the
parameters) are held in 16 bits, the weights rule for small engines: IEEE f16, rounded to nearest
after the fold, 164 MB. The GEMM widens them to f32 as it packs them, and all arithmetic is f32.
f16 rather than bf16 because a host engine has no bf16 hardware to match, and f16 keeps three
more mantissa bits in the same memory. Embeddings, norms, biases and the Snake alphas stay f32.
`OFLM_TTS_WEIGHTS=f32` holds everything in f32, for the parity check.

**Acceptance criteria:**
- Every weight-normed block in OPEN-TTS-BLOCKS loads from `weight_g`/`weight_v` and matches
  PyTorch, so the fold is checked wherever it is used (ops_test).
- f16 storage rounds to nearest even, keeps 65504, 2^-14 and 2^-24 exact, and a conv or linear
  with f16 weights equals a float64 product of the same rounded weights within 1e-5 (ops_test).
- A tensor of the wrong size is refused naming it.
- Manual: the chosen conversion's 548 tensors equal hexgrad's `.pth` bit for bit
  (`verify_sources.py`, see "Sources" below).

### OPEN-TTS-VOICES: a voice is a style table, and the row follows the phoneme count
**Applies to:** openflowlm-next (`src/open_tts/kokoro.cpp` `Voice`)
**Verification:** manual (OPEN-TTS-PARITY uses it)

`voices/<name>.safetensors` holds one tensor of 510 x 256 (`[510, 1, 256]` or `[510, 256]`).
KPipeline speaks a chunk with row `len(phonemes) - 1`, counting code points of the phoneme
string (characters the vocabulary drops still count), and so does the engine. The first 128
values style the decoder, the last 128 the prosody predictor.

### OPEN-TTS-OPS: the host kernels equal float64
**Applies to:** openflowlm-next (`src/open_tts/ops.cpp`)
**Verification:** test
**Test:** `src/open_tts/ops_test.cpp` (CTest `open_tts_ops`)

The AVX2/OpenMP GEMM (6x16 micro-kernel), convolutions as one GEMM whose B rows are shifted
pointers into the padded input (no im2col copy), transposed convolutions as a GEMM plus a
scatter, the bidirectional LSTM, and the STFT pair.

**Acceptance criteria:**
- `sgemm` at 7x33x300, 97x130x513, 1x17x5 and 200x1x64 is within 1e-5 (relative L2) of a
  float64 product; partial tiles in both dimensions are covered.
- Conv1d with dilation 3, with stride 6, depthwise, and with 514 input channels; ConvTranspose1d
  dense (k20 s10) and depthwise (k3 s2 with output padding): each within 1e-5 of float64 loops.
- The BiLSTM is within 1e-5 of a float64 recurrence; `istft(stft(x))` returns `x` within 1e-5.

### OPEN-TTS-BLOCKS: each block is the PyTorch module
**Applies to:** openflowlm-next (`src/open_tts/kokoro.cpp`, `src/open_tts/ops.cpp`)
**Verification:** test
**Test:** `src/open_tts/ops_test.cpp` (CTest `open_tts_ops`), fixtures from
`src/open_tts/testdata/gen_fixtures.py`

`gen_fixtures.py` builds small blocks from kokoro 0.9.4's own classes with random weights
(`torch.manual_seed(0)`), runs them, and stores weights, inputs and outputs in
`testdata/blocks.safetensors` (about 40 KB; no model weights). The C++ loads the same weights
through the engine's loaders and runs the engine's forward code.

**Acceptance criteria (relative L2 against PyTorch's output):**
- `AdainResBlk1d` with and without its upsampling path, and `AdaINResBlock1` (Snake, dilations
  1/3/5, AdaIN with an affine InstanceNorm): within 2e-5.
- `nn.LSTM` bidirectional, the weight-normed `ConvTranspose1d(4, 3, 20, 10, padding=5)`,
  `Conv1d(22, 4, 12, stride 6)`, the channel LayerNorm and `AdaLayerNorm`: within 1e-5.
- `torch.stft` as complex values and as magnitudes, and `torch.istft`: within 1e-5.
- The harmonic source (`SineGen` + `SourceModuleHnNSF`), with and without its noise: within
  1e-4. Its phases reach hundreds of radians in float32, where an ulp is 1e-5 rad.
- The tanh GELU: within 1e-6.

### OPEN-TTS-PARITY: the whole model against the reference
**Applies to:** openflowlm-next (`src/open_tts/`), `utilities/kokoro-ref/ref_kokoro.py`
**Verification:** manual (needs the model files)

**Verification (manual):**
1. `python ref_kokoro.py --pylib <kokoro 0.9.4> --config config.json --weights kokoro-v1_0.pth
   --voice voices/af_heart.safetensors --text "<sentence>" --seed 1234 --out ref.safetensors`
   (hexgrad's own `.pth`, so the reference is the author's checkpoint, not the conversion).
2. `kokoro_cli --model <install dir> --check ref.safetensors` prints, per block, the relative
   L2 error and cosine against the dump: `bert`, `d_en`, `pred.d`, the float durations, `pred.F0`,
   `pred.N`, `t_en`, `dec.x`, the harmonic source, the generator's stages, `conv_post` and the
   waveform; then the generator alone on the reference's inputs.
3. Pass, with `OFLM_TTS_WEIGHTS=f32`: integer durations identical, every block before the
   generator within 1e-5, and the generator on the reference's inputs within 1e-3. The free
   run's waveform is compared by log-mel distance (`utilities/kokoro-ref/mel_distance.py`)
   against the reference's own run-to-run spread (another noise seed), not sample by sample:
   the harmonic source integrates F0 into phase, so F0's float32 rounding becomes a phase
   drift that grows along the utterance. (Measured values: see "Results".)

### OPEN-TTS-NOISE: a request repeats exactly
**Applies to:** openflowlm-next (`src/open_tts/kokoro.cpp` `harmonic_source`, `src/open_tts/ops.cpp` `Gaussian`)
**Verification:** test
**Test:** `src/open_tts/ops_test.cpp` (the stream against `ref_kokoro.gaussian`, the source with
seeded noise); `oflm-test --tts` S2 on the wire

Kokoro's forward pass has two random inputs. `SineGen` adds a random initial phase per harmonic
to sample 0 only, and the 1/300 linear downsample that follows reads samples 149 and 150 of each
block, never sample 0, so that input has no effect at all. The other is the Gaussian noise added
to the excitation (std 0.003 where voiced, 0.033 where not). The engine draws it from a
splitmix64 + Box-Muller stream with a fixed seed, which `ref_kokoro.py` reproduces in numpy, so
the reference repeats and the engine matches it. Seed 0 turns the noise off.

### OPEN-TTS-G2P: English phonemes from misaki's lexicons, without spaCy or espeak-ng
**Applies to:** openflowlm-next (`src/open_tts/g2p_en.cpp`)
**Verification:** test (misaki's output on a lexicon excerpt) and manual (agreement on a corpus)
**Test:** `src/open_tts/g2p_test.cpp` (CTest `open_tts_g2p`), fixtures from
`src/open_tts/testdata/gen_g2p_fixtures.py`

Kokoro is trained on misaki's phonemes. misaki 0.9.4 (Apache-2.0) is ported for its dictionary
path: the gold and silver lexicons, its special cases (a, an, am, I, to, in, the, by, used, vs),
stress rules (`apply_stress`, capitalised words 0.5, all caps 2), the -s/-ed/-ing stems, numbers
(num2words' English for cardinals, ordinals, years, decimals, currency in $ £ €, `%`), its
punctuation mapping, the sub-token regex for words with digits, hyphens and camel case, and
the flap and glottal-stop substitutions it makes for v1.0 (`ɾ` -> `T`, `ʔ` -> `t`).

spaCy is replaced by a tokenizer that splits off what spaCy splits off (quotes, brackets, final
punctuation unless the word is an abbreviation, dashes, currency signs), and by fixed tags where
misaki asks the tagger: `a` is a determiner, `A` is one at a sentence start and a letter
elsewhere, `I` is a pronoun, `in` and `vs` are prepositions, `used` before `to` is past tense,
`that` before a subject or determiner is a conjunction (unstressed) and a determiner otherwise.
Words with per-tag pronunciations (790 in the gold lexicon: read, live, lead...) take their
DEFAULT, or the `None` variant at the end of a sentence, as misaki does for an unknown tag.

espeak-ng, misaki's fallback for words the lexicons lack, is GPL-3 and is not used. An unknown
word is spelled letter by letter (misaki's own acronym reading, `get_NNP`) by default, so the
listener hears that something was there; `OFLM_TTS_UNKNOWN=skip` drops it instead, which is
what KPipeline does without espeak. Either way the word is reported.

**Acceptance criteria:**
- num2words: 1234, 2,500,000, 1001 as cardinals; 21, 100, 20 as ordinals; 1984, 1905, 2001,
  2010 as years, each the string num2words gives.
- 14 sentences covering punctuation, the/a, stems, numbers, years, ordinals, decimals,
  currency, percent, an acronym, an abbreviation, a hyphenated word, a contraction, an em dash
  and an ellipsis: each the exact phoneme string misaki 0.9.4 gives with the same lexicon
  excerpt (`testdata/g2p_*.json`, about 300 entries of misaki's `us_gold`/`us_silver`).
- An unknown word is reported, spelled as misaki's `get_NNP` spells it, and dropped in skip mode.
- Manual: agreement with misaki + spaCy on a mixed corpus, and the unknown-word rate (Results).

### OPEN-TTS-CHUNK: long text is spoken in chunks the model accepts
**Applies to:** openflowlm-next (`src/open_tts/tts_engine.cpp`)
**Verification:** manual

Kokoro takes at most 510 phonemes per call. As KPipeline does, the text is split at newlines,
each paragraph is phonemized, and a paragraph longer than 510 phonemes is cut after the last
`. ! ? …` that fits, else `: ;`, else `, —`, else at a space. The chunks' audio is concatenated.

**Verification (manual):** `kokoro_cli --model <dir> --g2p-chunks <a 2700-character paragraph>`
prints one line per chunk with its length: each at most 510 and ending at sentence punctuation
(2026-10-17: 8 chunks, 179 to 510 phonemes, all ending at a full stop).

### OPEN-TTS-INSTALL: oflm-add installs Kokoro with its voices and lexicons
**Applies to:** openflowlm-next (`utilities/oflm-add/oflm_add/__init__.py` `install_kokoro`)
**Verification:** manual

`oflm-add <repo>` with "kokoro" in the name (or `--family kokoro`) installs `config.json`, the
safetensors checkpoint, every English voice (`voices/[ab][fm]_*.safetensors`) and misaki
0.9.4's four lexicons, fetched from its PyPI wheel and checked against pinned SHA-256 hashes (the
repository never carries them). It registers `kokoro:82m` in the user registry with `tts: true`,
family `kokoro`, format `safetensors`, label `["audio", "text-to-speech"]` and no download URL.
A repo with only the `.pth` (hexgrad's own) is refused with a pointer to a safetensors
conversion; a config that is not v1.0's is refused.

**Verification (manual):** `oflm-add mlx-community/Kokoro-82M-bf16 --config <scratch registry>
--models-root <scratch>/models`, then `oflm serve --tts 1` loads it (see Results).

## Sources (2026-10-16)

Surveyed on Hugging Face: 60 Kokoro repos. The author's repo has the `.pth` and `.pt` voices.
Safetensors conversions with the original module names: `mlx-community/Kokoro-82M-bf16`
(73k downloads; F32 despite its name; config, weights and every voice as safetensors),
`prince-canuma/Kokoro-82M` (the same file), `round-tower/Kokoro-82M-bf16` (BF16, the same 548
names, no voices, 115 downloads). ONNX (`onnx-community`, 686k downloads) and CoreML/OpenVINO
need other readers. GGUFs exist after all (`cstr/kokoro-82m-GGUF` F16 and Q8_0, for CrispASR, and
four smaller repos), contrary to the feasibility doc; none carries voices.

Chosen: `mlx-community/Kokoro-82M-bf16` at a71e4d38. One repo carries config, weights and
voices in a format the engine reads without a pickle parser, and its 548 tensors and the voices
checked are bit-identical to hexgrad's `.pth`/`.pt` (Results).

## Results

Measured 2026-10-17 on this box; logs in the branch's `.claude/plans/kokoro-run/` (gitignored).

- **Sources** (`verify-mlx-vs-original.log`): all 548 tensors of `mlx-community/Kokoro-82M-bf16`'s
  `kokoro-v1_0.safetensors` equal hexgrad's `kokoro-v1_0.pth` bit for bit (81,763,410
  parameters, all F32), and so do the three voices compared (af_heart, am_adam, bf_emma).
- **Blocks** (`ops_test.log`): every OPEN-TTS-OPS and OPEN-TTS-BLOCKS case passes; against
  PyTorch the f32 blocks are within 5e-7 relative L2, the harmonic source within 3e-8; with f16
  weights the blocks are within 3e-4.
- **Whole model**, one sentence (112 tokens, 7.15 s, af_heart, seed 1234), against `ref_kokoro.py`
  on hexgrad's `.pth` (`check-f32-vs-ref.log`, `check-f16-vs-ref-f32.log`):
  - f32 weights: the integer durations are identical (112 of 112); PL-BERT 2.0e-6, the duration
    encoder 1.6e-6, F0 3.7e-7, energy 1.8e-7, text encoder 2.5e-7, decoder output 4.2e-7
    (relative L2). The generator alone, on the reference's inputs, gives the waveform within
    2.0e-4. The free run's waveform is within 1.5e-2 (36 dB SNR); the difference enters at the
    harmonic source, whose phase integrates F0 (2.8e-3 there from F0's 3.7e-7).
  - f16 weights (the default): durations identical again; blocks within about 6e-4.
- **How it sounds**, log-mel distance to the f32 reference (`mel-distance-ref-sentence.log`;
  mean |dB| over voiced frames): the engine with f32 weights 0.10 dB; with f16 weights 0.46 dB;
  PyTorch itself with f16-rounded weights 0.48 dB; PyTorch with another noise seed 0.48 dB. So
  16-bit weights move the audio about as much as Kokoro's own noise does from one run to the
  next, in PyTorch as much as here.
- **Noise off is the worse comparison** (`check-f32-vs-ref-noiseoff.log`, waveform within 9e-2):
  without the noise an unvoiced stretch of the source is a constant, the STFT phase of a
  near-zero bin is undefined, and the two implementations' rounding picks different angles.
- **Phonemizer** (`g2p_test.log`, `g2p-agreement-3.log`, `g2p-agreement-3--british.log`,
  `g2p-agreement-that.log`): the 14 misaki cases are exact. On a 36-line mixed corpus (chat,
  news, prose, numbers, names, tech; 470 words) 34 lines are identical to misaki 0.9.4 with
  spaCy and 99.6% of the phoneme words agree; the two differences are the tagger's (`restart`
  as a noun, the `-` of `-m` read as a dash). British, on the corpus's first 18 lines: 18 of 18
  identical. The `that` rule (unstressed before a subject or determiner, stressed otherwise) was
  tuned on the corpus, so it was checked on ten more sentences: 9 of 10 match misaki ("the idea
  that people change" does not). Unknown words: 34 (7.2% of the words), exactly misaki's count,
  all names (Siobhan, Nguyen, Saoirse, Kilimanjaro, Xiaomi, Tchaikovsky...) and tech terms
  (Kubernetes, Redis, nginx, softmax, tokenizer, PostgreSQL, pytest).
- **Chunking** (`chunks-long-paragraph.log`): the corpus as one 2702-character paragraph splits
  into 8 chunks of 179 to 510 phonemes, each ending at a full stop.

## Not settled

- mp3, opus, aac and flac output: refused (SERVER-SPEECH-PARAMS). The bundled FFmpeg has no mp3
  encoder; its built-in opus, aac and flac encoders are not wired to the endpoint.
- Words with per-tag pronunciations always take the DEFAULT reading; a tagger would fix read/
  live/lead-type heteronyms.
- Other languages: misaki's other G2Ps (Japanese, Mandarin, ...) are separate ports.
- The NPU: the decoder's 1024-channel convs and the ALBERT GEMMs are the parts that could move.
