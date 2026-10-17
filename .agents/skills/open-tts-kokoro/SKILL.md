---
name: open-tts-kokoro
description: Install, serve, check and extend Kokoro-82M text to speech on the host CPU (src/open_tts, misaki's English G2P in C++, POST /v1/audio/speech). Use when installing Kokoro, adding a voice or a language, debugging speech that sounds wrong, touching the phonemizer, or porting another TTS model onto the same ops.
---

# Skill: open-tts-kokoro

Kokoro-82M text to speech on the host CPU (`src/open_tts`), served as `POST /v1/audio/speech`.
Use when: installing Kokoro, adding a voice or a language, debugging speech that sounds wrong,
touching the phonemizer, or porting another TTS model onto the same ops.

## What runs where

- `safetensors.cpp`: a read-only safetensors reader (F32/F16/BF16/F64).
- `ops.cpp`: the host kernels. One AVX2 GEMM (6x16 micro-kernel, OpenMP over 96-row tiles); a
  Conv1d is that GEMM with B's rows pointing at shifted rows of the padded input (no im2col);
  a ConvTranspose1d is a GEMM into [k*cout][t] plus a scatter; LSTM gates are one GEMM, then the
  recurrence; STFT/iSTFT are direct DFTs (n_fft 20). Weights are held as IEEE f16 and widened
  while the GEMM packs them (`OFLM_TTS_WEIGHTS=f32` keeps f32, for checks).
- `kokoro.cpp`: the model. `WeightReader` finds tensors by PyTorch module path and folds
  `weight_g`/`weight_v`; `blocks::` builds each module; `Kokoro::synthesize` is
  `KModel.forward_with_tokens` step for step; `harmonic_source` is SineGen + SourceModuleHnNSF.
- `g2p_en.cpp`: misaki 0.9.4's English lexicon path, ported (no spaCy, no espeak-ng).
- `tts_engine.cpp`: text -> paragraphs -> phonemes -> <=510-phoneme chunks -> audio; WAV/PCM.
- Server: `rest_handler.cpp handle_openai_audio_speech`, rules in `openai_compat::speech_request`,
  binary responses through `openai_compat::binary_body`.

## Install and serve

```
oflm-add mlx-community/Kokoro-82M-bf16        # registers kokoro:82m (tts: true)
oflm serve --tts 1 [--ttsmodel kokoro:82m]   # beside any chat model; CPU only
curl -s localhost:52625/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"kokoro:82m","input":"Hello there.","voice":"af_heart"}' -o hello.wav
```

The install holds config.json, `kokoro-v1_0.safetensors` (F32 in that repo despite its name),
the 28 English voices and misaki's lexicons (`lexicon/`, from its PyPI wheel, hash-pinned).
hexgrad's own repo has only the `.pth` and is refused with a pointer.

## Checking it

- `src/open_tts/build.cmd`: kokoro_cli + ops_test + g2p_test, run. The fixtures come from
  `testdata/gen_fixtures.py` and `testdata/gen_g2p_fixtures.py` (kokoro and misaki 0.9.4 on
  the path, torch for the first).
- Whole model: `utilities/kokoro-ref/ref_kokoro.py` dumps every block for one phoneme string
  (hexgrad's `.pth`, noise from the shared seeded stream), then
  `kokoro_cli --model <dir> --check ref.safetensors`. Run the reference with `--round f16` to
  compare against the engine's 16-bit weights exactly; without it you see the 16-bit cost.
- More sentences: `ref_kokoro.py --text "<sentence>" --seed 1234 --out ref-N.safetensors --wav ref-N.wav`,
  then `kokoro_cli --check ref-N.safetensors --out engine-N.wav` and
  `utilities/kokoro-ref/mel_distance.py ref-N.wav engine-N.wav` (log-mel dB; the free run drifts in
  phase, so compare by mel, not sample by sample).
- Phonemizer against misaki + spaCy: `utilities/kokoro-ref/g2p_agreement.py --pylib <misaki 0.9.4>
  --corpus g2p_corpus.txt --cli kokoro_cli.exe --model <dir>` (also prints the unknown-word rate).
- Server: `specs/server-api/tests/test_speech_api.py` with `OFLM_TEST_BASE_URL` pointing at an
  `oflm serve --tts 1`; `oflm-test --tts --model kokoro:82m --port <port>`. S6 (Whisper round trip)
  needs a Whisper model on the same server (`--asr 1`); otherwise it is SKIPPED.
- Round trip through Whisper on another server: `utilities/kokoro-ref/round_trip.py --port <whisper
  server> --text round_trip.txt --wav engine-{i}.wav` (word error rate; numerals scored as words).
- The CTests: `open_tts_ops`, `open_tts_g2p`, `openai_compat` (the request rules).

## Traps

- SineGen's random initial phase lands on sample 0, which the 1/300 linear downsample never
  reads: it has no effect. The only randomness is the excitation noise, now seeded.
- A transposed conv's weight norm is per INPUT channel (dim 0 of [in, out, k]).
- `torch.round` is round-half-even (`std::nearbyint`), and durations are sums of 50 sigmoids:
  a duration within 1e-6 of .5 can flip a frame between implementations. `--check` reruns with
  the reference's durations when that happens.
- The style row is `len(phonemes) - 1` in code points, including characters the vocab drops.
- misaki maps `ɾ` -> `T` and `ʔ` -> `t` unless version 2.0 is asked for; Kokoro v1.0 was trained
  on the mapped form.
- MSVC needs `/utf-8` for g2p_en.cpp's literals, or the phoneme sets are cp1252 garbage.
