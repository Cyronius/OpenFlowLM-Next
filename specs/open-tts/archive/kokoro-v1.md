# Plan: Kokoro-82M text to speech, version 1

2026-10-16. New spec home `specs/open-tts/`, prefix `OPEN-TTS`, plus `SERVER-SPEECH-*` in
`specs/server-api/spec.md`. No TTS existed in the tree before this.

Scope: English (American and British voices), WAV and raw PCM out, a host (CPU) engine in C++, and
`POST /v1/audio/speech` in OpenAI's shape. No NPU: the model runs faster than real time on the
CPU, and the NPU would add little (feasibility doc, track T, 5b).

## New requirements

- OPEN-TTS-WEIGHTS: the engine reads a Kokoro v1.0 safetensors file as stored (F32, F16 or BF16),
  by PyTorch module name, and folds weight norm at load. Other geometries are refused.
- OPEN-TTS-VOICES: voice tables from `voices/<name>.safetensors`; the style row is
  `len(phonemes) - 1`, KPipeline's rule.
- OPEN-TTS-OPS: the host GEMM, convolutions, LSTM and STFT equal float64 references.
- OPEN-TTS-BLOCKS: each Kokoro block equals hexgrad's PyTorch module on random-weight fixtures.
- OPEN-TTS-PARITY: the whole model against the PyTorch reference, block by block (manual).
- OPEN-TTS-NOISE: the harmonic source's noise is a seeded stream, so a request repeats exactly.
- OPEN-TTS-G2P: misaki 0.9.4's dictionary path in C++, without spaCy and without espeak-ng.
- OPEN-TTS-CHUNK: text splits at newlines and into chunks of at most 510 phonemes.
- OPEN-TTS-INSTALL: `oflm-add` installs a Kokoro safetensors repo with its voices and lexicons.
- SERVER-SPEECH, SERVER-SPEECH-PARAMS, SERVER-SPEECH-LOAD, SERVER-SPEECH-QUEUE.

No existing requirement changes. `SERVER-MODEL-IDENTITY` applies to the new endpoint as written.

## Steps

1. Sources: survey the conversions, check one against hexgrad's `.pth`, pick it.
2. Phonemizer: port misaki's `en.py` lexicon path; measure agreement and the unknown-word rate.
3. Reference: `utilities/kokoro-ref/ref_kokoro.py`, deterministic, every block dumped.
4. Engine: `src/open_tts/`, checked block by block (`kokoro_cli --check`).
5. Server, flags, registry label, `oflm-add` install, `oflm-test --tts`.
6. Validation: waveform against the reference, a Whisper round trip, `oflm-test --tts`.
