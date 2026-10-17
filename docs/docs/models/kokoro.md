---
layout: docs
title: kokoro
nav_order: 14
parent: Models
---

## 🧩 Model Card: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

- **Type:** Text-to-Speech
- **Base Model:** [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) v1.0 (Apache-2.0), StyleTTS 2 with an ISTFTNet vocoder
- **Installed from:** [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) (the same values as hexgrad's checkpoint, in safetensors)
- **Weights:** held in 16 bits (f16) on the host
- **Runs on:** the CPU (AVX2), not the NPU
- **Languages:** English (American `a*` and British `b*` voices)
- **Output:** 24 kHz mono: WAV, raw PCM, FLAC, AAC, or Opus (48 kHz); no mp3

▶️ Install it and serve it (server mode only):

```shell
oflm-add mlx-community/Kokoro-82M-bf16
oflm serve llama3.2:1b --tts 1
```

```shell
curl -s http://127.0.0.1:52625/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model": "kokoro:82m", "input": "Hello there.", "voice": "af_heart"}' -o hello.wav
```

The text goes through a C++ port of [misaki](https://github.com/hexgrad/misaki), the phonemizer
Kokoro was trained with, using its English lexicons (fetched from misaki's PyPI release at install
time). misaki's espeak-ng fallback is not used, so a word missing from the lexicons (mostly names
and tech terms) is spelled letter by letter; `OFLM_TTS_UNKNOWN=skip` on the server drops such words
instead.
