# Plan: flac, aac and opus from /v1/audio/speech

2026-10-17. The speech endpoint answered wav and pcm only and refused OpenAI's other formats. The
bundled FFmpeg (vcpkg's port on Windows; the static build on Linux) carries FFmpeg's own flac, aac
and opus encoders, so those three can be served without a new dependency. mp3 cannot: FFmpeg has no
mp3 encoder of its own (it needs LAME, which this build does not carry), so mp3 stays refused.

## Requirements

- **Modified: SERVER-SPEECH-PARAMS.** `response_format` accepts `flac`, `aac` and `opus` as well as `wav`
  and `pcm`; `mp3` stays a 400 `not_implemented` naming the missing encoder. An `input` with nothing the
  English phonemizer can say (only Chinese characters, say, or only emoji) is a 400 on `input` naming the
  unspoken words, where it was a 200 with an empty file.
- **Modified: SERVER-SPEECH.** The response carries `audio/flac`, `audio/aac` (ADTS) or `audio/ogg` (Opus
  in Ogg, 48 kHz) for those formats.
- **New: SERVER-SPEECH-ENCODE.** The encoder (`src/common/audio/audio_encoder.cpp`) round-trips through
  FFmpeg's decoder: flac to the same 16-bit samples, aac and opus to the same waveform (correlation above
  0.95 after the codec's delay). Verification: test, `src/common/audio/audio_encoder_test.cpp` (CTest
  `audio_encoder`).

## Steps

1. `audio_encode::encode_mono` over FFmpeg's built-in encoders and muxers (flac, adts, ogg), in memory.
2. `speech_request` accepts the three formats; the handler encodes the samples it would have put in a WAV.
3. The unit test, the wire test (`test_speech_api.py`: each format's content type and magic bytes) and
   `oflm-test --tts`.
