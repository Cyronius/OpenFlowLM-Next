// Mono float PCM to flac, aac or opus through the bundled FFmpeg's built-in encoders (2026-10-17).
#pragma once

#include <string>
#include <vector>

namespace audio_encode {

// "flac" (lossless, the input's rate), "aac" (ADTS, the input's rate) or "opus" (Ogg, 48 kHz).
bool supported(const std::string& format);
const char* content_type(const std::string& format);

// Samples in [-1, 1]; throws std::runtime_error naming the FFmpeg step that failed.
std::string encode_mono(const std::vector<float>& samples, int sample_rate, const std::string& format);

}  // namespace audio_encode
