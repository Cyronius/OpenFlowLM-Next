// Text to 24 kHz speech with Kokoro-82M: phonemizer, chunking, voices, WAV/PCM encoding (2026-10-16).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "g2p_en.hpp"
#include "kokoro.hpp"

namespace open_tts {

constexpr int kSampleRate = 24000;
// Fixed so a request gives the same audio every time; 0 would turn the source noise off.
constexpr uint64_t kDefaultNoiseSeed = 0x6B6F6B6F726F3031ull;

struct SpeakInfo {
    std::vector<std::string> chunks;     // phoneme strings, one model call each
    std::vector<std::string> unknown;    // words the lexicon could not read
    double seconds = 0.0;
};

class TtsEngine {
public:
    // model_dir: config.json, kokoro-v1_0.safetensors, voices/*.safetensors, lexicon/{us,gb}_{gold,silver}.json
    explicit TtsEngine(const std::string& model_dir);
    ~TtsEngine();

    std::vector<std::string> voices() const;
    // A voice name the request may use: an installed voice, or a unique "<lang><g>_<name>" match
    // for a bare name ("alloy" -> "af_alloy"). Empty when there is none.
    std::string resolve_voice(const std::string& name) const;
    // Only American (a*) and British (b*) English voices; the phonemizer is English only.
    static bool english_voice(const std::string& voice);

    std::vector<float> speak(const std::string& text, const std::string& voice, float speed,
                             SpeakInfo* info = nullptr);
    std::vector<float> speak_phonemes(const std::string& phonemes, const std::string& voice, float speed);
    G2PResult phonemize(const std::string& text, bool british);

    const Kokoro& model() const { return *model_; }
    void set_noise_seed(uint64_t seed) { noise_seed_ = seed; }

private:
    const Voice& voice(const std::string& name);
    std::vector<float> run_chunk(const std::string& ps, const Voice& v, float speed);

    std::string dir_;
    std::unique_ptr<Kokoro> model_;
    std::map<std::string, std::string> voice_files_;
    std::map<std::string, Voice> voice_cache_;
    std::unique_ptr<EnglishG2P> g2p_[2];
    std::mutex mu_;
    uint64_t noise_seed_ = kDefaultNoiseSeed;
};

// Kokoro takes at most 510 phonemes per call; longer text splits where KPipeline's
// waterfall would: after . ! ? ..., else : ;, else , and dashes, else at a space.
std::vector<std::u32string> chunk_phonemes(const std::u32string& ps, size_t limit = 510);

std::string encode_wav(const std::vector<float>& samples, int rate = kSampleRate);
std::string encode_pcm16(const std::vector<float>& samples);

}  // namespace open_tts
