// English grapheme-to-phoneme for Kokoro: misaki 0.9.4's dictionary path in C++, without spaCy (2026-10-16).
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace open_tts {

// One lexicon value: a pronunciation, or misaki's per-tag table (DEFAULT, VBD, None, ...).
struct LexEntry {
    bool table = false;
    std::u32string ps;
    std::vector<std::pair<std::string, std::optional<std::u32string>>> by_tag;
};
using LexMap = std::unordered_map<std::u32string, LexEntry>;

struct G2PResult {
    std::string phonemes;                 // UTF-8, as Kokoro's vocab spells them
    std::vector<std::string> unknown;     // words the lexicon could not read, in order
};

class EnglishG2P {
public:
    // misaki's us_gold.json + us_silver.json (or gb_*). Unknown words are reported, and spelled
    // letter by letter, or dropped as KPipeline drops them when espeak is absent (unk = "").
    EnglishG2P(const std::string& gold_json, const std::string& silver_json, bool british,
               bool spell_unknown = true);
    ~EnglishG2P();
    G2PResult operator()(const std::string& text) const;
    bool british() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// num2words' English, as misaki calls it (exposed for tests).
std::string num2words_cardinal(long long n);
std::string num2words_ordinal(long long n);
std::string num2words_year(long long n);

std::u32string utf8_to_u32(const std::string& s);
std::string u32_to_utf8(const std::u32string& s);

}  // namespace open_tts
