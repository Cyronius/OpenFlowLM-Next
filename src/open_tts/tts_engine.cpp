// Text to speech with Kokoro-82M on the host (2026-10-16).
#include "tts_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace open_tts {
namespace fs = std::filesystem;

TtsEngine::TtsEngine(const std::string& dir) : dir_(dir) {
    model_ = std::make_unique<Kokoro>(dir);
    const fs::path vd = fs::path(dir) / "voices";
    if (fs::is_directory(vd))
        for (const auto& e : fs::directory_iterator(vd))
            if (e.is_regular_file() && e.path().extension() == ".safetensors")
                voice_files_[e.path().stem().string()] = e.path().string();
    if (voice_files_.empty()) throw std::runtime_error(dir + ": no voices/*.safetensors");
}

TtsEngine::~TtsEngine() = default;

std::vector<std::string> TtsEngine::voices() const {
    std::vector<std::string> v;
    for (const auto& [name, path] : voice_files_) v.push_back(name);
    return v;
}

bool TtsEngine::english_voice(const std::string& v) { return !v.empty() && (v[0] == 'a' || v[0] == 'b'); }

std::string TtsEngine::resolve_voice(const std::string& name) const {
    if (voice_files_.count(name)) return name;
    std::string found;
    for (const auto& [v, path] : voice_files_) {
        const size_t u = v.find('_');
        if (u != std::string::npos && v.substr(u + 1) == name) {
            if (!found.empty()) return "";
            found = v;
        }
    }
    return found;
}

const Voice& TtsEngine::voice(const std::string& name) {
    auto it = voice_cache_.find(name);
    if (it != voice_cache_.end()) return it->second;
    auto f = voice_files_.find(name);
    if (f == voice_files_.end()) throw std::runtime_error("no voice " + name);
    return voice_cache_.emplace(name, Voice::load(f->second)).first->second;
}

G2PResult TtsEngine::phonemize(const std::string& text, bool british) {
    std::unique_ptr<EnglishG2P>& g = g2p_[british ? 1 : 0];
    if (!g) {
        const fs::path lex = fs::path(dir_) / "lexicon";
        const std::string p = british ? "gb" : "us";
        const char* mode = std::getenv("OFLM_TTS_UNKNOWN");
        const bool spell = !(mode && std::string(mode) == "skip");
        g = std::make_unique<EnglishG2P>((lex / (p + "_gold.json")).string(), (lex / (p + "_silver.json")).string(),
                                         british, spell);
    }
    return (*g)(text);
}

std::vector<float> TtsEngine::run_chunk(const std::string& ps, const Voice& v, float speed) {
    const std::u32string u = utf8_to_u32(ps);
    std::vector<int> ids{0};
    for (char32_t c : u) {
        auto it = model_->config().vocab.find(u32_to_utf8(std::u32string(1, c)));
        if (it != model_->config().vocab.end()) ids.push_back(it->second);
    }
    ids.push_back(0);
    if (ids.size() <= 2) return {};
    return model_->synthesize(ids, v.row_for((int)u.size()), speed, kDefaultNoiseSeed);
}

std::vector<float> TtsEngine::speak_phonemes(const std::string& ps, const std::string& name, float speed) {
    std::lock_guard<std::mutex> lock(mu_);
    const Voice& v = voice(name);
    std::vector<float> out;
    for (const std::u32string& c : chunk_phonemes(utf8_to_u32(ps))) {
        std::vector<float> a = run_chunk(u32_to_utf8(c), v, speed);
        out.insert(out.end(), a.begin(), a.end());
    }
    return out;
}

std::vector<float> TtsEngine::speak(const std::string& text, const std::string& name, float speed, SpeakInfo* info) {
    std::lock_guard<std::mutex> lock(mu_);
    const Voice& v = voice(name);
    const bool british = !name.empty() && name[0] == 'b';
    std::vector<float> out;
    // KPipeline's split_pattern r'\n+': each paragraph is phonemized and spoken on its own.
    size_t i = 0;
    while (i < text.size()) {
        size_t j = text.find('\n', i);
        if (j == std::string::npos) j = text.size();
        const std::string para = text.substr(i, j - i);
        i = j + 1;
        if (para.find_first_not_of(" \t\r") == std::string::npos) continue;
        G2PResult g = phonemize(para, british);
        if (info) info->unknown.insert(info->unknown.end(), g.unknown.begin(), g.unknown.end());
        for (const std::u32string& c : chunk_phonemes(utf8_to_u32(g.phonemes))) {
            const std::string ps = u32_to_utf8(c);
            if (info) info->chunks.push_back(ps);
            std::vector<float> a = run_chunk(ps, v, speed);
            out.insert(out.end(), a.begin(), a.end());
        }
    }
    if (info) info->seconds = (double)out.size() / kSampleRate;
    return out;
}

std::vector<std::u32string> chunk_phonemes(const std::u32string& ps_in, size_t limit) {
    std::u32string ps = ps_in;
    while (!ps.empty() && ps.front() == ' ') ps.erase(0, 1);
    while (!ps.empty() && ps.back() == ' ') ps.pop_back();
    std::vector<std::u32string> out;
    static const std::u32string levels[3] = {U"!.?…", U":;", U",—"};
    static const std::u32string bumps = U")”";
    while (ps.size() > limit) {
        size_t cut = std::u32string::npos;
        for (const std::u32string& lv : levels) {
            for (size_t k = std::min(limit, ps.size()) ; k-- > 0;) {
                if (lv.find(ps[k]) == std::u32string::npos) continue;
                size_t z = k + 1;
                if (z < ps.size() && bumps.find(ps[z]) != std::u32string::npos) ++z;
                if (z <= limit) { cut = z; break; }
            }
            if (cut != std::u32string::npos) break;
        }
        if (cut == std::u32string::npos) {
            const size_t sp = ps.rfind(U' ', limit);
            cut = (sp == std::u32string::npos || sp == 0) ? limit : sp;
        }
        std::u32string head = ps.substr(0, cut);
        while (!head.empty() && head.back() == ' ') head.pop_back();
        if (!head.empty()) out.push_back(head);
        ps = ps.substr(cut);
        while (!ps.empty() && ps.front() == ' ') ps.erase(0, 1);
    }
    if (!ps.empty()) out.push_back(ps);
    return out;
}

std::string encode_pcm16(const std::vector<float>& s) {
    std::string out(s.size() * 2, '\0');
    for (size_t i = 0; i < s.size(); ++i) {
        const float x = std::min(1.f, std::max(-1.f, s[i]));
        const int16_t v = (int16_t)std::lrint(x * 32767.f);
        out[2 * i] = (char)(v & 0xFF);
        out[2 * i + 1] = (char)((v >> 8) & 0xFF);
    }
    return out;
}

std::string encode_wav(const std::vector<float>& s, int rate) {
    const std::string pcm = encode_pcm16(s);
    std::string h;
    auto u32le = [&](uint32_t v) { for (int k = 0; k < 4; ++k) h.push_back((char)((v >> (8 * k)) & 0xFF)); };
    auto u16le = [&](uint16_t v) { h.push_back((char)(v & 0xFF)); h.push_back((char)(v >> 8)); };
    h += "RIFF";
    u32le((uint32_t)(36 + pcm.size()));
    h += "WAVEfmt ";
    u32le(16);
    u16le(1);
    u16le(1);
    u32le((uint32_t)rate);
    u32le((uint32_t)rate * 2);
    u16le(2);
    u16le(16);
    h += "data";
    u32le((uint32_t)pcm.size());
    return h + pcm;
}

}  // namespace open_tts
