// kokoro_cli: speak, phonemize, and check the open Kokoro engine against a PyTorch dump (2026-10-16).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "safetensors.hpp"
#include "tts_engine.hpp"

using namespace open_tts;

namespace {

struct Stats {
    double max_abs = 0, max_ref = 0, rel_l2 = 0, cos = 0;
};

Stats compare(const std::vector<float>& a, const std::vector<float>& b) {
    Stats s;
    double num = 0, den = 0, ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double d = (double)a[i] - b[i];
        s.max_abs = std::max(s.max_abs, std::fabs(d));
        s.max_ref = std::max(s.max_ref, std::fabs((double)b[i]));
        num += d * d;
        den += (double)b[i] * b[i];
        ab += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    s.rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    s.cos = (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : 0;
    return s;
}

void report(const char* name, const std::vector<float>& mine, const std::vector<float>& ref) {
    const Stats s = compare(mine, ref);
    std::printf("  %-18s n=%-9zu %s rel_l2 %.3e  max|d| %.3e  (max|ref| %.3g)  cos %.9f\n", name, ref.size(),
                mine.size() == ref.size() ? "  " : "!!", s.rel_l2, s.max_abs, s.max_ref, s.cos);
}

int check(const std::string& model_dir, const std::string& ref_path) {
    Kokoro model(model_dir);
    SafetensorsFile ref(ref_path);
    std::vector<int> ids;
    for (float v : ref.f32("ids")) ids.push_back((int)v);
    const std::vector<float> style = ref.f32("style");
    const uint64_t seed = (uint64_t)ref.f64_scalar("seed");
    const float speed = ref.f32("speed").at(0);
    std::printf("weights %s (%s); %zu ids, seed %llu, speed %g\n", model.weights_path().c_str(),
                model.weights_dtype().c_str(), ids.size(), (unsigned long long)seed, speed);

    Trace tr;
    std::vector<int> dur;
    auto t0 = std::chrono::steady_clock::now();
    model.synthesize(ids, style.data(), speed, seed, &tr, &dur);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<int> rdur;
    for (float v : ref.f32("pred.dur")) rdur.push_back((int)v);
    int differ = 0;
    for (size_t i = 0; i < dur.size() && i < rdur.size(); ++i) differ += dur[i] != rdur[i];
    std::printf("free run (%.0f ms): durations differ at %d of %zu tokens\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(), differ, dur.size());
    for (const char* n : {"bert", "d_en", "pred.d", "pred.dur_float", "pred.dur", "pred.F0", "pred.N", "t_en", "dec.x",
                          "gen.har_source", "gen.har", "gen.stage0", "gen.stage1", "gen.conv_post", "audio"})
        if (ref.has(n) && tr.t.count(n)) report(n, tr.t[n], ref.f32(n));
    if (differ) {
        Trace tf;
        model.synthesize(ids, style.data(), speed, seed, &tf, nullptr, &rdur);
        std::printf("with the reference's durations:\n");
        for (const char* n : {"pred.F0", "pred.N", "dec.x", "gen.har_source", "gen.conv_post", "audio"})
            if (ref.has(n) && tf.t.count(n)) report(n, tf.t[n], ref.f32(n));
    }
    // The generator alone, fed the reference's decoder output, F0 and harmonic source.
    const std::vector<float> rx = ref.f32("dec.x"), rf0 = ref.f32("pred.F0"), rhar = ref.f32("gen.har_source");
    Trace tg;
    const std::vector<float> ga = model.generator(rx, (int)(rx.size() / 512), style.data(), rf0, seed, &tg, &rhar);
    std::printf("generator alone, on the reference's inputs:\n");
    for (const char* n : {"gen.har", "gen.stage0", "gen.stage1", "gen.conv_post"})
        if (ref.has(n) && tg.t.count(n)) report(n, tg.t[n], ref.f32(n));
    report("audio", ga, ref.f32("audio"));
    std::printf("harmonic source alone, on the reference's F0:\n");
    report("gen.har_source", model.harmonic_source(rf0, seed), rhar);
    return 0;
}

std::string read_file(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void usage() {
    std::fprintf(stderr,
                 "kokoro_cli --model DIR [--voice af_heart] [--speed 1]\n"
                 "    --text TEXT | --text-file F | --phonemes PS   [--out x.wav] [--bench N]\n"
                 "    --g2p TEXT | --g2p-lines FILE                  print phonemes (and unknown words)\n"
                 "    --check REF.safetensors                        block-by-block against ref_kokoro.py\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    if (!a.count("--model")) {
        usage();
        return 2;
    }
    try {
        const std::string dir = a["--model"];
        if (a.count("--check")) return check(dir, a["--check"]);
        TtsEngine tts(dir);
        const std::string voice = a.count("--voice") ? a["--voice"] : "af_heart";
        const bool british = !voice.empty() && voice[0] == 'b';
        if (a.count("--g2p")) {
            G2PResult r = tts.phonemize(a["--g2p"], british);
            std::cout << r.phonemes << "\n";
            for (auto& u : r.unknown) std::cout << "unknown: " << u << "\n";
            return 0;
        }
        if (a.count("--g2p-lines")) {
            std::ifstream in(a["--g2p-lines"], std::ios::binary);
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                G2PResult r = tts.phonemize(line, british);
                std::string unk;
                for (auto& u : r.unknown) unk += (unk.empty() ? "" : "|") + u;
                std::cout << r.phonemes << "\t" << unk << "\n";
            }
            return 0;
        }
        const float speed = a.count("--speed") ? std::stof(a["--speed"]) : 1.f;
        const int reps = a.count("--bench") ? std::stoi(a["--bench"]) : 1;
        std::vector<float> audio;
        SpeakInfo info;
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            info = SpeakInfo{};
            auto t0 = std::chrono::steady_clock::now();
            if (a.count("--phonemes")) audio = tts.speak_phonemes(a["--phonemes"], voice, speed);
            else if (a.count("--text-file")) audio = tts.speak(read_file(a["--text-file"]), voice, speed, &info);
            else audio = tts.speak(a.count("--text") ? a["--text"] : "Hello world.", voice, speed, &info);
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        const double secs = (double)audio.size() / kSampleRate;
        for (auto& c : info.chunks) std::cout << "phonemes: " << c << "\n";
        for (auto& u : info.unknown) std::cout << "unknown: " << u << "\n";
        std::printf("audio %.2f s, synthesis %.3f s (best of %d), real-time factor %.3f\n", secs, best, reps,
                    secs > 0 ? best / secs : 0.0);
        if (a.count("--out")) {
            std::ofstream out(a["--out"], std::ios::binary);
            const std::string w = encode_wav(audio);
            out.write(w.data(), (std::streamsize)w.size());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
