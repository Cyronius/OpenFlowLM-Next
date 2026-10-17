// Traces: SERVER-SPEECH-ENCODE (canonical spec: specs/server-api/spec.md)
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "audio/audio_encoder.hpp"
#include "audio/audio_reader.hpp"

static int failures = 0, checks = 0;

static void ok(bool cond, const std::string& what) {
    ++checks;
    if (!cond) ++failures;
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", what.c_str());
}

// Two partials under a slow envelope: speech-like enough that a lossy codec keeps the shape.
static std::vector<float> tone(int rate, double secs) {
    const double pi = 3.14159265358979323846;
    std::vector<float> x((size_t)(rate * secs));
    for (size_t i = 0; i < x.size(); ++i) {
        const double t = (double)i / rate;
        const double env = 0.5 - 0.5 * std::cos(2 * pi * t / secs);
        x[i] = (float)(env * (0.3 * std::sin(2 * pi * 220 * t) + 0.15 * std::sin(2 * pi * 660 * t)));
    }
    return x;
}

static std::vector<float> decode(const std::string& bytes, int rate) {
    AudioReader r;
    audio_data_t a;
    if (!r.load_audio_from_memory(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), a, rate,
                                  MonoDownmixMode::MEAN))
        return {};
    return a.samples;
}

// Best normalized cross-correlation of y against x over lags 0..max_lag (codec delay).
static double best_corr(const std::vector<float>& x, const std::vector<float>& y, int max_lag, int& lag_out) {
    double best = -1;
    for (int lag = 0; lag <= max_lag; ++lag) {
        double xy = 0, xx = 0, yy = 0;
        for (size_t i = 0; i < x.size() && i + lag < y.size(); ++i) {
            xy += (double)x[i] * y[i + lag];
            xx += (double)x[i] * x[i];
            yy += (double)y[i + lag] * y[i + lag];
        }
        const double c = (xx > 0 && yy > 0) ? xy / std::sqrt(xx * yy) : 0;
        if (c > best) {
            best = c;
            lag_out = lag;
        }
    }
    return best;
}

int main() {
    const int rate = 24000;
    const std::vector<float> x = tone(rate, 1.0);

    ok(audio_encode::supported("flac") && audio_encode::supported("aac") && audio_encode::supported("opus"),
       "the bundled FFmpeg encodes flac, aac and opus");
    ok(!audio_encode::supported("mp3"), "mp3 is not offered");

    const std::string flac = audio_encode::encode_mono(x, rate, "flac");
    ok(flac.compare(0, 4, "fLaC") == 0, "flac starts with fLaC (" + std::to_string(flac.size()) + " bytes)");
    const std::vector<float> fy = decode(flac, rate);
    double worst = fy.size() == x.size() ? 0 : 1;
    for (size_t i = 0; i < x.size() && i < fy.size(); ++i) worst = std::max(worst, (double)std::fabs(fy[i] - x[i]));
    ok(fy.size() == x.size() && worst <= 2.0 / 32768,
       "flac decodes to the same samples within 16-bit rounding (" + std::to_string(fy.size()) + " samples, max |d| " +
           std::to_string(worst) + ")");

    const std::string aac = audio_encode::encode_mono(x, rate, "aac");
    ok(aac.size() > 2 && (unsigned char)aac[0] == 0xFF && ((unsigned char)aac[1] & 0xF0) == 0xF0,
       "aac is an ADTS stream (" + std::to_string(aac.size()) + " bytes)");
    const std::vector<float> ay = decode(aac, rate);
    int lag = 0;
    const double ac = best_corr(x, ay, 4096, lag);
    ok(ac > 0.95, "aac decodes to the tone: correlation " + std::to_string(ac) + " at lag " + std::to_string(lag));

    const std::string opus = audio_encode::encode_mono(x, rate, "opus");
    ok(opus.compare(0, 4, "OggS") == 0 && opus.find("OpusHead") != std::string::npos,
       "opus is an Ogg stream with an OpusHead (" + std::to_string(opus.size()) + " bytes)");
    const std::vector<float> oy = decode(opus, rate);
    const double oc = best_corr(x, oy, 4096, lag);
    ok(oc > 0.95, "opus decodes to the tone: correlation " + std::to_string(oc) + " at lag " + std::to_string(lag));
    ok(oy.size() + 2400 >= x.size() && oy.size() <= x.size() + 4800,
       "opus keeps the length within 0.1/0.2 s (" + std::to_string(oy.size()) + " samples)");

    const std::string empty = audio_encode::encode_mono({}, rate, "flac");
    ok(empty.compare(0, 4, "fLaC") == 0, "no samples still give a valid flac header");

    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
