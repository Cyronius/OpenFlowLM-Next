// Traces: OPEN-TTS-OPS, OPEN-TTS-BLOCKS (specs/open-tts/spec.md). No device, no model weights.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "kokoro.hpp"
#include "ops.hpp"
#include "safetensors.hpp"

using namespace open_tts;

static int failures = 0;

static double rel_l2(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double d = (i < a.size() ? a[i] : 0.0) - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1));
}

static void expect(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    const double e = got.size() == want.size() ? rel_l2(got, want) : 1e9;
    const bool ok = e <= tol;
    if (!ok) ++failures;
    std::printf("%s %-34s rel_l2 %.2e (tol %.0e)%s\n", ok ? "PASS" : "FAIL", what, e, tol,
                got.size() == want.size() ? "" : "  [size mismatch]");
}

static std::vector<double> dbl(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

static std::vector<float> randv(std::mt19937& g, size_t n, float scale = 1.f) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

static void test_gemm(std::mt19937& g) {
    for (auto [M, N, K] : {std::tuple<int, int, int>{7, 33, 300}, {97, 130, 513}, {1, 17, 5}, {200, 1, 64}}) {
        const auto A = randv(g, (size_t)M * K), B = randv(g, (size_t)K * N);
        std::vector<float> C((size_t)M * N, 0.f);
        sgemm(M, N, K, A.data(), K, B.data(), N, C.data(), N, false);
        std::vector<double> R((size_t)M * N, 0.0);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                for (int j = 0; j < N; ++j) R[(size_t)i * N + j] += (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
        char name[64];
        std::snprintf(name, sizeof name, "sgemm %dx%dx%d", M, N, K);
        expect(name, C, R, 1e-5);
    }
}

static std::vector<double> naive_conv(const std::vector<float>& x, int cin, int t, const std::vector<float>& w,
                                      const std::vector<float>& b, int cout, int k, int stride, int pad, int dil,
                                      int groups) {
    const int to = (t + 2 * pad - dil * (k - 1) - 1) / stride + 1, cpg = cin / groups, opg = cout / groups;
    std::vector<double> y((size_t)cout * to, 0.0);
    for (int co = 0; co < cout; ++co)
        for (int n = 0; n < to; ++n) {
            double s = b.empty() ? 0.0 : b[(size_t)co];
            for (int ci = 0; ci < cpg; ++ci)
                for (int tap = 0; tap < k; ++tap) {
                    const int pos = n * stride - pad + tap * dil;
                    if (pos < 0 || pos >= t) continue;
                    s += (double)w[((size_t)co * cpg + ci) * k + tap] * x[(size_t)((co / opg) * cpg + ci) * t + pos];
                }
            y[(size_t)co * to + n] = s;
        }
    return y;
}

static void test_conv(std::mt19937& g) {
    struct Case { int cin, cout, k, stride, pad, dil, groups, t; const char* name; };
    for (const Case& c : {Case{16, 12, 7, 1, 9, 3, 1, 40, "conv1d k7 dilation 3"},
                          Case{22, 8, 12, 6, 3, 1, 1, 61, "conv1d k12 stride 6"},
                          Case{6, 6, 3, 1, 1, 1, 6, 11, "conv1d depthwise"},
                          Case{514, 20, 3, 1, 1, 1, 1, 9, "conv1d 514 in"}}) {
        Conv1d conv;
        conv.cin = c.cin; conv.cout = c.cout; conv.k = c.k; conv.stride = c.stride; conv.pad = c.pad;
        conv.dil = c.dil; conv.groups = c.groups;
        const auto w = randv(g, (size_t)c.cout * (c.cin / c.groups) * c.k), x = randv(g, (size_t)c.cin * c.t);
        conv.set_weight(w);
        conv.b = randv(g, (size_t)c.cout);
        expect(c.name, conv(x, c.t), naive_conv(x, c.cin, c.t, w, conv.b, c.cout, c.k, c.stride, c.pad, c.dil, c.groups), 1e-5);
    }
}

static void test_conv_transpose(std::mt19937& g) {
    for (auto [cin, cout, k, s, p, op, groups, t] : {std::tuple<int, int, int, int, int, int, int, int>{8, 5, 20, 10, 5, 0, 1, 6},
                                                    {7, 7, 3, 2, 1, 1, 7, 9}, {600, 4, 12, 6, 3, 0, 1, 3}}) {
        ConvTranspose1d ct;
        ct.cin = cin; ct.cout = cout; ct.k = k; ct.stride = s; ct.pad = p; ct.out_pad = op; ct.groups = groups;
        const auto w = randv(g, (size_t)cin * (cout / groups) * k), x = randv(g, (size_t)cin * t);
        ct.set_weight(w);
        ct.b = randv(g, (size_t)cout);
        const int to = ct.out_len(t), opg = cout / groups, cpg = cin / groups;
        std::vector<double> y((size_t)cout * to, 0.0);
        for (int co = 0; co < cout; ++co)
            for (int o = 0; o < to; ++o) y[(size_t)co * to + o] = ct.b[(size_t)co];
        for (int ci = 0; ci < cin; ++ci)
            for (int col = 0; col < opg; ++col)
                for (int i = 0; i < t; ++i)
                    for (int tap = 0; tap < k; ++tap) {
                        const int o = i * s - p + tap, co = (ci / cpg) * opg + col;
                        if (o >= 0 && o < to)
                            y[(size_t)co * to + o] += (double)x[(size_t)ci * t + i] * w[((size_t)ci * opg + col) * k + tap];
                    }
        char name[64];
        std::snprintf(name, sizeof name, "conv_transpose1d k%d s%d g%d", k, s, groups);
        expect(name, ct(x, t), y, 1e-5);
    }
}

static void test_lstm(std::mt19937& g) {
    const int in = 10, H = 6, T = 13;
    BiLSTM l;
    l.in = in;
    l.hidden = H;
    std::vector<float> wih[2], whh[2], b[2];
    for (int d = 0; d < 2; ++d) {
        wih[d] = randv(g, (size_t)4 * H * in, 0.4f);
        whh[d] = randv(g, (size_t)4 * H * H, 0.4f);
        b[d] = randv(g, (size_t)4 * H, 0.4f);
        std::vector<float> t(wih[d].size());
        for (int r = 0; r < 4 * H; ++r)
            for (int i = 0; i < in; ++i) t[(size_t)i * 4 * H + r] = wih[d][(size_t)r * in + i];
        l.wih_t[d].set(t, weight_dtype());
        l.whh[d].set(whh[d], weight_dtype());
        l.bias[d] = b[d];
    }
    const auto x = randv(g, (size_t)T * in);
    std::vector<double> y((size_t)T * 2 * H);
    for (int d = 0; d < 2; ++d) {
        std::vector<double> h((size_t)H, 0), c((size_t)H, 0);
        for (int s = 0; s < T; ++s) {
            const int t = d ? T - 1 - s : s;
            std::vector<double> z((size_t)4 * H);
            for (int r = 0; r < 4 * H; ++r) {
                double v = b[d][(size_t)r];
                for (int i = 0; i < in; ++i) v += (double)wih[d][(size_t)r * in + i] * x[(size_t)t * in + i];
                for (int j = 0; j < H; ++j) v += (double)whh[d][(size_t)r * H + j] * h[(size_t)j];
                z[(size_t)r] = v;
            }
            for (int j = 0; j < H; ++j) {
                auto sg = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
                c[(size_t)j] = sg(z[(size_t)(H + j)]) * c[(size_t)j] + sg(z[(size_t)j]) * std::tanh(z[(size_t)(2 * H + j)]);
                h[(size_t)j] = sg(z[(size_t)(3 * H + j)]) * std::tanh(c[(size_t)j]);
                y[(size_t)t * 2 * H + (size_t)d * H + j] = h[(size_t)j];
            }
        }
    }
    expect("bilstm vs float64 recurrence", l(x, T), y, weight_dtype() == DType::F16 ? 2e-3 : 1e-5);
}

// 16-bit weights round each weight to 11 significant bits; products stay within a few 1e-4.
static void test_f16_weights(std::mt19937& g) {
    for (float v : {0.f, 1.f, -2.5f, 65504.f, 6.1035156e-05f, 5.9604645e-08f, 0.33333334f})
        if (f16_to_f32(f32_to_f16(v)) != (v == 0.33333334f ? 0.33325195f : v)) {
            ++failures;
            std::printf("FAIL f16 round trip of %g gives %g\n", v, f16_to_f32(f32_to_f16(v)));
        }
    set_weight_dtype(DType::F16);
    Conv1d conv;
    conv.cin = 64; conv.cout = 24; conv.k = 3; conv.pad = 1;
    const auto w = randv(g, (size_t)24 * 64 * 3), x = randv(g, (size_t)64 * 50);
    conv.set_weight(w);
    std::vector<float> wr(w.size());
    for (size_t i = 0; i < w.size(); ++i) wr[i] = f16_to_f32(f32_to_f16(w[i]));
    expect("conv1d, f16 weights vs f16-rounded float64", conv(x, 50), naive_conv(x, 64, 50, wr, {}, 24, 3, 1, 1, 1, 1), 1e-5);
    expect("conv1d, f16 weights vs f32 float64", conv(x, 50), naive_conv(x, 64, 50, w, {}, 24, 3, 1, 1, 1, 1), 1e-3);
    Linear lin;
    lin.in = 37; lin.out = 21;
    const auto lw = randv(g, (size_t)37 * 21), lx = randv(g, (size_t)5 * 37);
    lin.set_weight(lw);
    std::vector<double> ly((size_t)5 * 21, 0.0);
    for (int r = 0; r < 5; ++r)
        for (int o = 0; o < 21; ++o)
            for (int i = 0; i < 37; ++i) ly[(size_t)r * 21 + o] += (double)lx[(size_t)r * 37 + i] * f16_to_f32(f32_to_f16(lw[(size_t)o * 37 + i]));
    expect("linear, f16 weights vs f16-rounded float64", lin(lx, 5), ly, 1e-5);
    test_lstm(g);
    set_weight_dtype(DType::F32);
}

static void test_stft(std::mt19937& g) {
    const int len = 600;
    const auto x = randv(g, (size_t)len, 0.3f);
    std::vector<float> mag, ph;
    int frames = 0;
    stft_mag_phase(x.data(), len, 20, 5, mag, ph, frames);
    const std::vector<float> back = istft(mag.data(), ph.data(), frames, 20, 5);
    std::vector<double> want(x.begin(), x.end());
    expect("istft(stft(x)) == x", back, want, 1e-5);
}

static void test_fixtures(const std::string& path, double scale) {
    if (!std::filesystem::is_regular_file(path)) {
        std::printf("FAIL fixtures missing: %s (run testdata/gen_fixtures.py)\n", path.c_str());
        ++failures;
        return;
    }
    SafetensorsFile f(path);
    const WeightReader w(f);
    const std::vector<float> s = f.f32("style");
    const std::string tag = scale > 1 ? "[f16 weights] " : "";
    auto check = [&](const std::string& what, const std::vector<float>& got, const char* ref, double tol) {
        expect((tag + what).c_str(), got, dbl(f.f32(ref)), tol * scale);
    };
    int t_out;
    {
        AdainResBlk1d b = blocks::resblk(w, "rb_up", 6, 4, 5, true);
        check("AdainResBlk1d, upsampling", b.forward(f.f32("rb_up.in"), 7, s.data(), t_out), "rb_up.out", 2e-5);
        AdainResBlk1d c = blocks::resblk(w, "rb", 4, 4, 5, false);
        check("AdainResBlk1d", c.forward(f.f32("rb.in"), 9, s.data(), t_out), "rb.out", 2e-5);
    }
    {
        AdaINResBlock1 r = blocks::resblock1(w, "rb1", 4, 7, {1, 3, 5}, 5);
        std::vector<float> x = f.f32("rb1.in");
        r.forward(x, 30, s.data());
        check("AdaINResBlock1 (Snake, dilations)", x, "rb1.out", 2e-5);
    }
    check("LSTM, bidirectional", blocks::lstm(w, "lstm", 6, 3)(f.f32("lstm.in"), 9), "lstm.out", 1e-5);
    check("ConvTranspose1d, weight norm", blocks::upsampler(w, "ups", 4, 3, 20, 10)(f.f32("ups.in"), 5), "ups.out", 1e-5);
    check("Conv1d k12 stride 6 (noise_convs)",
          blocks::conv(w, "nconv", 22, 4, 12, 3, 1, false, true, 6)(f.f32("nconv.in"), 61), "nconv.out", 1e-5);
    if (scale > 1) return;   // the rest carries no weight matrix
    {
        std::vector<float> x = f.f32("ln.in");
        layer_norm_channels(x.data(), 5, 7, w.get("ln.gamma", 5).data(), w.get("ln.beta", 5).data(), 1e-5f);
        expect("LayerNorm over channels", x, dbl(f.f32("ln.out")), 1e-5);
    }
    {
        std::vector<float> x = f.f32("aln.in");
        const Linear fc = blocks::linear(w, "aln.fc", 5, 12);
        std::vector<float> gb(12);
        fc.forward(s.data(), 1, gb.data());
        layer_norm_rows(x.data(), 7, 6, nullptr, nullptr, 1e-5f);
        for (int t = 0; t < 7; ++t)
            for (int j = 0; j < 6; ++j) x[(size_t)t * 6 + j] = (1.f + gb[(size_t)j]) * x[(size_t)t * 6 + j] + gb[(size_t)(6 + j)];
        expect("AdaLayerNorm", x, dbl(f.f32("aln.out")), 1e-5);
    }
    {
        std::vector<float> mag, ph;
        int frames = 0;
        const std::vector<float> x = f.f32("stft.in");
        stft_mag_phase(x.data(), (int)x.size(), 20, 5, mag, ph, frames);
        const std::vector<float> rm = f.f32("stft.mag"), rp = f.f32("stft.phase");
        std::vector<float> got(2 * mag.size());
        std::vector<double> want(2 * mag.size());
        for (size_t i = 0; i < mag.size() && i < rm.size(); ++i) {
            got[2 * i] = mag[i] * std::cos(ph[i]);
            got[2 * i + 1] = mag[i] * std::sin(ph[i]);
            want[2 * i] = rm[i] * std::cos((double)rp[i]);
            want[2 * i + 1] = rm[i] * std::sin((double)rp[i]);
        }
        expect("torch.stft (as complex)", got, want, 1e-5);
        expect("torch.stft magnitude", mag, dbl(rm), 1e-5);
        const std::vector<float> m2 = f.f32("istft.mag"), p2 = f.f32("istft.phase");
        expect("torch.istft", istft(m2.data(), p2.data(), 121, 20, 5), dbl(f.f32("istft.out")), 1e-5);
    }
    {
        const std::vector<float> lw = w.get("src.l_linear.weight", 9), lb = w.get("src.l_linear.bias", 1);
        const std::vector<float> f0 = f.f32("src.f0");
        expect("harmonic source, no noise", harmonic_source(f0, 300, lw.data(), lb[0], 0), dbl(f.f32("src.out0")), 1e-4);
        expect("harmonic source, seeded noise", harmonic_source(f0, 300, lw.data(), lb[0], 7), dbl(f.f32("src.out7")), 1e-4);
        Gaussian gs(7);
        std::vector<float> gv(11);
        for (auto& v : gv) v = gs.next();
        expect("noise stream == ref_kokoro.gaussian", gv, dbl(f.f32("gauss7")), 1e-7);
    }
    {
        std::vector<float> v = f.f32("gelu.in");
        gelu_tanh(v.data(), (int64_t)v.size());
        expect("gelu (tanh form)", v, dbl(f.f32("gelu.out")), 1e-6);
    }
}

int main(int argc, char** argv) {
    std::mt19937 g(1234);
    set_weight_dtype(DType::F32);
    test_gemm(g);
    test_conv(g);
    test_conv_transpose(g);
    test_lstm(g);
    test_stft(g);
    test_f16_weights(g);
    const std::string fx = (std::filesystem::path(argc > 1 ? argv[1] : "testdata") / "blocks.safetensors").string();
    test_fixtures(fx, 1.0);
    // The engine's default: the same blocks with 16-bit weights, against PyTorch's f32.
    set_weight_dtype(DType::F16);
    test_fixtures(fx, 100.0);
    set_weight_dtype(DType::F32);
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
