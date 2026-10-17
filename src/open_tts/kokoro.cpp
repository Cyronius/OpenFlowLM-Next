// Kokoro-82M on the host: weight loading (weight norm folded at load) and the forward pass (2026-10-16).
#include "kokoro.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "safetensors.hpp"

namespace open_tts {
namespace fs = std::filesystem;

std::string WeightReader::find(const std::string& name) const {
    for (const std::string& c : candidates(name))
        if (f_.has(c)) return c;
    return "";
}

std::vector<float> WeightReader::get(const std::string& name, int64_t numel) const {
    const std::string n = find(name);
    if (n.empty()) throw std::runtime_error(f_.path() + ": missing tensor " + name);
    std::vector<float> v = f_.f32(n);
    if (numel >= 0 && (int64_t)v.size() != numel)
        throw std::runtime_error(f_.path() + ": tensor " + n + " has " + std::to_string(v.size()) +
                                 " values, expected " + std::to_string(numel));
    return v;
}

std::vector<float> WeightReader::conv3(const std::string& name, int d0, int d1, int d2) const {
    const std::string n = find(name);
    if (n.empty()) throw std::runtime_error(f_.path() + ": missing tensor " + name);
    const TensorInfo& ti = f_.info(n);
    std::vector<float> v = f_.f32(n);
    if ((int64_t)v.size() != (int64_t)d0 * d1 * d2)
        throw std::runtime_error(f_.path() + ": tensor " + n + " has the wrong size");
    // MLX stores conv weights channels-last, [out][k][in]
    const std::vector<int64_t> mlx{d0, d2, d1};
    if (d1 != d2 && ti.shape == mlx) {
        std::vector<float> t(v.size());
        for (int a = 0; a < d0; ++a)
            for (int c = 0; c < d2; ++c)
                for (int b = 0; b < d1; ++b) t[((size_t)a * d1 + b) * d2 + c] = v[((size_t)a * d2 + c) * d1 + b];
        return t;
    }
    return v;
}

std::vector<float> WeightReader::wn3(const std::string& prefix, int d0, int d1, int d2) const {
    for (const auto& [g, v] : {std::pair<std::string, std::string>{".weight_g", ".weight_v"},
                               {".parametrizations.weight.original0", ".parametrizations.weight.original1"}}) {
        if (has(prefix + g) && has(prefix + v))
            return fold_weight_norm(get(prefix + g, d0), conv3(prefix + v, d0, d1, d2), d0);
    }
    return conv3(prefix + ".weight", d0, d1, d2);
}

std::vector<std::string> WeightReader::candidates(const std::string& name) const {
    std::vector<std::string> out{name};
    const size_t dot = name.find('.');
    if (dot != std::string::npos) out.push_back(name.substr(0, dot) + ".module" + name.substr(dot));
    // mlx-audio's renames: LSTM directions and the text encoder's LayerNorm gamma/beta
    static const std::pair<const char*, const char*> rules[] = {
        {"weight_ih_l0_reverse", "backward_lstm.Wx"}, {"weight_hh_l0_reverse", "backward_lstm.Wh"},
        {"bias_ih_l0_reverse", "backward_lstm.bias_ih"}, {"bias_hh_l0_reverse", "backward_lstm.bias_hh"},
        {"weight_ih_l0", "forward_lstm.Wx"}, {"weight_hh_l0", "forward_lstm.Wh"},
        {"bias_ih_l0", "forward_lstm.bias_ih"}, {"bias_hh_l0", "forward_lstm.bias_hh"},
        {"gamma", "weight"}, {"beta", "bias"}};
    const size_t last = name.rfind('.');
    const std::string head = last == std::string::npos ? "" : name.substr(0, last + 1);
    const std::string tail = last == std::string::npos ? name : name.substr(last + 1);
    for (const auto& [from, to] : rules)
        if (tail == from) out.push_back(head + to);
    return out;
}

namespace blocks {

Linear linear(const WeightReader& w, const std::string& p, int in, int out, bool bias) {
    Linear l;
    l.in = in;
    l.out = out;
    l.set_weight(w.get(p + ".weight", (int64_t)in * out));
    if (bias) l.b = w.get(p + ".bias", out);
    return l;
}

Conv1d conv(const WeightReader& w, const std::string& p, int cin, int cout, int k, int pad, int dil, bool norm,
            bool bias, int stride) {
    Conv1d c;
    c.cin = cin;
    c.cout = cout;
    c.k = k;
    c.pad = pad;
    c.dil = dil;
    c.stride = stride;
    c.set_weight(norm ? w.wn3(p, cout, cin, k) : w.conv3(p + ".weight", cout, cin, k));
    if (bias) c.b = w.get(p + ".bias", cout);
    return c;
}

BiLSTM lstm(const WeightReader& w, const std::string& p, int in, int hidden) {
    BiLSTM l;
    l.in = in;
    l.hidden = hidden;
    const int g = 4 * hidden;
    for (int d = 0; d < 2; ++d) {
        const std::string sfx = d ? "_l0_reverse" : "_l0";
        const std::vector<float> wih = w.get(p + ".weight_ih" + sfx, (int64_t)g * in);
        std::vector<float> t((size_t)in * g);
        for (int r = 0; r < g; ++r)
            for (int i = 0; i < in; ++i) t[(size_t)i * g + r] = wih[(size_t)r * in + i];
        l.wih_t[d].set(t, weight_dtype());
        l.whh[d].set(w.get(p + ".weight_hh" + sfx, (int64_t)g * hidden), weight_dtype());
        const std::vector<float> bi = w.get(p + ".bias_ih" + sfx, g), bh = w.get(p + ".bias_hh" + sfx, g);
        l.bias[d].resize((size_t)g);
        for (int r = 0; r < g; ++r) l.bias[d][(size_t)r] = bi[(size_t)r] + bh[(size_t)r];
    }
    return l;
}

AdaIN1d adain(const WeightReader& w, const std::string& p, int style, int c) {
    AdaIN1d a;
    a.c = c;
    a.fc = linear(w, p + ".fc", style, 2 * c);
    if (w.has(p + ".norm.weight")) {
        a.norm_w = w.get(p + ".norm.weight", c);
        a.norm_b = w.get(p + ".norm.bias", c);
    }
    return a;
}

AdainResBlk1d resblk(const WeightReader& w, const std::string& p, int din, int dout, int style, bool up) {
    AdainResBlk1d b;
    b.din = din;
    b.dout = dout;
    b.upsample = up;
    b.conv1 = conv(w, p + ".conv1", din, dout, 3, 1);
    b.conv2 = conv(w, p + ".conv2", dout, dout, 3, 1);
    b.norm1 = adain(w, p + ".norm1", style, din);
    b.norm2 = adain(w, p + ".norm2", style, dout);
    if (din != dout) b.conv1x1 = conv(w, p + ".conv1x1", din, dout, 1, 0, 1, true, false);
    if (up) {
        ConvTranspose1d& t = b.pool;
        t.cin = din;
        t.cout = din;
        t.k = 3;
        t.stride = 2;
        t.pad = 1;
        t.out_pad = 1;
        t.groups = din;
        t.set_weight(w.wn3(p + ".pool", din, 1, 3));
        t.b = w.get(p + ".pool.bias", din);
    }
    return b;
}

AdaINResBlock1 resblock1(const WeightReader& w, const std::string& p, int ch, int k, const std::vector<int>& dil,
                         int style) {
    AdaINResBlock1 r;
    r.ch = ch;
    for (int i = 0; i < 3; ++i) {
        const std::string s = std::to_string(i);
        const int d = dil[(size_t)i];
        r.convs1[i] = conv(w, p + ".convs1." + s, ch, ch, k, (k * d - d) / 2, d);
        r.convs2[i] = conv(w, p + ".convs2." + s, ch, ch, k, (k - 1) / 2, 1);
        r.adain1[i] = adain(w, p + ".adain1." + s, style, ch);
        r.adain2[i] = adain(w, p + ".adain2." + s, style, ch);
        r.alpha1[i] = w.get(p + ".alpha1." + s, ch);
        r.alpha2[i] = w.get(p + ".alpha2." + s, ch);
    }
    return r;
}

ConvTranspose1d upsampler(const WeightReader& w, const std::string& p, int cin, int cout, int k, int stride) {
    ConvTranspose1d t;
    t.cin = cin;
    t.cout = cout;
    t.k = k;
    t.stride = stride;
    t.pad = (k - stride) / 2;
    t.set_weight(w.wn3(p, cin, cout, k));
    t.b = w.get(p + ".bias", cout);
    return t;
}

}  // namespace blocks

namespace {

std::vector<float> transpose(const std::vector<float>& x, int rows, int cols) {
    std::vector<float> y(x.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) y[(size_t)c * rows + r] = x[(size_t)r * cols + c];
    return y;
}

}  // namespace

std::vector<float> harmonic_source(const std::vector<float>& f0, int up, const float* w9, float bias,
                                   uint64_t noise_seed) {
    // SineGen + SourceModuleHnNSF in torch's float32 order. The random initial phase torch adds
    // lands on sample 0 only, which the 1/up linear downsample never reads, so it has no effect.
    const int n = (int)f0.size(), L = n * up, NH = 9;
    std::vector<float> phd((size_t)n * NH);
    for (int h = 0; h < NH; ++h) {
        double cum = 0.0;
        for (int i = 0; i < n; ++i) {
            float r = (f0[(size_t)i] * (float)(h + 1)) / 24000.f;
            r = r - std::floor(r);
            cum += r;
            const float cs = (float)cum;
            phd[(size_t)i * NH + h] = ((cs * 2.f) * 3.14159274f) * (float)up;
        }
    }
    const float rw = (float)(1.0 / up);
    std::vector<float> noise;
    if (noise_seed != 0) {
        noise.resize((size_t)L * NH);
        Gaussian g(noise_seed);
        for (size_t i = 0; i < noise.size(); ++i) noise[i] = g.next();
    }
    std::vector<float> src((size_t)L);
#pragma omp parallel for if (L > 20000)
    for (int j = 0; j < L; ++j) {
        float s = rw * ((float)j + 0.5f) - 0.5f;
        if (s < 0.f) s = 0.f;
        const int i0 = std::min((int)std::floor(s), n - 1);
        const float lam = std::min(std::max(s - (float)i0, 0.f), 1.f);
        const int i1 = i0 + (i0 < n - 1 ? 1 : 0);
        const float f0j = f0[(size_t)(j / up)];
        const float uv = f0j > 10.f ? 1.f : 0.f;
        const float namp = uv * 0.003f + ((1.f - uv) * 0.1f) / 3.f;
        float acc = 0.f;
        for (int h = 0; h < NH; ++h) {
            const float ph = phd[(size_t)i0 * NH + h] * (1.f - lam) + phd[(size_t)i1 * NH + h] * lam;
            float sw = (std::sin(ph) * 0.1f) * uv;
            if (!noise.empty()) sw = sw + namp * noise[(size_t)j * NH + h];
            acc += w9[h] * sw;
        }
        src[(size_t)j] = std::tanh(acc + bias);
    }
    return src;
}

KokoroConfig KokoroConfig::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    const auto j = nlohmann::json::parse(in);
    KokoroConfig c;
    if (!j.contains("vocab") || !j.contains("istftnet") || !j.contains("plbert"))
        throw std::runtime_error(path + " is not a Kokoro config (no vocab, istftnet or plbert)");
    for (auto it = j["vocab"].begin(); it != j["vocab"].end(); ++it) c.vocab[it.key()] = it.value().get<int>();
    c.n_token = j.value("n_token", c.n_token);
    c.hidden = j.value("hidden_dim", c.hidden);
    c.style_dim = j.value("style_dim", c.style_dim);
    c.n_layer = j.value("n_layer", c.n_layer);
    c.max_dur = j.value("max_dur", c.max_dur);
    const auto& pb = j["plbert"];
    c.bert_hidden = pb.value("hidden_size", c.bert_hidden);
    c.bert_heads = pb.value("num_attention_heads", c.bert_heads);
    c.bert_ff = pb.value("intermediate_size", c.bert_ff);
    c.bert_layers = pb.value("num_hidden_layers", c.bert_layers);
    c.bert_ctx = pb.value("max_position_embeddings", c.bert_ctx);
    c.bert_emb = pb.value("embedding_size", c.bert_emb);
    const auto& is = j["istftnet"];
    c.upsample_initial_channel = is.value("upsample_initial_channel", c.upsample_initial_channel);
    c.n_fft = is.value("gen_istft_n_fft", c.n_fft);
    c.hop = is.value("gen_istft_hop_size", c.hop);
    c.upsample_rates = is.value("upsample_rates", c.upsample_rates);
    c.upsample_kernels = is.value("upsample_kernel_sizes", c.upsample_kernels);
    c.resblock_kernels = is.value("resblock_kernel_sizes", c.resblock_kernels);
    c.resblock_dilations = is.value("resblock_dilation_sizes", c.resblock_dilations);
    // The forward pass below is written for v1.0's geometry; refuse any other rather than misread it.
    const bool v10 = c.hidden == 512 && c.style_dim == 128 && c.n_layer == 3 && c.max_dur == 50 &&
                     c.bert_hidden == 768 && c.bert_heads == 12 && c.bert_ff == 2048 && c.bert_emb == 128 &&
                     c.upsample_initial_channel == 512 && c.n_fft == 20 && c.hop == 5 &&
                     c.upsample_rates == std::vector<int>{10, 6} && c.upsample_kernels == std::vector<int>{20, 12} &&
                     c.resblock_kernels == std::vector<int>{3, 7, 11} && j.value("n_mels", 80) == 80 &&
                     j.value("dim_in", 64) == 64 && c.resblock_dilations.size() == 3;
    if (!v10) throw std::runtime_error(path + ": not Kokoro-82M v1.0's geometry, the only one this engine runs");
    return c;
}

Kokoro::Kokoro(const std::string& dir) {
    cfg_ = KokoroConfig::load((fs::path(dir) / "config.json").string());
    for (const char* name : {"kokoro-v1_0.safetensors", "model.safetensors"}) {
        const fs::path p = fs::path(dir) / name;
        if (fs::is_regular_file(p)) {
            weights_path_ = p.string();
            break;
        }
    }
    if (weights_path_.empty())
        throw std::runtime_error(dir + ": no kokoro-v1_0.safetensors or model.safetensors");
    SafetensorsFile f(weights_path_);
    weights_dtype_ = f.main_dtype();
    load(f);
}

Kokoro::~Kokoro() = default;

void Kokoro::load(SafetensorsFile& file) {
    const WeightReader w(file);
    const KokoroConfig& c = cfg_;
    const int E = c.bert_emb, Hb = c.bert_hidden, V = c.n_token, S = c.style_dim, H = c.hidden;

    const std::string al = "bert.encoder.albert_layer_groups.0.albert_layers.0";
    word_emb_ = w.get("bert.embeddings.word_embeddings.weight", (int64_t)V * E);
    pos_emb_ = w.get("bert.embeddings.position_embeddings.weight", (int64_t)c.bert_ctx * E);
    type_emb_ = w.get("bert.embeddings.token_type_embeddings.weight", -1);
    type_emb_.resize((size_t)E);
    emb_ln_g_ = w.get("bert.embeddings.LayerNorm.weight", E);
    emb_ln_b_ = w.get("bert.embeddings.LayerNorm.bias", E);
    emb_map_ = blocks::linear(w, "bert.encoder.embedding_hidden_mapping_in", E, Hb);
    q_ = blocks::linear(w, al + ".attention.query", Hb, Hb);
    k_ = blocks::linear(w, al + ".attention.key", Hb, Hb);
    v_ = blocks::linear(w, al + ".attention.value", Hb, Hb);
    o_ = blocks::linear(w, al + ".attention.dense", Hb, Hb);
    attn_ln_g_ = w.get(al + ".attention.LayerNorm.weight", Hb);
    attn_ln_b_ = w.get(al + ".attention.LayerNorm.bias", Hb);
    ffn_ = blocks::linear(w, al + ".ffn", Hb, c.bert_ff);
    ffn_out_ = blocks::linear(w, al + ".ffn_output", c.bert_ff, Hb);
    ffn_ln_g_ = w.get(al + ".full_layer_layer_norm.weight", Hb);
    ffn_ln_b_ = w.get(al + ".full_layer_layer_norm.bias", Hb);
    bert_encoder_ = blocks::linear(w, "bert_encoder", Hb, H);

    for (int i = 0; i < 3; ++i) {
        dur_lstm_[i] = blocks::lstm(w, "predictor.text_encoder.lstms." + std::to_string(2 * i), H + S, H / 2);
        dur_aln_[i] = blocks::linear(w, "predictor.text_encoder.lstms." + std::to_string(2 * i + 1) + ".fc", S, 2 * H);
    }
    pred_lstm_ = blocks::lstm(w, "predictor.lstm", H + S, H / 2);
    dur_proj_ = blocks::linear(w, "predictor.duration_proj.linear_layer", H, c.max_dur);
    shared_ = blocks::lstm(w, "predictor.shared", H + S, H / 2);
    for (int i = 0; i < 3; ++i) {
        const int din = i <= 1 ? H : H / 2, dout = i == 0 ? H : H / 2;
        f0_blk_[i] = blocks::resblk(w, "predictor.F0." + std::to_string(i), din, dout, S, i == 1);
        n_blk_[i] = blocks::resblk(w, "predictor.N." + std::to_string(i), din, dout, S, i == 1);
    }
    f0_proj_ = blocks::conv(w, "predictor.F0_proj", H / 2, 1, 1, 0, 1, false);
    n_proj_ = blocks::conv(w, "predictor.N_proj", H / 2, 1, 1, 0, 1, false);

    te_emb_ = w.get("text_encoder.embedding.weight", (int64_t)V * H);
    for (int i = 0; i < 3; ++i) {
        const std::string p = "text_encoder.cnn." + std::to_string(i);
        te_cnn_[i] = blocks::conv(w, p + ".0", H, H, 5, 2);
        te_ln_g_[i] = w.get(p + ".1.gamma", H);
        te_ln_b_[i] = w.get(p + ".1.beta", H);
    }
    te_lstm_ = blocks::lstm(w, "text_encoder.lstm", H, H / 2);

    dec_f0_ = blocks::conv(w, "decoder.F0_conv", 1, 1, 3, 1, 1, true, true, 2);
    dec_n_ = blocks::conv(w, "decoder.N_conv", 1, 1, 3, 1, 1, true, true, 2);
    asr_res_ = blocks::conv(w, "decoder.asr_res.0", H, 64, 1, 0);
    encode_ = blocks::resblk(w, "decoder.encode", H + 2, 1024, S, false);
    for (int i = 0; i < 4; ++i)
        decode_[i] = blocks::resblk(w, "decoder.decode." + std::to_string(i), 1024 + 2 + 64, i == 3 ? 512 : 1024, S, i == 3);

    const std::string g = "decoder.generator";
    src_linear_ = blocks::linear(w, g + ".m_source.l_linear", 9, 1);
    const int C0 = c.upsample_initial_channel;
    for (int i = 0; i < 2; ++i) {
        const int ch = C0 >> (i + 1), u = c.upsample_rates[(size_t)i], k = c.upsample_kernels[(size_t)i];
        ups_[i] = blocks::upsampler(w, g + ".ups." + std::to_string(i), C0 >> i, ch, k, u);
        for (int j = 0; j < 3; ++j)
            resblocks_[i * 3 + j] = blocks::resblock1(w, g + ".resblocks." + std::to_string(i * 3 + j), ch,
                                                   c.resblock_kernels[(size_t)j], c.resblock_dilations[(size_t)j], S);
        if (i == 0) {
            const int sf = c.upsample_rates[1];
            noise_convs_[0] = blocks::conv(w, g + ".noise_convs.0", c.n_fft + 2, ch, sf * 2, (sf + 1) / 2, 1, false, true, sf);
            noise_res_[0] = blocks::resblock1(w, g + ".noise_res.0", ch, 7, {1, 3, 5}, S);
        } else {
            noise_convs_[1] = blocks::conv(w, g + ".noise_convs.1", c.n_fft + 2, ch, 1, 0, 1, false);
            noise_res_[1] = blocks::resblock1(w, g + ".noise_res.1", ch, 11, {1, 3, 5}, S);
        }
    }
    conv_post_ = blocks::conv(w, g + ".conv_post", C0 >> 2, c.n_fft + 2, 7, 3);
}

void AdaIN1d::apply(float* x, int t, const float* style) const {
    std::vector<float> h(2 * (size_t)c);
    fc.forward(style, 1, h.data());
    adain(x, c, t, h.data(), h.data() + c, norm_w.empty() ? nullptr : norm_w.data(),
          norm_b.empty() ? nullptr : norm_b.data(), 1e-5f);
}

std::vector<float> AdainResBlk1d::forward(const std::vector<float>& x, int t, const float* style, int& t_out) const {
    std::vector<float> r = x;
    norm1.apply(r.data(), t, style);
    leaky_relu(r.data(), (int64_t)r.size(), 0.2f);
    int tt = t;
    if (upsample) {
        r = pool(r, t);
        tt = pool.out_len(t);
    }
    r = conv1(r, tt);
    norm2.apply(r.data(), tt, style);
    leaky_relu(r.data(), (int64_t)r.size(), 0.2f);
    r = conv2(r, tt);
    std::vector<float> sc;
    if (upsample) {
        sc.resize((size_t)din * 2 * t);
        for (int ch = 0; ch < din; ++ch)
            for (int j = 0; j < t; ++j) {
                const float v = x[(size_t)ch * t + j];
                sc[(size_t)ch * 2 * t + 2 * j] = v;
                sc[(size_t)ch * 2 * t + 2 * j + 1] = v;
            }
    } else {
        sc = x;
    }
    if (din != dout) sc = conv1x1(sc, tt);
    const float rs2 = 1.f / std::sqrt(2.f);
    for (size_t i = 0; i < r.size(); ++i) r[i] = (r[i] + sc[i]) * rs2;
    t_out = tt;
    return r;
}

void AdaINResBlock1::forward(std::vector<float>& x, int t, const float* style) const {
    std::vector<float> xt(x.size()), tmp(x.size());
    for (int i = 0; i < 3; ++i) {
        tmp = x;
        adain1[i].apply(tmp.data(), t, style);
        snake(tmp.data(), ch, t, alpha1[i].data(), xt.data());
        convs1[i].forward(xt.data(), t, tmp.data());
        adain2[i].apply(tmp.data(), t, style);
        snake(tmp.data(), ch, t, alpha2[i].data(), xt.data());
        convs2[i].forward(xt.data(), t, tmp.data());
        for (size_t j = 0; j < x.size(); ++j) x[j] = tmp[j] + x[j];
    }
}

std::vector<float> Kokoro::albert(const std::vector<int>& ids) const {
    const int T = (int)ids.size(), E = cfg_.bert_emb, Hb = cfg_.bert_hidden, NH = cfg_.bert_heads, HD = Hb / NH;
    std::vector<float> e((size_t)T * E);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < E; ++j)
            e[(size_t)t * E + j] = (word_emb_[(size_t)ids[(size_t)t] * E + j] + type_emb_[(size_t)j]) +
                                   pos_emb_[(size_t)t * E + j];
    layer_norm_rows(e.data(), T, E, emb_ln_g_.data(), emb_ln_b_.data(), 1e-12f);
    std::vector<float> x = emb_map_(e, T);
    const float scale = 1.f / std::sqrt((float)HD);
    std::vector<float> ctx((size_t)T * Hb);
    for (int layer = 0; layer < cfg_.bert_layers; ++layer) {
        const std::vector<float> q = q_(x, T), k = k_(x, T), v = v_(x, T);
#pragma omp parallel for if (T > 16)
        for (int hq = 0; hq < NH * T; ++hq) {
            const int h = hq / T, i = hq % T;
            std::vector<float> p((size_t)T);
            const float* qi = q.data() + (size_t)i * Hb + (size_t)h * HD;
            float mx = -INFINITY;
            for (int j = 0; j < T; ++j) {
                const float* kj = k.data() + (size_t)j * Hb + (size_t)h * HD;
                float s = 0.f;
                for (int d = 0; d < HD; ++d) s += qi[d] * kj[d];
                p[(size_t)j] = s * scale;
                mx = std::max(mx, p[(size_t)j]);
            }
            double sum = 0.0;
            for (int j = 0; j < T; ++j) {
                p[(size_t)j] = std::exp(p[(size_t)j] - mx);
                sum += p[(size_t)j];
            }
            const float inv = (float)(1.0 / sum);
            float* ci = ctx.data() + (size_t)i * Hb + (size_t)h * HD;
            for (int d = 0; d < HD; ++d) ci[d] = 0.f;
            for (int j = 0; j < T; ++j) {
                const float pj = p[(size_t)j] * inv;
                const float* vj = v.data() + (size_t)j * Hb + (size_t)h * HD;
                for (int d = 0; d < HD; ++d) ci[d] += pj * vj[d];
            }
        }
        std::vector<float> a = o_(ctx, T);
        for (size_t i = 0; i < a.size(); ++i) a[i] += x[i];
        layer_norm_rows(a.data(), T, Hb, attn_ln_g_.data(), attn_ln_b_.data(), 1e-12f);
        std::vector<float> f = ffn_(a, T);
        gelu_tanh(f.data(), (int64_t)f.size());
        x = ffn_out_(f, T);
        for (size_t i = 0; i < x.size(); ++i) x[i] += a[i];
        layer_norm_rows(x.data(), T, Hb, ffn_ln_g_.data(), ffn_ln_b_.data(), 1e-12f);
    }
    return x;
}

namespace {
// OFLM_TTS_PROFILE=1 prints each stage's host wall clock, to find where a synthesis spends its time.
struct Laps {
    bool on = std::getenv("OFLM_TTS_PROFILE") != nullptr;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    void lap(const char* what) {
        if (!on) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[open_tts] %-16s %8.1f ms\n", what, std::chrono::duration<double, std::milli>(now - t).count());
        t = now;
    }
};
}  // namespace

std::vector<float> Kokoro::harmonic_source(const std::vector<float>& f0, uint64_t noise_seed) const {
    const int up = cfg_.upsample_rates[0] * cfg_.upsample_rates[1] * cfg_.hop;
    float w9[9];
    for (int h = 0; h < 9; ++h) w9[h] = src_linear_.wt.at((size_t)h);
    return open_tts::harmonic_source(f0, up, w9, src_linear_.b[0], noise_seed);
}

std::vector<float> Kokoro::generator(const std::vector<float>& xin, int t, const float* style,
                                     const std::vector<float>& f0, uint64_t noise_seed, Trace* tr,
                                     const std::vector<float>* har_source) const {
    Laps laps;
    const std::vector<float> src = har_source ? *har_source : harmonic_source(f0, noise_seed);
    if (tr) tr->put("gen.har_source", src);
    laps.lap("source");
    std::vector<float> mag, ph;
    int frames = 0;
    stft_mag_phase(src.data(), (int)src.size(), cfg_.n_fft, cfg_.hop, mag, ph, frames);
    laps.lap("stft");
    const int nb = cfg_.n_fft / 2 + 1;
    std::vector<float> har((size_t)2 * nb * frames);
    std::memcpy(har.data(), mag.data(), mag.size() * sizeof(float));
    std::memcpy(har.data() + mag.size(), ph.data(), ph.size() * sizeof(float));
    if (tr) tr->put("gen.har", har);

    std::vector<float> x = xin;
    int tt = t;
    for (int i = 0; i < 2; ++i) {
        leaky_relu(x.data(), (int64_t)x.size(), 0.1f);
        std::vector<float> xs = noise_convs_[i](har, frames);
        const int ts = noise_convs_[i].out_len(frames);
        noise_res_[i].forward(xs, ts, style);
        x = ups_[i](x, tt);
        tt = ups_[i].out_len(tt);
        if (i == 1) {
            const int ch = ups_[i].cout;
            std::vector<float> p((size_t)ch * (tt + 1));
            for (int c = 0; c < ch; ++c) {
                p[(size_t)c * (tt + 1)] = x[(size_t)c * tt + 1];
                std::memcpy(p.data() + (size_t)c * (tt + 1) + 1, x.data() + (size_t)c * tt, (size_t)tt * sizeof(float));
            }
            x.swap(p);
            ++tt;
        }
        if (ts != tt) throw std::runtime_error("generator: source and upsampled lengths differ");
        for (size_t j = 0; j < x.size(); ++j) x[j] += xs[j];
        std::vector<float> acc;
        for (int j = 0; j < 3; ++j) {
            std::vector<float> r = x;
            resblocks_[i * 3 + j].forward(r, tt, style);
            if (acc.empty()) acc.swap(r);
            else for (size_t q = 0; q < acc.size(); ++q) acc[q] += r[q];
        }
        for (size_t q = 0; q < acc.size(); ++q) acc[q] = acc[q] / 3.f;
        x.swap(acc);
        if (tr) tr->put("gen.stage" + std::to_string(i), x);
        laps.lap(i ? "gen stage 1" : "gen stage 0");
    }
    leaky_relu(x.data(), (int64_t)x.size(), 0.01f);
    std::vector<float> post = conv_post_(x, tt);
    if (tr) tr->put("gen.conv_post", post);
    std::vector<float> spec((size_t)nb * tt), phase((size_t)nb * tt);
    for (size_t j = 0; j < spec.size(); ++j) {
        spec[j] = std::exp(post[j]);
        phase[j] = std::sin(post[spec.size() + j]);
    }
    std::vector<float> audio = istft(spec.data(), phase.data(), tt, cfg_.n_fft, cfg_.hop);
    laps.lap("post + istft");
    return audio;
}

std::vector<float> Kokoro::synthesize(const std::vector<int>& ids, const float* style, float speed,
                                      uint64_t noise_seed, Trace* tr, std::vector<int>* durations,
                                      const std::vector<int>* force_durations) const {
    const int T = (int)ids.size(), H = cfg_.hidden, S = cfg_.style_dim;
    if (T < 3 || T > cfg_.bert_ctx) throw std::runtime_error("Kokoro: 1 to 510 tokens per chunk");
    for (int id : ids)
        if (id < 0 || id >= cfg_.n_token) throw std::runtime_error("Kokoro: token id out of range");
    const float* s_dec = style;
    const float* s_pro = style + S;
    Laps laps;

    const std::vector<float> bert = albert(ids);
    if (tr) tr->put("bert", bert);
    const std::vector<float> d_en = bert_encoder_(bert, T);   // [T][H]
    if (tr) tr->put("d_en", d_en);
    laps.lap("albert");

    std::vector<float> cur((size_t)T * (H + S));
    auto cat_style = [&](const std::vector<float>& a) {
        for (int t = 0; t < T; ++t) {
            std::memcpy(cur.data() + (size_t)t * (H + S), a.data() + (size_t)t * H, (size_t)H * sizeof(float));
            std::memcpy(cur.data() + (size_t)t * (H + S) + H, s_pro, (size_t)S * sizeof(float));
        }
    };
    cat_style(d_en);
    for (int i = 0; i < 3; ++i) {
        std::vector<float> l = dur_lstm_[i](cur, T);
        std::vector<float> gb((size_t)2 * H);
        dur_aln_[i].forward(s_pro, 1, gb.data());
        layer_norm_rows(l.data(), T, H, nullptr, nullptr, 1e-5f);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < H; ++j) {
                float& v = l[(size_t)t * H + j];
                v = (1.f + gb[(size_t)j]) * v + gb[(size_t)(H + j)];
            }
        cat_style(l);
    }
    const std::vector<float> d = cur;   // [T][H + S]
    if (tr) tr->put("pred.d", d);
    const std::vector<float> pl = pred_lstm_(d, T);
    const std::vector<float> dl = dur_proj_(pl, T);
    std::vector<int> dur((size_t)T);
    int F = 0;
    for (int t = 0; t < T; ++t) {
        double s = 0.0;
        for (int j = 0; j < cfg_.max_dur; ++j) s += 1.f / (1.f + std::exp(-dl[(size_t)t * cfg_.max_dur + j]));
        const float dv = (float)s / speed;
        dur[(size_t)t] = std::max(1, (int)std::nearbyint(dv));
    }
    if (tr) tr->put("pred.dur_float", [&] {
        std::vector<float> df((size_t)T);
        for (int t = 0; t < T; ++t) {
            double s = 0.0;
            for (int j = 0; j < cfg_.max_dur; ++j) s += 1.f / (1.f + std::exp(-dl[(size_t)t * cfg_.max_dur + j]));
            df[(size_t)t] = (float)s / speed;
        }
        return df;
    }());
    if (force_durations) {
        if ((int)force_durations->size() != T) throw std::runtime_error("forced durations: one per token");
        dur = *force_durations;
    }
    for (int t = 0; t < T; ++t) F += dur[(size_t)t];
    if (durations) *durations = dur;
    if (tr) {
        std::vector<float> df(dur.begin(), dur.end());
        tr->put("pred.dur", df);
    }
    std::vector<int> tok((size_t)F);
    for (int t = 0, f = 0; t < T; ++t)
        for (int r = 0; r < dur[(size_t)t]; ++r) tok[(size_t)f++] = t;
    laps.lap("durations");

    // F0 and energy, from the shared LSTM over the frame-expanded d
    std::vector<float> en((size_t)F * (H + S));
    for (int f = 0; f < F; ++f)
        std::memcpy(en.data() + (size_t)f * (H + S), d.data() + (size_t)tok[(size_t)f] * (H + S), (size_t)(H + S) * sizeof(float));
    const std::vector<float> shf = transpose(shared_(en, F), F, H);   // [H][F]
    std::vector<float> f0 = shf, nn = shf;
    int tf = F, tn = F;
    for (int i = 0; i < 3; ++i) {
        int o;
        f0 = f0_blk_[i].forward(f0, tf, s_pro, o);
        tf = o;
        nn = n_blk_[i].forward(nn, tn, s_pro, o);
        tn = o;
    }
    f0 = f0_proj_(f0, tf);   // [2F]
    nn = n_proj_(nn, tn);
    laps.lap("F0 and energy");
    if (tr) {
        tr->put("pred.F0", f0);
        tr->put("pred.N", nn);
    }

    // text encoder
    std::vector<float> te((size_t)H * T);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < H; ++j) te[(size_t)j * T + t] = te_emb_[(size_t)ids[(size_t)t] * H + j];
    for (int i = 0; i < 3; ++i) {
        te = te_cnn_[i](te, T);
        layer_norm_channels(te.data(), H, T, te_ln_g_[i].data(), te_ln_b_[i].data(), 1e-5f);
        leaky_relu(te.data(), (int64_t)te.size(), 0.2f);
    }
    const std::vector<float> t_en = transpose(te_lstm_(transpose(te, H, T), T), T, H);   // [H][T]
    if (tr) tr->put("t_en", t_en);
    std::vector<float> asr((size_t)H * F);
    for (int j = 0; j < H; ++j)
        for (int f = 0; f < F; ++f) asr[(size_t)j * F + f] = t_en[(size_t)j * T + tok[(size_t)f]];

    // decoder
    const std::vector<float> f0d = dec_f0_(f0, 2 * F), nd = dec_n_(nn, 2 * F);   // [F] each
    std::vector<float> x((size_t)(H + 2) * F);
    std::memcpy(x.data(), asr.data(), asr.size() * sizeof(float));
    std::memcpy(x.data() + (size_t)H * F, f0d.data(), (size_t)F * sizeof(float));
    std::memcpy(x.data() + (size_t)(H + 1) * F, nd.data(), (size_t)F * sizeof(float));
    int tx;
    x = encode_.forward(x, F, s_dec, tx);
    const std::vector<float> ar = asr_res_(asr, F);   // [64][F]
    bool res = true;
    for (int i = 0; i < 4; ++i) {
        if (res) {
            const int c = decode_[i].din - 66;
            std::vector<float> y((size_t)decode_[i].din * tx);
            std::memcpy(y.data(), x.data(), (size_t)c * tx * sizeof(float));
            std::memcpy(y.data() + (size_t)c * tx, ar.data(), ar.size() * sizeof(float));
            std::memcpy(y.data() + (size_t)(c + 64) * tx, f0d.data(), (size_t)tx * sizeof(float));
            std::memcpy(y.data() + (size_t)(c + 65) * tx, nd.data(), (size_t)tx * sizeof(float));
            x.swap(y);
        }
        int o;
        x = decode_[i].forward(x, tx, s_dec, o);
        tx = o;
        if (decode_[i].upsample) res = false;
    }
    if (tr) tr->put("dec.x", x);
    laps.lap("text enc + decoder");
    std::vector<float> audio = generator(x, tx, s_dec, f0, noise_seed, tr);
    if (tr) tr->put("audio", audio);
    return audio;
}

Voice Voice::load(const std::string& path) {
    SafetensorsFile f(path);
    if (f.tensors().size() != 1) throw std::runtime_error(path + ": a voice file holds one tensor");
    const std::string name = f.tensors().begin()->first;
    Voice v;
    v.name = fs::path(path).stem().string();
    v.table = f.f32(name);
    if (v.table.size() % 256 || v.table.empty()) throw std::runtime_error(path + ": voice rows are not 256 wide");
    v.rows = (int)(v.table.size() / 256);
    return v;
}

const float* Voice::row_for(int phoneme_chars) const {
    const int r = std::min(std::max(phoneme_chars - 1, 0), rows - 1);
    return table.data() + (size_t)r * 256;
}

}  // namespace open_tts
