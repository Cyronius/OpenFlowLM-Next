// Kokoro-82M (StyleTTS 2 + ISTFTNet) on the host, fp32 (2026-10-16).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ops.hpp"

namespace open_tts {

class SafetensorsFile;

// Intermediate outputs by block name, for checking against the PyTorch reference.
struct Trace {
    std::map<std::string, std::vector<float>> t;
    void put(const std::string& name, const std::vector<float>& v) { t[name] = v; }
};

struct AdaIN1d {
    int c = 0;
    Linear fc;                         // style -> [gamma | beta]
    std::vector<float> norm_w, norm_b;  // InstanceNorm1d affine; empty means identity
    void apply(float* x, int t, const float* style) const;
};

// StyleTTS 2's AdainResBlk1d, the predictor's and the decoder's block.
struct AdainResBlk1d {
    int din = 0, dout = 0;
    bool upsample = false;
    Conv1d conv1, conv2, conv1x1;
    AdaIN1d norm1, norm2;
    ConvTranspose1d pool;   // depthwise, when upsampling
    std::vector<float> forward(const std::vector<float>& x, int t, const float* style, int& t_out) const;
};

// ISTFTNet's AdaINResBlock1: three dilated (Snake, conv) pairs with AdaIN.
struct AdaINResBlock1 {
    int ch = 0;
    Conv1d convs1[3], convs2[3];
    AdaIN1d adain1[3], adain2[3];
    std::vector<float> alpha1[3], alpha2[3];
    void forward(std::vector<float>& x, int t, const float* style) const;
};

struct KokoroConfig {
    std::map<std::string, int> vocab;   // phoneme (one UTF-8 character) -> token id
    int n_token = 178, hidden = 512, style_dim = 128, n_layer = 3, max_dur = 50;
    int bert_hidden = 768, bert_heads = 12, bert_ff = 2048, bert_layers = 12, bert_emb = 128, bert_ctx = 512;
    int upsample_initial_channel = 512, n_fft = 20, hop = 5;
    std::vector<int> upsample_rates{10, 6}, upsample_kernels{20, 12}, resblock_kernels{3, 7, 11};
    std::vector<std::vector<int>> resblock_dilations{{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    static KokoroConfig load(const std::string& path);
};

class Kokoro {
public:
    // dir holds config.json and the weights (kokoro-v1_0.safetensors or model.safetensors)
    explicit Kokoro(const std::string& dir);
    ~Kokoro();

    const KokoroConfig& config() const { return cfg_; }
    const std::string& weights_path() const { return weights_path_; }
    const std::string& weights_dtype() const { return weights_dtype_; }

    // ids: token ids including the leading and trailing 0. style: one 256-wide voice row.
    // noise_seed == 0 turns the harmonic source's noise off, which the parity check uses.
    // force_durations replaces the predicted frame counts, so a check can follow the reference's.
    std::vector<float> synthesize(const std::vector<int>& ids, const float* style, float speed,
                                  uint64_t noise_seed, Trace* trace = nullptr,
                                  std::vector<int>* durations = nullptr,
                                  const std::vector<int>* force_durations = nullptr) const;

    // The halves synthesize() runs, exposed so a check can feed one block the reference's input.
    std::vector<float> generator(const std::vector<float>& x, int t, const float* style,
                                 const std::vector<float>& f0, uint64_t noise_seed, Trace* trace,
                                 const std::vector<float>* har_source = nullptr) const;
    std::vector<float> harmonic_source(const std::vector<float>& f0, uint64_t noise_seed) const;

private:
    void load(SafetensorsFile& f);
    std::vector<float> albert(const std::vector<int>& ids) const;

    KokoroConfig cfg_;
    std::string weights_path_, weights_dtype_;

    // PL-BERT (ALBERT, one shared layer)
    std::vector<float> word_emb_, pos_emb_, type_emb_, emb_ln_g_, emb_ln_b_;
    Linear emb_map_, q_, k_, v_, o_, ffn_, ffn_out_;
    std::vector<float> attn_ln_g_, attn_ln_b_, ffn_ln_g_, ffn_ln_b_;
    Linear bert_encoder_;
    // prosody predictor
    BiLSTM dur_lstm_[3];
    Linear dur_aln_[3];
    BiLSTM pred_lstm_, shared_;
    Linear dur_proj_;
    AdainResBlk1d f0_blk_[3], n_blk_[3];
    Conv1d f0_proj_, n_proj_;
    // text encoder
    std::vector<float> te_emb_;
    Conv1d te_cnn_[3];
    std::vector<float> te_ln_g_[3], te_ln_b_[3];
    BiLSTM te_lstm_;
    // decoder
    Conv1d dec_f0_, dec_n_, asr_res_;
    AdainResBlk1d encode_, decode_[4];
    // generator
    Linear src_linear_;
    Conv1d noise_convs_[2], conv_post_;
    AdaINResBlock1 noise_res_[2], resblocks_[6];
    ConvTranspose1d ups_[2];
};

// Reads tensors by PyTorch module path, whatever names or conv layout a conversion stored,
// folding weight norm (weight_g, weight_v) at load. Exposed for the block tests.
class WeightReader {
public:
    explicit WeightReader(const SafetensorsFile& f) : f_(f) {}
    std::string find(const std::string& name) const;
    bool has(const std::string& name) const { return !find(name).empty(); }
    std::vector<float> get(const std::string& name, int64_t numel) const;
    std::vector<float> conv3(const std::string& name, int d0, int d1, int d2) const;
    std::vector<float> wn3(const std::string& prefix, int d0, int d1, int d2) const;
private:
    std::vector<std::string> candidates(const std::string& name) const;
    const SafetensorsFile& f_;
};

namespace blocks {
Linear linear(const WeightReader& w, const std::string& p, int in, int out, bool bias = true);
Conv1d conv(const WeightReader& w, const std::string& p, int cin, int cout, int k, int pad, int dil = 1,
            bool norm = true, bool bias = true, int stride = 1);
BiLSTM lstm(const WeightReader& w, const std::string& p, int in, int hidden);
AdaIN1d adain(const WeightReader& w, const std::string& p, int style, int c);
AdainResBlk1d resblk(const WeightReader& w, const std::string& p, int din, int dout, int style, bool up);
AdaINResBlock1 resblock1(const WeightReader& w, const std::string& p, int ch, int k, const std::vector<int>& dil,
                         int style);
ConvTranspose1d upsampler(const WeightReader& w, const std::string& p, int cin, int cout, int k, int stride);
}  // namespace blocks

// SineGen + SourceModuleHnNSF over an F0 curve at frame rate: `up` samples per F0 value,
// nine harmonics merged by l_linear (w9, b) and tanh. noise_seed 0 means no noise.
std::vector<float> harmonic_source(const std::vector<float>& f0, int up, const float* w9, float b,
                                   uint64_t noise_seed);

// A voice pack: 510 style rows of 256. KPipeline takes row len(phonemes) - 1.
struct Voice {
    std::string name;
    std::vector<float> table;   // [rows][256]
    int rows = 0;
    static Voice load(const std::string& path);
    const float* row_for(int phoneme_chars) const;
};

}  // namespace open_tts
