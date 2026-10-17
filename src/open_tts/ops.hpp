// Host ops for the Kokoro TTS engine: GEMM, convolutions, LSTM, norms, STFT (2026-10-16).
#pragma once

#include <cstdint>
#include <vector>

namespace open_tts {

enum class DType { F32, F16 };

// F16 by default (the weights rule's 16-bit); OFLM_TTS_WEIGHTS=f32 for checks against an f32 reference.
DType weight_dtype();
void set_weight_dtype(DType t);

uint16_t f32_to_f16(float f);   // round to nearest even
float f16_to_f32(uint16_t h);

// A weight matrix in the dtype chosen at load. Products widen it to f32 as they pack it.
struct Weight {
    DType t = DType::F32;
    std::vector<float> f;
    std::vector<uint16_t> h;
    void set(const std::vector<float>& v, DType dtype);
    const void* ptr() const { return t == DType::F16 ? (const void*)h.data() : (const void*)f.data(); }
    size_t size() const { return t == DType::F16 ? h.size() : f.size(); }
    float at(size_t i) const { return t == DType::F16 ? f16_to_f32(h[i]) : f[i]; }
};

// B is given as row pointers, so a convolution's taps are shifted rows of its input (no im2col).
void sgemm_rows(int M, int N, int K, const void* A, DType at, int lda, const void* const* brow, DType bt,
                int bcs, float* C, int ldc, bool accumulate);
void sgemm(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
           bool accumulate);

// Feature maps are [channels][time], row-major.
struct Conv1d {
    int cin = 0, cout = 0, k = 1, stride = 1, pad = 0, dil = 1, groups = 1;
    Weight w;                  // groups == 1: [cout][k * cin], tap-major; else torch [cout][cin/groups][k], f32
    std::vector<float> b;      // empty when the conv has no bias
    int out_len(int t) const { return (t + 2 * pad - dil * (k - 1) - 1) / stride + 1; }
    void set_weight(const std::vector<float>& torch_w);   // torch [cout][cin/groups][k]
    void forward(const float* x, int t, float* y) const;
    std::vector<float> operator()(const std::vector<float>& x, int t) const;
};

struct ConvTranspose1d {
    int cin = 0, cout = 0, k = 1, stride = 1, pad = 0, out_pad = 0, groups = 1;
    Weight w;                  // groups == 1: [k * cout][cin]; depthwise: torch [cin][1][k], f32
    std::vector<float> b;
    int out_len(int t) const { return (t - 1) * stride - 2 * pad + k + out_pad; }
    void set_weight(const std::vector<float>& torch_w);   // torch [cin][cout/groups][k]
    void forward(const float* x, int t, float* y) const;
    std::vector<float> operator()(const std::vector<float>& x, int t) const;
};

struct Linear {
    int in = 0, out = 0;
    Weight wt;                 // [in][out], W transposed
    std::vector<float> b;
    void set_weight(const std::vector<float>& torch_w);   // torch [out][in]
    // x [rows][in] -> y [rows][out]
    void forward(const float* x, int rows, float* y) const;
    std::vector<float> operator()(const std::vector<float>& x, int rows) const;
};

// torch.nn.LSTM, one layer, bidirectional, batch_first; gate order i, f, g, o.
struct BiLSTM {
    int in = 0, hidden = 0;
    Weight wih_t[2];           // [in][4H]
    Weight whh[2];             // [4H][H]
    std::vector<float> bias[2];   // b_ih + b_hh
    // x [T][in] -> y [T][2H] (forward half, then reverse half)
    std::vector<float> operator()(const std::vector<float>& x, int t) const;
};

// Weight norm (torch.nn.utils.weight_norm, dim 0): w[i] = g[i] * v[i] / ||v[i]||, row i of dim 0.
std::vector<float> fold_weight_norm(const std::vector<float>& g, const std::vector<float>& v, int rows);

void layer_norm_rows(float* x, int rows, int cols, const float* gamma, const float* beta, float eps);
void layer_norm_channels(float* x, int c, int t, const float* gamma, const float* beta, float eps);
// InstanceNorm1d over time, then the AdaIN affine: y = (1 + gamma) * (norm(x) * w + b) + beta.
void adain(float* x, int c, int t, const float* gamma, const float* beta,
           const float* norm_w, const float* norm_b, float eps);
void leaky_relu(float* x, int64_t n, float slope);
// Snake from ISTFTNet: x + (1/a) * sin(a x)^2, one a per channel.
void snake(const float* x, int c, int t, const float* alpha, float* y);
void gelu_tanh(float* x, int64_t n);

// torch.stft with a periodic Hann window, center and reflect padding: frames = 1 + len / hop.
void stft_mag_phase(const float* x, int len, int n_fft, int hop, std::vector<float>& mag,
                    std::vector<float>& phase, int& frames);
// torch.istft(mag * exp(i * phase), n_fft, hop, periodic hann, center) -> hop * (frames - 1) samples.
std::vector<float> istft(const float* mag, const float* phase, int frames, int n_fft, int hop);

// splitmix64 + Box-Muller: the noise the harmonic source adds, reproducible in Python.
class Gaussian {
public:
    explicit Gaussian(uint64_t seed) : s_(seed) {}
    float next();
private:
    uint64_t s_;
    bool have_ = false;
    double spare_ = 0.0;
    double uniform();
};

}  // namespace open_tts
