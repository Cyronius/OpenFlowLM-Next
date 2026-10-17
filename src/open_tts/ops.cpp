// Host ops for the Kokoro TTS engine (2026-10-16).
#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace open_tts {

static DType g_weight_dtype = [] {
    const char* e = std::getenv("OFLM_TTS_WEIGHTS");
    return (e && std::string(e) == "f32") ? DType::F32 : DType::F16;
}();
DType weight_dtype() { return g_weight_dtype; }
void set_weight_dtype(DType t) { g_weight_dtype = t; }

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const uint32_t ax = x & 0x7FFFFFFF;
    if (ax >= 0x7F800000) return (uint16_t)(sign | 0x7C00 | (ax > 0x7F800000 ? 0x200 : 0));
    if (ax >= 0x477FF000) return (uint16_t)(sign | 0x7C00);          // overflows to inf
    if (ax < 0x38800000) {                                           // f16 subnormal or zero
        if (ax < 0x33000000) return (uint16_t)sign;
        const uint32_t m = (ax & 0x7FFFFF) | 0x800000;
        const int shift = 126 - (int)(ax >> 23);
        const uint32_t half = 1u << (shift - 1);
        uint32_t v = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1);
        if (rem > half || (rem == half && (v & 1))) ++v;
        return (uint16_t)(sign | v);
    }
    uint32_t v = ((ax >> 13) - (112u << 10));
    const uint32_t rem = ax & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (v & 1))) ++v;
    return (uint16_t)(sign | v);
}

void Weight::set(const std::vector<float>& v, DType dtype) {
    t = dtype;
    if (t == DType::F16) {
        h.resize(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = f32_to_f16(v[i]);
        f.clear();
        f.shrink_to_fit();
    } else {
        f = v;
        h.clear();
    }
}

namespace {

constexpr int MR = 6;
constexpr int NR = 16;
constexpr int KC = 256;
constexpr int MB = 96;  // rows per tile, a multiple of MR
constexpr double kPi = 3.14159265358979323846;

inline void widen(const uint16_t* src, float* dst, int n) {
    int j = 0;
#if defined(__AVX2__)
    for (; j + 8 <= n; j += 8)
        _mm256_storeu_ps(dst + j, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + j))));
#endif
    for (; j < n; ++j) dst[j] = f16_to_f32(src[j]);
}

void pack_a(const void* A, DType at, int lda, int m0, int mr, int k0, int kc, float* pa) {
    float row[KC];
    for (int i = 0; i < mr; ++i) {
        const float* src;
        if (at == DType::F16) {
            widen(static_cast<const uint16_t*>(A) + (size_t)(m0 + i) * lda + k0, row, kc);
            src = row;
        } else {
            src = static_cast<const float*>(A) + (size_t)(m0 + i) * lda + k0;
        }
        for (int k = 0; k < kc; ++k) pa[k * MR + i] = src[k];
    }
    for (int i = mr; i < MR; ++i)
        for (int k = 0; k < kc; ++k) pa[k * MR + i] = 0.f;
}

void pack_b(const void* const* brow, DType bt, int bcs, int k0, int kc, int n0, int nr, float* pb) {
    for (int k = 0; k < kc; ++k) {
        float* dst = pb + (size_t)k * NR;
        int j = 0;
        if (bt == DType::F16) {
            const uint16_t* src = static_cast<const uint16_t*>(brow[k0 + k]);
            if (bcs == 1) {
                widen(src + n0, dst, nr);
                j = nr;
            } else {
                for (; j < nr; ++j) dst[j] = f16_to_f32(src[(size_t)(n0 + j) * bcs]);
            }
        } else {
            const float* src = static_cast<const float*>(brow[k0 + k]);
            if (bcs == 1) {
                const float* s = src + n0;
                for (; j < nr; ++j) dst[j] = s[j];
            } else {
                for (; j < nr; ++j) dst[j] = src[(size_t)(n0 + j) * bcs];
            }
        }
        for (; j < NR; ++j) dst[j] = 0.f;
    }
}

void micro_kernel(int kc, const float* pa, const float* pb, float* c, int ldc, bool acc, int mr, int nr) {
#if defined(__AVX2__)
    __m256 c0a = _mm256_setzero_ps(), c0b = _mm256_setzero_ps();
    __m256 c1a = _mm256_setzero_ps(), c1b = _mm256_setzero_ps();
    __m256 c2a = _mm256_setzero_ps(), c2b = _mm256_setzero_ps();
    __m256 c3a = _mm256_setzero_ps(), c3b = _mm256_setzero_ps();
    __m256 c4a = _mm256_setzero_ps(), c4b = _mm256_setzero_ps();
    __m256 c5a = _mm256_setzero_ps(), c5b = _mm256_setzero_ps();
    for (int k = 0; k < kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(pb + k * NR);
        const __m256 b1 = _mm256_loadu_ps(pb + k * NR + 8);
        const float* a = pa + k * MR;
        __m256 av = _mm256_broadcast_ss(a + 0);
        c0a = _mm256_fmadd_ps(av, b0, c0a); c0b = _mm256_fmadd_ps(av, b1, c0b);
        av = _mm256_broadcast_ss(a + 1);
        c1a = _mm256_fmadd_ps(av, b0, c1a); c1b = _mm256_fmadd_ps(av, b1, c1b);
        av = _mm256_broadcast_ss(a + 2);
        c2a = _mm256_fmadd_ps(av, b0, c2a); c2b = _mm256_fmadd_ps(av, b1, c2b);
        av = _mm256_broadcast_ss(a + 3);
        c3a = _mm256_fmadd_ps(av, b0, c3a); c3b = _mm256_fmadd_ps(av, b1, c3b);
        av = _mm256_broadcast_ss(a + 4);
        c4a = _mm256_fmadd_ps(av, b0, c4a); c4b = _mm256_fmadd_ps(av, b1, c4b);
        av = _mm256_broadcast_ss(a + 5);
        c5a = _mm256_fmadd_ps(av, b0, c5a); c5b = _mm256_fmadd_ps(av, b1, c5b);
    }
    alignas(32) float tmp[MR * NR];
    _mm256_store_ps(tmp + 0, c0a);  _mm256_store_ps(tmp + 8, c0b);
    _mm256_store_ps(tmp + 16, c1a); _mm256_store_ps(tmp + 24, c1b);
    _mm256_store_ps(tmp + 32, c2a); _mm256_store_ps(tmp + 40, c2b);
    _mm256_store_ps(tmp + 48, c3a); _mm256_store_ps(tmp + 56, c3b);
    _mm256_store_ps(tmp + 64, c4a); _mm256_store_ps(tmp + 72, c4b);
    _mm256_store_ps(tmp + 80, c5a); _mm256_store_ps(tmp + 88, c5b);
    if (nr == NR) {
        for (int i = 0; i < mr; ++i) {
            float* r = c + (size_t)i * ldc;
            __m256 x0 = _mm256_load_ps(tmp + i * NR), x1 = _mm256_load_ps(tmp + i * NR + 8);
            if (acc) {
                x0 = _mm256_add_ps(x0, _mm256_loadu_ps(r));
                x1 = _mm256_add_ps(x1, _mm256_loadu_ps(r + 8));
            }
            _mm256_storeu_ps(r, x0);
            _mm256_storeu_ps(r + 8, x1);
        }
        return;
    }
#else
    float tmp[MR * NR] = {};
    for (int k = 0; k < kc; ++k)
        for (int i = 0; i < MR; ++i) {
            const float a = pa[k * MR + i];
            for (int j = 0; j < NR; ++j) tmp[i * NR + j] += a * pb[k * NR + j];
        }
#endif
    for (int i = 0; i < mr; ++i) {
        float* r = c + (size_t)i * ldc;
        for (int j = 0; j < nr; ++j) r[j] = (acc ? r[j] : 0.f) + tmp[i * NR + j];
    }
}

inline float dot(const float* a, const float* b, int n) {
#if defined(__AVX2__)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    alignas(32) float t[8];
    _mm256_store_ps(t, s0);
    float s = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
#endif
}

inline float dot_f16(const uint16_t* a, const float* b, int n) {
#if defined(__AVX2__)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 a0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
        const __m256 a1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 8)));
        s0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b + i + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    alignas(32) float t[8];
    _mm256_store_ps(t, s0);
    float s = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    for (; i < n; ++i) s += f16_to_f32(a[i]) * b[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += f16_to_f32(a[i]) * b[i];
    return s;
#endif
}

inline float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }

size_t elem(DType t) { return t == DType::F16 ? 2 : 4; }

}  // namespace

void sgemm_rows(int M, int N, int K, const void* A, DType at, int lda, const void* const* brow, DType bt, int bcs,
                float* C, int ldc, bool accumulate) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        if (!accumulate)
            for (int i = 0; i < M; ++i) std::fill(C + (size_t)i * ldc, C + (size_t)i * ldc + N, 0.f);
        return;
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const int mt = (M + MB - 1) / MB;
    int nb = 512;
    while (nb > 64 && (long long)mt * ((N + nb - 1) / nb) < 2LL * threads) nb /= 2;
    const int nt = (N + nb - 1) / nb;
    const int tiles = mt * nt;
    const bool par = tiles > 1 && (double)M * N * K > 4e5;
#pragma omp parallel if (par)
    {
        std::vector<float> pa((size_t)MB * KC), pb((size_t)nb * KC);
#pragma omp for schedule(dynamic)
        for (int t = 0; t < tiles; ++t) {
            const int m0 = (t / nt) * MB, m1 = std::min(M, m0 + MB);
            const int n0 = (t % nt) * nb, n1 = std::min(N, n0 + nb);
            for (int k0 = 0; k0 < K; k0 += KC) {
                const int kc = std::min(KC, K - k0);
                const bool acc = accumulate || k0 > 0;
                int ns = 0;
                for (int n = n0; n < n1; n += NR, ++ns)
                    pack_b(brow, bt, bcs, k0, kc, n, std::min(NR, n1 - n), pb.data() + (size_t)ns * kc * NR);
                int ms = 0;
                for (int m = m0; m < m1; m += MR, ++ms)
                    pack_a(A, at, lda, m, std::min(MR, m1 - m), k0, kc, pa.data() + (size_t)ms * kc * MR);
                ms = 0;
                for (int m = m0; m < m1; m += MR, ++ms) {
                    ns = 0;
                    for (int n = n0; n < n1; n += NR, ++ns)
                        micro_kernel(kc, pa.data() + (size_t)ms * kc * MR, pb.data() + (size_t)ns * kc * NR,
                                     C + (size_t)m * ldc + n, ldc, acc, std::min(MR, m1 - m), std::min(NR, n1 - n));
                }
            }
        }
    }
}

void sgemm(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
           bool accumulate) {
    std::vector<const void*> rows((size_t)std::max(K, 0));
    for (int k = 0; k < K; ++k) rows[(size_t)k] = B + (size_t)k * ldb;
    sgemm_rows(M, N, K, A, DType::F32, lda, rows.data(), DType::F32, 1, C, ldc, accumulate);
}

void Conv1d::set_weight(const std::vector<float>& tw) {
    const int cpg = cin / groups;
    if ((int64_t)tw.size() != (int64_t)cout * cpg * k)
        throw std::runtime_error("Conv1d weight has the wrong size");
    if (groups != 1) {
        w.set(tw, DType::F32);
        return;
    }
    std::vector<float> v(tw.size());
    for (int co = 0; co < cout; ++co)
        for (int ci = 0; ci < cin; ++ci)
            for (int tap = 0; tap < k; ++tap)
                v[(size_t)co * k * cin + (size_t)tap * cin + ci] = tw[((size_t)co * cin + ci) * k + tap];
    w.set(v, weight_dtype());
}

void Conv1d::forward(const float* x, int t, float* y) const {
    const int to = out_len(t);
    if (to <= 0) return;
    const int tp = t + 2 * pad;
    std::vector<float> xp;
    const float* base = x;
    int ld = t;
    if (pad > 0) {
        xp.assign((size_t)cin * tp, 0.f);
        for (int ci = 0; ci < cin; ++ci)
            std::memcpy(xp.data() + (size_t)ci * tp + pad, x + (size_t)ci * t, (size_t)t * sizeof(float));
        base = xp.data();
        ld = tp;
    }
    if (groups == 1) {
        std::vector<const void*> rows((size_t)k * cin);
        for (int tap = 0; tap < k; ++tap)
            for (int ci = 0; ci < cin; ++ci)
                rows[(size_t)tap * cin + ci] = base + (size_t)ci * ld + (size_t)tap * dil;
        sgemm_rows(cout, to, k * cin, w.ptr(), w.t, k * cin, rows.data(), DType::F32, stride, y, to, false);
    } else {
        const int cpg = cin / groups, opg = cout / groups;
        const float* wf = w.f.data();
#pragma omp parallel for if ((long long)cout * to * cpg * k > 200000)
        for (int co = 0; co < cout; ++co) {
            const int g = co / opg;
            float* yr = y + (size_t)co * to;
            for (int n = 0; n < to; ++n) {
                float s = 0.f;
                for (int ci = 0; ci < cpg; ++ci) {
                    const float* xr = base + (size_t)(g * cpg + ci) * ld + (size_t)n * stride;
                    const float* wr = wf + ((size_t)co * cpg + ci) * k;
                    for (int tap = 0; tap < k; ++tap) s += wr[tap] * xr[(size_t)tap * dil];
                }
                yr[n] = s;
            }
        }
    }
    if (!b.empty())
        for (int co = 0; co < cout; ++co) {
            float* yr = y + (size_t)co * to;
            const float bv = b[(size_t)co];
            for (int n = 0; n < to; ++n) yr[n] += bv;
        }
}

std::vector<float> Conv1d::operator()(const std::vector<float>& x, int t) const {
    std::vector<float> y((size_t)cout * std::max(out_len(t), 0));
    forward(x.data(), t, y.data());
    return y;
}

void ConvTranspose1d::set_weight(const std::vector<float>& tw) {
    const int opg = cout / groups;
    if ((int64_t)tw.size() != (int64_t)cin * opg * k)
        throw std::runtime_error("ConvTranspose1d weight has the wrong size");
    if (groups != 1) {
        if (groups != cin || cin != cout)
            throw std::runtime_error("ConvTranspose1d: only dense and depthwise are implemented");
        w.set(tw, DType::F32);
        return;
    }
    std::vector<float> v(tw.size());
    for (int ci = 0; ci < cin; ++ci)
        for (int co = 0; co < cout; ++co)
            for (int tap = 0; tap < k; ++tap)
                v[((size_t)tap * cout + co) * cin + ci] = tw[((size_t)ci * cout + co) * k + tap];
    w.set(v, weight_dtype());
}

void ConvTranspose1d::forward(const float* x, int t, float* y) const {
    const int to = out_len(t);
    std::fill(y, y + (size_t)cout * to, 0.f);
    if (groups == 1) {
        const int tb = 512;
        std::vector<float> g((size_t)k * cout * std::min(tb, t));
        std::vector<const void*> rows((size_t)cin);
        for (int i0 = 0; i0 < t; i0 += tb) {
            const int nb = std::min(tb, t - i0);
            for (int ci = 0; ci < cin; ++ci) rows[(size_t)ci] = x + (size_t)ci * t + i0;
            sgemm_rows(k * cout, nb, cin, w.ptr(), w.t, cin, rows.data(), DType::F32, 1, g.data(), nb, false);
#pragma omp parallel for if ((long long)cout * nb * k > 100000)
            for (int co = 0; co < cout; ++co) {
                float* yr = y + (size_t)co * to;
                for (int tap = 0; tap < k; ++tap) {
                    const float* gr = g.data() + ((size_t)tap * cout + co) * nb;
                    for (int j = 0; j < nb; ++j) {
                        const int o = (i0 + j) * stride - pad + tap;
                        if (o >= 0 && o < to) yr[o] += gr[j];
                    }
                }
            }
        }
    } else {
        const float* wf = w.f.data();
#pragma omp parallel for if ((long long)cin * t * k > 100000)
        for (int c = 0; c < cin; ++c) {
            const float* xr = x + (size_t)c * t;
            const float* wr = wf + (size_t)c * k;
            float* yr = y + (size_t)c * to;
            for (int i = 0; i < t; ++i)
                for (int tap = 0; tap < k; ++tap) {
                    const int o = i * stride - pad + tap;
                    if (o >= 0 && o < to) yr[o] += xr[i] * wr[tap];
                }
        }
    }
    if (!b.empty())
        for (int co = 0; co < cout; ++co) {
            float* yr = y + (size_t)co * to;
            const float bv = b[(size_t)co];
            for (int n = 0; n < to; ++n) yr[n] += bv;
        }
}

std::vector<float> ConvTranspose1d::operator()(const std::vector<float>& x, int t) const {
    std::vector<float> y((size_t)cout * out_len(t));
    forward(x.data(), t, y.data());
    return y;
}

void Linear::set_weight(const std::vector<float>& tw) {
    if ((int64_t)tw.size() != (int64_t)in * out) throw std::runtime_error("Linear weight has the wrong size");
    std::vector<float> v(tw.size());
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i) v[(size_t)i * out + o] = tw[(size_t)o * in + i];
    wt.set(v, weight_dtype());
}

void Linear::forward(const float* x, int rows, float* y) const {
    std::vector<const void*> brow((size_t)in);
    const char* base = static_cast<const char*>(wt.ptr());
    for (int i = 0; i < in; ++i) brow[(size_t)i] = base + (size_t)i * out * elem(wt.t);
    sgemm_rows(rows, out, in, x, DType::F32, in, brow.data(), wt.t, 1, y, out, false);
    if (!b.empty())
        for (int r = 0; r < rows; ++r) {
            float* yr = y + (size_t)r * out;
            for (int o = 0; o < out; ++o) yr[o] += b[(size_t)o];
        }
}

std::vector<float> Linear::operator()(const std::vector<float>& x, int rows) const {
    std::vector<float> y((size_t)rows * out);
    forward(x.data(), rows, y.data());
    return y;
}

std::vector<float> BiLSTM::operator()(const std::vector<float>& x, int t) const {
    const int H = hidden, G = 4 * hidden;
    std::vector<float> y((size_t)t * 2 * H);
    std::vector<float> gx[2];
    for (int d = 0; d < 2; ++d) {
        gx[d].resize((size_t)t * G);
        std::vector<const void*> brow((size_t)in);
        const char* base = static_cast<const char*>(wih_t[d].ptr());
        for (int i = 0; i < in; ++i) brow[(size_t)i] = base + (size_t)i * G * elem(wih_t[d].t);
        sgemm_rows(t, G, in, x.data(), DType::F32, in, brow.data(), wih_t[d].t, 1, gx[d].data(), G, false);
        for (int s = 0; s < t; ++s) {
            float* r = gx[d].data() + (size_t)s * G;
            for (int j = 0; j < G; ++j) r[j] += bias[d][(size_t)j];
        }
    }
#pragma omp parallel for num_threads(2)
    for (int d = 0; d < 2; ++d) {
        std::vector<float> h((size_t)H, 0.f), c((size_t)H, 0.f), g((size_t)G);
        const Weight& W = whh[d];
        for (int s = 0; s < t; ++s) {
            const int ti = d == 0 ? s : t - 1 - s;
            const float* gr = gx[d].data() + (size_t)ti * G;
            for (int r = 0; r < G; ++r)
                g[(size_t)r] = gr[r] + (W.t == DType::F16 ? dot_f16(W.h.data() + (size_t)r * H, h.data(), H)
                                                          : dot(W.f.data() + (size_t)r * H, h.data(), H));
            for (int j = 0; j < H; ++j) {
                const float ig = sigmoidf(g[(size_t)j]);
                const float fg = sigmoidf(g[(size_t)(H + j)]);
                const float gg = std::tanh(g[(size_t)(2 * H + j)]);
                const float og = sigmoidf(g[(size_t)(3 * H + j)]);
                c[(size_t)j] = fg * c[(size_t)j] + ig * gg;
                h[(size_t)j] = og * std::tanh(c[(size_t)j]);
            }
            std::memcpy(y.data() + (size_t)ti * 2 * H + (size_t)d * H, h.data(), (size_t)H * sizeof(float));
        }
    }
    return y;
}

std::vector<float> fold_weight_norm(const std::vector<float>& g, const std::vector<float>& v, int rows) {
    if (rows <= 0 || (int64_t)g.size() != rows || v.size() % (size_t)rows)
        throw std::runtime_error("weight norm: g and v do not agree");
    const size_t cols = v.size() / (size_t)rows;
    std::vector<float> w(v.size());
    for (int r = 0; r < rows; ++r) {
        const float* vr = v.data() + (size_t)r * cols;
        double ss = 0.0;
        for (size_t j = 0; j < cols; ++j) ss += (double)vr[j] * vr[j];
        const float scale = g[(size_t)r] / (float)std::sqrt(ss);
        float* wr = w.data() + (size_t)r * cols;
        for (size_t j = 0; j < cols; ++j) wr[j] = vr[j] * scale;
    }
    return w;
}

void layer_norm_rows(float* x, int rows, int cols, const float* gamma, const float* beta, float eps) {
#pragma omp parallel for if ((long long)rows * cols > 200000)
    for (int r = 0; r < rows; ++r) {
        float* xr = x + (size_t)r * cols;
        double m = 0.0, v = 0.0;
        for (int j = 0; j < cols; ++j) m += xr[j];
        m /= cols;
        for (int j = 0; j < cols; ++j) v += ((double)xr[j] - m) * ((double)xr[j] - m);
        v /= cols;
        const float mf = (float)m, inv = (float)(1.0 / std::sqrt(v + eps));
        for (int j = 0; j < cols; ++j) {
            float y = (xr[j] - mf) * inv;
            if (gamma) y = y * gamma[j];
            if (beta) y = y + beta[j];
            xr[j] = y;
        }
    }
}

void layer_norm_channels(float* x, int c, int t, const float* gamma, const float* beta, float eps) {
    std::vector<double> m((size_t)t, 0.0), v((size_t)t, 0.0);
    for (int ch = 0; ch < c; ++ch) {
        const float* xr = x + (size_t)ch * t;
        for (int j = 0; j < t; ++j) m[(size_t)j] += xr[j];
    }
    for (int j = 0; j < t; ++j) m[(size_t)j] /= c;
    for (int ch = 0; ch < c; ++ch) {
        const float* xr = x + (size_t)ch * t;
        for (int j = 0; j < t; ++j) {
            const double d = (double)xr[j] - m[(size_t)j];
            v[(size_t)j] += d * d;
        }
    }
    std::vector<float> mf((size_t)t), inv((size_t)t);
    for (int j = 0; j < t; ++j) {
        mf[(size_t)j] = (float)m[(size_t)j];
        inv[(size_t)j] = (float)(1.0 / std::sqrt(v[(size_t)j] / c + eps));
    }
    for (int ch = 0; ch < c; ++ch) {
        float* xr = x + (size_t)ch * t;
        for (int j = 0; j < t; ++j) {
            float y = (xr[j] - mf[(size_t)j]) * inv[(size_t)j];
            if (gamma) y = y * gamma[ch];
            if (beta) y = y + beta[ch];
            xr[j] = y;
        }
    }
}

void adain(float* x, int c, int t, const float* gamma, const float* beta,
           const float* norm_w, const float* norm_b, float eps) {
#pragma omp parallel for if ((long long)c * t > 100000)
    for (int ch = 0; ch < c; ++ch) {
        float* xr = x + (size_t)ch * t;
        double m = 0.0, v = 0.0;
        for (int j = 0; j < t; ++j) m += xr[j];
        m /= t;
        for (int j = 0; j < t; ++j) v += ((double)xr[j] - m) * ((double)xr[j] - m);
        v /= t;
        const float mf = (float)m, inv = (float)(1.0 / std::sqrt(v + eps));
        const float g1 = 1.f + gamma[ch], bt = beta[ch];
        const float nw = norm_w ? norm_w[ch] : 1.f, nb = norm_b ? norm_b[ch] : 0.f;
        for (int j = 0; j < t; ++j) {
            float n = (xr[j] - mf) * inv;
            if (norm_w) n = n * nw + nb;
            xr[j] = g1 * n + bt;
        }
    }
}

void leaky_relu(float* x, int64_t n, float slope) {
#pragma omp parallel for if (n > 200000)
    for (int64_t i = 0; i < n; ++i)
        if (!(x[i] > 0.f)) x[i] = x[i] * slope;
}

void snake(const float* x, int c, int t, const float* alpha, float* y) {
#pragma omp parallel for if ((long long)c * t > 50000)
    for (int ch = 0; ch < c; ++ch) {
        const float a = alpha[ch], ia = 1.f / a;
        const float* xr = x + (size_t)ch * t;
        float* yr = y + (size_t)ch * t;
        for (int j = 0; j < t; ++j) {
            const float s = std::sin(a * xr[j]);
            yr[j] = xr[j] + ia * (s * s);
        }
    }
}

void gelu_tanh(float* x, int64_t n) {
    const float k = 0.7978845608028654f;
#pragma omp parallel for if (n > 100000)
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = (0.5f * v) * (1.f + std::tanh(k * (v + 0.044715f * (v * v * v))));
    }
}

// torch.hann_window(n, periodic=True) in float32, the way torch computes it.
static std::vector<double> hann_periodic(int n) {
    std::vector<double> w((size_t)n);
    const float step = (float)(kPi * 2.0 / n);
    for (int i = 0; i < n; ++i) w[(size_t)i] = (double)(0.5f + (-0.5f) * std::cos((float)i * step));
    return w;
}

void stft_mag_phase(const float* x, int len, int n_fft, int hop, std::vector<float>& mag,
                    std::vector<float>& phase, int& frames) {
    const int pad = n_fft / 2, nb = n_fft / 2 + 1;
    if (len <= pad) throw std::runtime_error("stft: signal shorter than the reflect padding");
    frames = 1 + len / hop;
    std::vector<float> xp((size_t)len + 2 * pad);
    for (int i = 0; i < len + 2 * pad; ++i) {
        int s = i - pad;
        if (s < 0) s = -s;
        if (s >= len) s = 2 * (len - 1) - s;
        xp[(size_t)i] = x[s];
    }
    const std::vector<double> w = hann_periodic(n_fft);
    std::vector<double> cs((size_t)nb * n_fft), sn((size_t)nb * n_fft);
    for (int k = 0; k < nb; ++k)
        for (int n = 0; n < n_fft; ++n) {
            const double a = 2.0 * kPi * (double)((long long)k * n % n_fft) / n_fft;
            cs[(size_t)k * n_fft + n] = std::cos(a);
            sn[(size_t)k * n_fft + n] = std::sin(a);
        }
    mag.assign((size_t)nb * frames, 0.f);
    phase.assign((size_t)nb * frames, 0.f);
#pragma omp parallel for if (frames > 2000)
    for (int f = 0; f < frames; ++f) {
        double seg[64];
        for (int n = 0; n < n_fft; ++n) seg[n] = (double)(xp[(size_t)f * hop + n] * (float)w[(size_t)n]);
        for (int k = 0; k < nb; ++k) {
            double re = 0.0, im = 0.0;
            for (int n = 0; n < n_fft; ++n) {
                re += seg[n] * cs[(size_t)k * n_fft + n];
                im -= seg[n] * sn[(size_t)k * n_fft + n];
            }
            if (k == 0 || 2 * k == n_fft) im = 0.0;  // a real FFT's DC and Nyquist bins are exactly real
            mag[(size_t)k * frames + f] = (float)std::hypot(re, im);
            phase[(size_t)k * frames + f] = (float)std::atan2(im, re);
        }
    }
}

std::vector<float> istft(const float* mag, const float* ph, int frames, int n_fft, int hop) {
    const int pad = n_fft / 2, nb = n_fft / 2 + 1;
    const int full = n_fft + hop * (frames - 1);
    const std::vector<double> w = hann_periodic(n_fft);
    std::vector<double> cs((size_t)nb * n_fft), sn((size_t)nb * n_fft);
    for (int k = 0; k < nb; ++k)
        for (int n = 0; n < n_fft; ++n) {
            const double a = 2.0 * kPi * (double)((long long)k * n % n_fft) / n_fft;
            cs[(size_t)k * n_fft + n] = std::cos(a);
            sn[(size_t)k * n_fft + n] = std::sin(a);
        }
    std::vector<double> frame_out((size_t)frames * n_fft);
#pragma omp parallel for if (frames > 2000)
    for (int f = 0; f < frames; ++f) {
        double re[33], im[33];
        for (int k = 0; k < nb; ++k) {
            const float m = mag[(size_t)k * frames + f], p = ph[(size_t)k * frames + f];
            re[k] = (double)(m * std::cos(p));
            im[k] = (double)(m * std::sin(p));
        }
        for (int n = 0; n < n_fft; ++n) {
            double s = re[0];
            if (n_fft % 2 == 0) s += (n % 2 ? -re[nb - 1] : re[nb - 1]);
            const int last = n_fft % 2 == 0 ? nb - 1 : nb;
            for (int k = 1; k < last; ++k)
                s += 2.0 * (re[k] * cs[(size_t)k * n_fft + n] - im[k] * sn[(size_t)k * n_fft + n]);
            frame_out[(size_t)f * n_fft + n] = (s / n_fft) * w[(size_t)n];
        }
    }
    std::vector<double> y((size_t)full, 0.0), env((size_t)full, 0.0);
    for (int f = 0; f < frames; ++f)
        for (int n = 0; n < n_fft; ++n) {
            y[(size_t)f * hop + n] += frame_out[(size_t)f * n_fft + n];
            env[(size_t)f * hop + n] += w[(size_t)n] * w[(size_t)n];
        }
    std::vector<float> out((size_t)(full - 2 * pad));
    for (int i = pad; i < full - pad; ++i) {
        if (env[(size_t)i] < 1e-11) throw std::runtime_error("istft: window envelope vanishes");
        out[(size_t)(i - pad)] = (float)(y[(size_t)i] / env[(size_t)i]);
    }
    return out;
}

double Gaussian::uniform() {
    uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return ((double)(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

float Gaussian::next() {
    if (have_) {
        have_ = false;
        return (float)spare_;
    }
    const double u1 = uniform(), u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1)), th = 2.0 * kPi * u2;
    spare_ = r * std::sin(th);
    have_ = true;
    return (float)(r * std::cos(th));
}

}  // namespace open_tts
