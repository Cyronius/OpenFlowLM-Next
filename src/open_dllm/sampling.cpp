/// \file sampling.cpp
/// \brief The row arithmetic every policy shares: softmax, entropy, argmax, sampling.

#include "open_dllm/policy.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace open_dllm {

void softmax(const float* logits, size_t n, std::vector<double>& probs) {
    probs.resize(n);
    double mx = -INFINITY;
    for (size_t i = 0; i < n; ++i) mx = std::max(mx, static_cast<double>(logits[i]));
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(static_cast<double>(logits[i]) - mx);
        sum += probs[i];
    }
    for (double& p : probs) p /= sum;
}

double entropy(const std::vector<double>& probs) {
    double h = 0;
    for (double p : probs)
        if (p > 0) h -= p * std::log(p);
    return h;
}

size_t argmax(const float* logits, size_t n) {
    return static_cast<size_t>(std::max_element(logits, logits + n) - logits);
}

int32_t sample(const float* logits, size_t n, const Sampling& s, std::mt19937_64& rng, double* prob_out) {
    if (n == 0) throw std::invalid_argument("open_dllm::sample: empty logits");
    if (s.temperature <= 0.f) {
        const size_t t = argmax(logits, n);
        if (prob_out) {
            std::vector<double> p;
            softmax(logits, n, p);
            *prob_out = p[t];
        }
        return static_cast<int32_t>(t);
    }
    // z: the row as the filters see it; `live` marks the tokens still in the support
    std::vector<double> z(logits, logits + n);
    const double inv_t = 1.0 / static_cast<double>(s.temperature);
    for (double& v : z) v *= inv_t;
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return z[a] > z[b]; });
    size_t live = n;  // the support is idx[0..live)
    if (s.top_k > 0 && static_cast<size_t>(s.top_k) < n) {
        // torch.topk(...).values[..., -1] then `z < kth -> -inf`: ties with the k-th stay
        const double kth = z[idx[static_cast<size_t>(s.top_k) - 1]];
        live = static_cast<size_t>(s.top_k);
        while (live < n && z[idx[live]] >= kth) ++live;
    }
    if (s.top_p < 1.f) {
        // softmax over the support, cumulative in descending order; `cum > p` shifted right
        // by one, so token i stays while the mass before it is <= p
        const double mx = z[idx[0]];
        double sum = 0;
        for (size_t i = 0; i < live; ++i) sum += std::exp(z[idx[i]] - mx);
        double before = 0;
        size_t cut = live;
        for (size_t i = 0; i < live; ++i) {
            if (i > 0 && before > static_cast<double>(s.top_p)) { cut = i; break; }
            before += std::exp(z[idx[i]] - mx) / sum;
        }
        live = cut;
    }
    const double mx = z[idx[0]];
    std::vector<double> w(live);
    double sum = 0;
    for (size_t i = 0; i < live; ++i) sum += (w[i] = std::exp(z[idx[i]] - mx));
    std::uniform_real_distribution<double> u(0.0, sum);
    const double r = u(rng);
    double acc = 0;
    size_t pick = live - 1;
    for (size_t i = 0; i < live; ++i) {
        acc += w[i];
        if (r < acc) { pick = i; break; }
    }
    if (prob_out) *prob_out = w[pick] / sum;
    return static_cast<int32_t>(idx[pick]);
}

}  // namespace open_dllm
