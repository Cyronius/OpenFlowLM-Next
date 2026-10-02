/// \file policy.hpp
/// \brief Diffusion-LM decode policies, independent of the engine that runs them.
///
/// A policy decides what each forward pass contains and what to do with its logits;
/// a backend (the llama.cpp reference driver in utilities/dllm-ref, later the open
/// engine) runs the passes. The loop a backend runs:
///
///     policy->start(prompt);
///     while (!policy->done()) {
///         Pass p = policy->next();
///         run p.rows (write each row's K/V at p.rows[i].slot, RoPE at .pos, mask per p.attn)
///         policy->feed(logits of p.logit_rows, vocab);
///         drop every cache row at slot >= policy->keep();
///         stream policy->take();
///     }
///
/// For every policy here a row's slot is >= keep() exactly when its position is, so a
/// backend whose cache is keyed by position (llama.cpp) can trim by position.
///
/// Plan: .claude/plans/diffusion-lm-support.md (worktree feat/dllm).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace open_dllm {

/// One row of a pass. `slot` is the absolute cache row its K/V is written to; `pos` is
/// the position its RoPE uses. They differ only under WeDLM, which moves filled tokens
/// ahead of masks in physical order while keeping their logical positions.
struct Row {
    int32_t token;
    int32_t pos;
    int32_t slot;
};

enum class Attn {
    Causal,  ///< row j sees the kept cache and rows 0..j of this pass, in physical (row) order
    Block    ///< every row sees the kept cache and every row of this pass
};

struct Pass {
    std::vector<Row> rows;
    Attn attn = Attn::Causal;
    std::vector<int> logit_rows;  ///< indices into rows whose logits feed() expects, in this order
};

/// Sampling as the references apply it per row. temperature 0 is greedy.
struct Sampling {
    float temperature = 0.f;
    int top_k = 0;        ///< 0 = off
    float top_p = 1.f;    ///< 1 = off
    uint64_t seed = 0;    ///< the policy's RNG is reset to this at start()
};

/// What every policy needs from the model and the request.
struct Common {
    int32_t mask_id = -1;
    std::vector<int32_t> stop_ids;
    int max_new_tokens = 512;
    Sampling sampling;
};

class Policy {
public:
    virtual ~Policy() = default;
    /// The request's prompt. The policy's first passes prefill it.
    virtual void start(const std::vector<int32_t>& prompt) = 0;
    virtual bool done() const = 0;
    virtual Pass next() = 0;
    /// logits[i] holds the vocab-wide logits of row next().logit_rows[i].
    virtual void feed(const std::vector<const float*>& logits, size_t vocab) = 0;
    /// After feed(): the backend discards every cache row at slot >= keep().
    virtual int32_t keep() const = 0;
    /// Tokens that became final since the last call, in sequence order (generated
    /// tokens only, stop token included, nothing after it).
    virtual std::vector<int32_t> take() = 0;
    /// Forward passes run so far, for tokens-per-forward accounting (prefill excluded).
    virtual int decode_passes() const = 0;
};

// ---- shared row arithmetic (sampling.cpp) --------------------------------------------

/// softmax of logits[0..n) into probs (double precision accumulation).
void softmax(const float* logits, size_t n, std::vector<double>& probs);
/// Shannon entropy (nats) of a probability vector.
double entropy(const std::vector<double>& probs);
size_t argmax(const float* logits, size_t n);

/// Where the temperature is applied relative to the top-k / top-p filters. The references
/// disagree: SDAR (gh_generate.py) and LLaDA2 (_diffusion_sample) divide first; WeDLM
/// (sampler.py) filters the raw logits and divides last, so the same top_p keeps a
/// different set of tokens at T != 1.
enum class Order { ScaleFirst, FilterFirst };

/// One token from logits under `s`: greedy at temperature 0; otherwise temperature, top-k
/// (every token tied with the k-th is kept) and top-p (a token is kept while the
/// probability mass before it is <= top_p, so the first is always kept), in `order`, then a
/// draw from rng -- the filters every reference here implements. `prob_out` (optional)
/// receives the drawn token's probability under the distribution it was drawn from.
int32_t sample(const float* logits, size_t n, const Sampling& s, std::mt19937_64& rng,
               double* prob_out = nullptr, Order order = Order::ScaleFirst);

// ---- the policies ---------------------------------------------------------------------

struct WeDLMConfig {
    Common common;
    int window = 16;       ///< W; wedlm/config.py default
    double tau = 0.4;      ///< entropy threshold (sampling_params.py)
    double lambda = 0.02;  ///< distance penalty per slot (sampling_params.py)
};

struct SDARConfig {
    Common common;
    int block = 16;
    int steps = 16;        ///< denoising steps per block (the per-step minimum is block / steps)
    bool dynamic = true;   ///< low_confidence_dynamic (else low_confidence_static)
    double tau = 0.9;
};

struct LLaDA2Config {
    Common common;
    int block = 32;
    int steps = 32;
    double threshold = 0.5;
    double editing_threshold = 0.0;
    int max_post_steps = 16;
    int max_steps_per_block = 1000;
    int32_t delete_id = 156930;
    int32_t insert_id = 156931;
    bool eos_early_stop = true;
};

std::unique_ptr<Policy> make_wedlm(const WeDLMConfig& c);
std::unique_ptr<Policy> make_sdar(const SDARConfig& c);
std::unique_ptr<Policy> make_llada2(const LLaDA2Config& c);

}  // namespace open_dllm
