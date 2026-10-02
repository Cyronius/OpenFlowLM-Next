/// \file policy_sdar.cpp
/// \brief SDAR block-diffusion decode (OPEN-DLLM-SDAR): JetAstra/SDAR generate.py's
///        block_diffusion_generate as a Policy.
///
/// The reference is JetAstra/SDAR `generate.py` (blob a24d2a82), cited below as
/// generate.py:N. Where its siblings differ -- the OpenCompass wrapper
/// (evaluation/opencompass/opencompass/models/huggingface_bd3.py, "the wrapper"), JetEngine,
/// lmdeploy -- the comment says so. utilities/dllm-ref/ref_sdar.py runs generate.py's own
/// loop against a scripted model and records every pass; src/open_dllm/sdar_test.cpp
/// replays that against this file.
///
/// The sequence is cut into absolute-aligned blocks of `block` tokens from position 0
/// (generate.py:79-93). Per request:
///
///   prefill   one Attn::Block pass per FULL prompt block, no logits, keep() = its end.
///             generate.py does them as one forward with a block-causal mask
///             (generate.py:96-106); per block is the same K/V.
///   per block the block holds the prompt's tail (first decode block only) and MASKs.
///     denoise every row of the block, Attn::Block, logits for the MASK rows only;
///             keep() = block start, so the block's K/V is dropped (store_kv=False).
///             Each MASK row samples a token and a confidence; the step fills some.
///     commit  once no MASK is left: the same rows again, no logits, keep() = block end
///             (store_kv=True, generate.py:120-129). JetEngine's notes record pass@1 0.68
///             -> 0.14 without it, so it is never skipped -- generate.py runs it on the
///             request's last block too, and so does this.
///
/// Slot == pos for every row.

#include "open_dllm/policy.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace open_dllm {

namespace {

/// SDAR's <|MASK|> (hf added_tokens.json); used when Common::mask_id is unset.
constexpr int32_t kSdarMaskId = 151669;

class SDARPolicy final : public Policy {
public:
    explicit SDARPolicy(const SDARConfig& c) : c_(c) {
        if (c_.block < 1) throw std::invalid_argument("make_sdar: block must be >= 1");
        // generate.py accepts steps that don't divide the block (the remainder goes to the
        // first steps, generate.py:49-54); the wrapper asserts block % steps == 0. Steps
        // past the block would schedule zero-token steps, and the block could end its
        // steps with MASKs left, where generate.py indexes past its schedule.
        if (c_.steps < 1 || c_.steps > c_.block)
            throw std::invalid_argument("make_sdar: steps must be in [1, block]");
        if (c_.common.mask_id < 0) c_.common.mask_id = kSdarMaskId;
        // get_num_transfer_tokens (generate.py:49-54): the per-step minimum.
        sched_.assign(static_cast<size_t>(c_.steps), c_.block / c_.steps);
        for (int i = 0; i < c_.block % c_.steps; ++i) ++sched_[static_cast<size_t>(i)];
    }

    void start(const std::vector<int32_t>& prompt) override {
        const int B = c_.block;
        p_ = static_cast<int32_t>(prompt.size());
        // max_new_tokens is generate.py's gen_length. The sequence runs to the end of the
        // block holding position P + gen_length - 1 (generate.py:79-81) and every block
        // up to there is decoded and returned whole, so a request with no stop yields
        // ceil((P + max_new) / block) * block - P tokens: up to block - 1 past max_new.
        // Even max_new = 0 decodes the prompt's tail block when P is not block-aligned.
        // JetEngine cuts at max_tokens exactly; generate.py and the wrapper do not.
        // A negative budget is taken as 0.
        const int32_t gen = std::max(0, c_.common.max_new_tokens);
        nblocks_ = (p_ + gen + B - 1) / B;
        x_.assign(static_cast<size_t>(nblocks_) * B, c_.common.mask_id);
        std::copy(prompt.begin(), prompt.end(), x_.begin());
        prefill_blocks_ = p_ / B;
        keep_ = 0;
        decode_passes_ = 0;
        out_.clear();
        stopped_ = false;
        rng_.seed(c_.common.sampling.seed);
        blk_ = 0;
        step_ = 0;
        if (prefill_blocks_ > 0) phase_ = Phase::Prefill;
        else begin_decode();
    }

    bool done() const override { return phase_ == Phase::Done; }

    Pass next() override {
        if (phase_ == Phase::Done) throw std::logic_error("SDAR policy: next() after done()");
        const int32_t b0 = blk_ * c_.block;
        Pass p;
        p.attn = Attn::Block;
        p.rows.reserve(static_cast<size_t>(c_.block));
        for (int32_t i = 0; i < c_.block; ++i) p.rows.push_back({x_[static_cast<size_t>(b0 + i)], b0 + i, b0 + i});
        // generate.py samples every row of the block but only MASK rows can be selected
        // (their confidence is -inf otherwise, generate.py:161,169), so only they need logits.
        if (phase_ == Phase::Denoise) p.logit_rows = mask_rows(b0);
        return p;
    }

    void feed(const std::vector<const float*>& logits, size_t vocab) override {
        const int32_t b0 = blk_ * c_.block;
        switch (phase_) {
        case Phase::Prefill:
            if (!logits.empty()) throw std::invalid_argument("SDAR policy: a prefill pass takes no logits");
            keep_ = b0 + c_.block;
            if (++blk_ == prefill_blocks_) begin_decode();
            return;
        case Phase::Denoise:
            denoise(b0, logits, vocab);
            return;
        case Phase::Commit:
            if (!logits.empty()) throw std::invalid_argument("SDAR policy: the commit pass takes no logits");
            keep_ = b0 + c_.block;
            ++decode_passes_;
            // generate.py:199-200: stop after a block that holds a stop id at a generated
            // position, or after the last block.
            if (stopped_ || blk_ + 1 == nblocks_) {
                phase_ = Phase::Done;
            } else {
                ++blk_;
                step_ = 0;
                phase_ = Phase::Denoise;
            }
            return;
        case Phase::Done:
            break;
        }
        throw std::logic_error("SDAR policy: feed() after done()");
    }

    int32_t keep() const override { return keep_; }

    std::vector<int32_t> take() override {
        std::vector<int32_t> t;
        t.swap(out_);
        return t;
    }

    int decode_passes() const override { return decode_passes_; }

private:
    enum class Phase { Prefill, Denoise, Commit, Done };

    void begin_decode() {
        if (prefill_blocks_ == nblocks_) {  // aligned prompt and max_new_tokens 0
            phase_ = Phase::Done;
            return;
        }
        blk_ = prefill_blocks_;
        step_ = 0;
        phase_ = Phase::Denoise;
    }

    std::vector<int> mask_rows(int32_t b0) const {
        std::vector<int> r;
        for (int i = 0; i < c_.block; ++i)
            if (x_[static_cast<size_t>(b0 + i)] == c_.common.mask_id) r.push_back(i);
        return r;
    }

    void denoise(int32_t b0, const std::vector<const float*>& logits, size_t vocab) {
        const std::vector<int> rows = mask_rows(b0);
        if (logits.size() != rows.size())
            throw std::invalid_argument("SDAR policy: a denoise pass takes one logit row per MASK row");
        const size_t n = rows.size();

        // x0 and its confidence per MASK row (generate.py:140-145), no shift: row i's
        // logits predict row i's token.
        //
        // The confidence is the drawn token's probability AFTER top-k / top-p filtering
        // and renormalising, as generate.py:41-44 computes it; sample()'s prob_out is that.
        // The wrapper (huggingface_bd3.py:75,90) and JetEngine read it BEFORE the filter
        // (after temperature); lmdeploy reads the softmax of the logits it is handed.
        // generate.py has no T=0 path (it divides by the temperature, generate.py:35-36),
        // and its top_k=1 makes every confidence 1.0, so dynamic decoding fills a whole
        // block per pass. The published greedy numbers come from the wrapper at T=1,
        // top_k=1, whose confidence is softmax(logits)[argmax] -- which is what sample()
        // returns at temperature 0. So T=0 here is that greedy, and T=1/top_k=1 is
        // generate.py's own.
        std::vector<int32_t> x0(n);
        std::vector<double> conf(n);
        for (size_t j = 0; j < n; ++j) x0[j] = sample(logits[j], vocab, c_.common.sampling, rng_, &conf[j]);

        // The step's minimum, clamped to the MASKs left. generate.py's top-k takes
        // sched[step] rows even when fewer MASKs are left, so it also picks rows whose
        // confidence is -inf -- the prompt's tail or tokens already filled -- and
        // generate.py:196 overwrites them with their x0. The wrapper guards this by
        // replacing x0 with the current token on non-MASK rows (huggingface_bd3.py:233);
        // clamping k selects exactly the same rows. It can only bite when block/steps > 1.
        const size_t k = std::min(static_cast<size_t>(sched_[static_cast<size_t>(step_)]), n);

        std::vector<char> sel(n, 0);
        size_t high = 0;
        if (c_.dynamic) {
            // low_confidence_dynamic (generate.py:168-179): every MASK row strictly over tau,
            // if there are at least k of them.
            for (size_t j = 0; j < n; ++j)
                if (conf[j] > c_.tau) { sel[j] = 1; ++high; }
        }
        if (!c_.dynamic || high < k) {
            // low_confidence_static (generate.py:160-166), or dynamic's fallback: the k most
            // confident MASK rows. Ties go to the lower row; torch.topk leaves their order
            // unspecified, and the fixture's generator refuses a tie at the cut.
            std::fill(sel.begin(), sel.end(), 0);
            std::vector<size_t> order(n);
            std::iota(order.begin(), order.end(), size_t{0});
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                              [&](size_t a, size_t b) { return conf[a] > conf[b] || (conf[a] == conf[b] && a < b); });
            for (size_t j = 0; j < k; ++j) sel[order[j]] = 1;
        }
        for (size_t j = 0; j < n; ++j)
            if (sel[j]) x_[static_cast<size_t>(b0 + rows[j])] = x0[j];

        ++step_;
        keep_ = b0;  // store_kv=False: this pass's K/V goes
        ++decode_passes_;

        // Every step fills at least min(sched[step], MASKs left) and the schedule sums to
        // the block, so the block is MASK-free after `steps` passes -- unless the model
        // sampled the MASK token itself, which leaves the row masked. generate.py then
        // indexes past its schedule (generate.py:119,165,174); JetEngine ends the block
        // after `steps` passes regardless, and so does this.
        if (mask_rows(b0).empty() || step_ >= c_.steps) {
            finish_block(b0);
            phase_ = Phase::Commit;
        }
    }

    /// The block's tokens are final once it has no MASK: hand the generated ones to take()
    /// now, before the commit pass, cut after the first stop id. generate.py checks only
    /// positions >= P (generate.py:199), so a stop id in the prompt's tail -- the chat
    /// template's <|im_end|> -- does not end the request. generate.py itself returns the
    /// whole block, tokens after the stop included; the wrapper cuts the text before the
    /// stop word; take()'s contract keeps the stop and nothing after it. A row left masked
    /// (above) is dropped, as generate.py's caller and the wrapper strip <|MASK|> from the text.
    void finish_block(int32_t b0) {
        const std::vector<int32_t>& stops = c_.common.stop_ids;
        for (int32_t q = std::max(b0, p_); q < b0 + c_.block; ++q) {
            const int32_t t = x_[static_cast<size_t>(q)];
            if (t == c_.common.mask_id) continue;
            out_.push_back(t);
            if (std::find(stops.begin(), stops.end(), t) != stops.end()) {
                stopped_ = true;
                break;
            }
        }
    }

    SDARConfig c_;
    std::vector<int> sched_;
    std::vector<int32_t> x_;  ///< the whole sequence, prompt then MASKs to the last block's end
    int32_t p_ = 0;
    int32_t nblocks_ = 0;
    int32_t prefill_blocks_ = 0;
    Phase phase_ = Phase::Done;
    int32_t blk_ = 0;  ///< prefill: the next prompt block; decode: the block being denoised
    int step_ = 0;
    int32_t keep_ = 0;
    std::vector<int32_t> out_;
    bool stopped_ = false;
    int decode_passes_ = 0;
    std::mt19937_64 rng_;
};

}  // namespace

std::unique_ptr<Policy> make_sdar(const SDARConfig& c) { return std::make_unique<SDARPolicy>(c); }

}  // namespace open_dllm
