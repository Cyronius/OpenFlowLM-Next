/// \file policy_wedlm.cpp
/// \brief WeDLM's streaming parallel decode as an open_dllm::Policy.
///
/// The reference is tencent/WeDLM's engine (wedlm/engine/wedlm_decoder.py and sampler.py,
/// with the stop handling in scheduler.py), not the paper's Algorithm 1 and not the
/// block-wise generate_wedlm() in the HF modeling file. Where the paper and the engine
/// differ, this follows the engine; the differences are noted where they matter.
///
/// One request, batch size 1. The state is the committed length `ctx` and a window of
/// slots at logical positions ctx, ctx+1, ...; each slot holds a token and a filled flag.
/// One decode pass:
///
///   1. refill: p = the window's leading filled slots; append p masks (W + p rows);
///   2. rows: filled slots in ascending slot order, then mask slots in ascending order.
///      A row's RoPE position is its logical one (ctx + slot index), its cache slot is
///      ctx + its physical index. Causal attention in physical order then lets every mask
///      see every filled token, while the leading filled slots -- physical rows 0..p-1 --
///      see exactly their true left context and land at their own logical positions;
///   3. after the forward, commit those p slots: their K/V is already in place. A stop
///      token or max_new_tokens ends the request inside the commit;
///   4. each remaining mask predicts its own position (no shift). Fill every mask whose
///      entropy plus lambda times its distance from the first mask is below tau, or the
///      single lowest one if none is.
///
/// Tokens filled in pass t are committed in pass t+1: the commit needs the K/V that only
/// a forward computes. So the request's last pass exists only to commit; its logits are
/// never read, and next() asks for none.
#include "open_dllm/policy.hpp"

#include <algorithm>
#include <stdexcept>

namespace open_dllm {

namespace {

/// WeDLM's [MASK]. Its config.json leaves mask_token_id null and the engine falls back
/// to this (model_runner.py:122-129); a caller that knows better sets Common::mask_id.
constexpr int32_t kDefaultMaskId = 151665;

class WeDLM final : public Policy {
public:
    explicit WeDLM(const WeDLMConfig& c) : c_(c) {
        if (c_.window < 1) throw std::invalid_argument("make_wedlm: window must be >= 1");
        if (c_.common.mask_id < 0) c_.common.mask_id = kDefaultMaskId;
    }

    void start(const std::vector<int32_t>& prompt) override {
        // The reference cannot take an empty prompt either (sequence.py:92 reads token_ids[-1]).
        if (prompt.empty()) throw std::invalid_argument("WeDLM: empty prompt");
        prompt_ = prompt;
        rng_.seed(c_.common.sampling.seed);
        phase_ = Phase::Prefill;
        finished_ = false;
        in_flight_ = false;
        ctx_ = 0;
        generated_ = 0;
        passes_ = 0;
        tok_.clear();
        filled_.clear();
        out_.clear();
    }

    bool done() const override { return finished_; }

    Pass next() override {
        if (finished_ || phase_ == Phase::Idle) throw std::logic_error("WeDLM::next: no request in progress");
        if (in_flight_) throw std::logic_error("WeDLM::next: the previous pass was not fed");
        in_flight_ = true;
        Pass p;
        p.attn = Attn::Causal;
        if (phase_ == Phase::Prefill) {
            // Plain causal prefill. Its logits are discarded (model_runner.py:517-522): the
            // first token comes from the first window pass, which predicts its own position.
            p.rows.reserve(prompt_.size());
            for (size_t i = 0; i < prompt_.size(); ++i) {
                const int32_t at = static_cast<int32_t>(i);
                p.rows.push_back({prompt_[i], at, at});
            }
            return p;
        }

        // 1. Refill. The reference does this at the top of the step that commits these slots
        //    (wedlm_decoder.py:167-189, called from prepare_decode_inputs), so a pass has
        //    W + p rows; the paper's Algorithm 1 refills after the commit and keeps W.
        prefix_ = leading_filled();
        tok_.insert(tok_.end(), prefix_, c_.common.mask_id);
        filled_.insert(filled_.end(), prefix_, 0);

        // 2. Topological reordering (wedlm_decoder.py:235-241, slots :157-163).
        order_.clear();
        for (size_t i = 0; i < tok_.size(); ++i)
            if (filled_[i]) order_.push_back(static_cast<int32_t>(i));
        n_filled_ = order_.size();
        for (size_t i = 0; i < tok_.size(); ++i)
            if (!filled_[i]) order_.push_back(static_cast<int32_t>(i));
        p.rows.reserve(order_.size());
        for (size_t j = 0; j < order_.size(); ++j)
            p.rows.push_back({tok_[order_[j]], ctx_ + order_[j], ctx_ + static_cast<int32_t>(j)});

        // The commit depends only on the state before the pass, so whether it ends the
        // request is known now. If it does, the reference never reads this pass's logits
        // (wedlm_decoder.py:452), and we don't ask the backend for them.
        commit_ = plan_commit();
        if (!commit_.finishes)
            for (size_t j = n_filled_; j < order_.size(); ++j) p.logit_rows.push_back(static_cast<int>(j));
        return p;
    }

    void feed(const std::vector<const float*>& logits, size_t vocab) override {
        if (!in_flight_) throw std::logic_error("WeDLM::feed: no pass in flight");
        in_flight_ = false;
        if (phase_ == Phase::Prefill) {
            if (!logits.empty()) throw std::invalid_argument("WeDLM::feed: prefill takes no logits");
            ctx_ = static_cast<int32_t>(prompt_.size());
            phase_ = Phase::Decode;
            // The first window: W masks (wedlm_decoder.py:118-126; the KV budget the engine
            // also caps it with is the backend's job, and never binds at 2W free rows).
            tok_.assign(static_cast<size_t>(c_.window), c_.common.mask_id);
            filled_.assign(static_cast<size_t>(c_.window), 0);
            return;
        }
        const size_t n_mask = commit_.finishes ? 0 : order_.size() - n_filled_;
        if (logits.size() != n_mask)
            throw std::invalid_argument("WeDLM::feed: expected logits for exactly next().logit_rows");
        if (n_mask > 0 && vocab == 0) throw std::invalid_argument("WeDLM::feed: vocab is 0");
        ++passes_;

        // 3. Commit the leading filled slots.
        out_.insert(out_.end(), tok_.begin(), tok_.begin() + commit_.emit);
        ctx_ += commit_.emit;
        generated_ += commit_.emit;
        if (commit_.finishes) {
            finished_ = true;
            tok_.clear();
            filled_.clear();
            return;
        }
        tok_.erase(tok_.begin(), tok_.begin() + prefix_);
        filled_.erase(filled_.begin(), filled_.begin() + prefix_);

        // 4. Fill. Mask rows came last in the pass, in ascending slot order, and the commit
        //    only removed filled slots, so logits[k] belongs to the k-th mask of the window.
        std::vector<int32_t> mask_slot;
        mask_slot.reserve(n_mask);
        for (size_t i = 0; i < tok_.size(); ++i)
            if (!filled_[i]) mask_slot.push_back(static_cast<int32_t>(i));

        // Adjusted entropy (sampler.py:198-212): the entropy of the raw logits -- no
        // temperature, top-k or top-p -- plus lambda per slot of distance from the first
        // remaining mask. Entropy measures confidence here, not the sampling distribution.
        // (The commit leaves a mask in slot 0, so the distance is also the slot index.)
        std::vector<double> adj(n_mask), probs;
        for (size_t k = 0; k < n_mask; ++k) {
            softmax(logits[k], vocab, probs);
            adj[k] = entropy(probs) + c_.lambda * static_cast<double>(mask_slot[k] - mask_slot[0]);
        }
        // Every mask under tau (sampler.py:215-219), else the first argmin (:222). The
        // reference's tau = None (one fill per pass) is any tau <= 0 here: with lambda >= 0,
        // which the reference enforces (sampling_params.py:74), neither term is negative.
        std::vector<size_t> pick;
        for (size_t k = 0; k < n_mask; ++k)
            if (adj[k] < c_.tau) pick.push_back(k);
        if (pick.empty()) {
            size_t best = 0;
            for (size_t k = 1; k < n_mask; ++k)
                if (adj[k] < adj[best]) best = k;
            pick.push_back(best);
        }
        // The reference samples every mask row and keeps the picked ones (sampler.py:261-272).
        // Which rows are picked never depends on the draw, so sampling only those is the same
        // distribution, and spares a vocab-wide sort per discarded row when temperature > 0.
        for (size_t k : pick) {
            tok_[mask_slot[k]] = sample(logits[k], vocab, c_.common.sampling, rng_, nullptr,
                                        Order::FilterFirst);  // sampler.py:156-165
            filled_[mask_slot[k]] = 1;
        }
    }

    int32_t keep() const override { return ctx_; }

    std::vector<int32_t> take() override {
        std::vector<int32_t> t;
        t.swap(out_);
        return t;
    }

    int decode_passes() const override { return passes_; }

private:
    enum class Phase { Idle, Prefill, Decode };

    /// What this pass's commit does: how many of the leading filled slots become output,
    /// and whether the request ends with them.
    struct Commit {
        int32_t emit = 0;
        bool finishes = false;
    };

    int32_t leading_filled() const {
        int32_t n = 0;
        while (n < static_cast<int32_t>(filled_.size()) && filled_[n]) ++n;
        return n;
    }

    bool is_stop(int32_t t) const {
        const auto& s = c_.common.stop_ids;
        return std::find(s.begin(), s.end(), t) != s.end();
    }

    /// The commit rule, token by token (wedlm_decoder.py:359-372): count it, then end on a
    /// stop token or when max_new_tokens have been generated, the ending token included.
    /// The decoder truncates its return only at a stop (:374-381) and the scheduler cuts at
    /// max_new_tokens again while appending (scheduler.py:134-151); together they end at
    /// whichever comes first, which is what this does. Like the reference, the check comes
    /// after the token is counted, so max_new_tokens <= 0 still yields one token.
    Commit plan_commit() const {
        Commit c;
        for (int32_t i = 0; i < prefix_; ++i) {
            ++c.emit;
            if (is_stop(tok_[i]) || generated_ + c.emit >= c_.common.max_new_tokens) {
                c.finishes = true;
                break;
            }
        }
        return c;
    }

    WeDLMConfig c_;
    std::vector<int32_t> prompt_;
    std::mt19937_64 rng_;
    Phase phase_ = Phase::Idle;
    bool finished_ = false;
    bool in_flight_ = false;   ///< next() returned a pass that feed() has not consumed
    int32_t ctx_ = 0;          ///< committed length: prompt plus committed tokens
    int generated_ = 0;        ///< committed tokens, what max_new_tokens counts
    int passes_ = 0;

    /// The window: slot i sits at logical position ctx_ + i. `filled_` is kept apart from
    /// the token because a filled slot may hold the mask id itself, if that is what the
    /// model predicted; the reference tracks flags (window_mask_flags) for the same reason.
    std::vector<int32_t> tok_;
    std::vector<char> filled_;

    // The pass in flight.
    int32_t prefix_ = 0;          ///< p: the leading filled slots it commits
    std::vector<int32_t> order_;  ///< window slot of each physical row
    size_t n_filled_ = 0;         ///< rows [0, n_filled_) are filled slots, the rest masks
    Commit commit_;

    std::vector<int32_t> out_;    ///< committed since the last take()
};

}  // namespace

std::unique_ptr<Policy> make_wedlm(const WeDLMConfig& c) { return std::make_unique<WeDLM>(c); }

}  // namespace open_dllm
