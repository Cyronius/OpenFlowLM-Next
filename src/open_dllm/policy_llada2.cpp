/// \file policy_llada2.cpp
/// \brief LLaDA2.2 block diffusion with Levenshtein editing (OPEN-DLLM-LLADA2): the decode
///        loop of hf.co/inclusionAI/LLaDA2.2-mini's modeling_llada2_moe.py as a Policy.
///
/// The reference is LLaDA2MoeModelLM in modeling_llada2_moe.py (revision 038922ee), cited
/// below as M:N: generate() M:1649-1798, _joint_decode_block() M:1388-1646, the edit pass
/// M:1248-1307, the anti-loop resample M:1333-1386, sampling M:1309-1331.
/// utilities/dllm-ref/ref_llada2.py runs those functions, unchanged, against a scripted
/// model and records every pass; src/open_dllm/llada2_test.cpp replays that against this.
///
/// The sequence is cut into `block`-token blocks aligned to absolute position 0 (M:1716-1737).
/// Per request:
///
///   prefill   one Attn::Block pass per FULL prompt block, no logits, keep() = its end.
///             The reference has no separate prefill: it re-forwards those blocks every
///             step under the block-causal mask (M:1719-1729), which is the same K/V.
///   per block the prompt's tail (first decode block only) and MASKs.
///     denoise every row of the block, Attn::Block over the kept cache, logits for the
///             non-prompt rows. Each pass is one iteration of M:1455-1646: fill masks
///             (M2T), rewrite tokens (T2T), consume DELETE/INSERT. keep() = block start.
///     commit  the reference keeps no cache (M:1506 re-forwards x[:, :block_end] every
///             step); caching a finished block is exact because the mask is block-causal
///             and the MoE pool is chosen per aligned block (LLaDA2MoeGate.block_routing),
///             so a block's K/V depends only on itself and what precedes it -- provided it
///             was computed from the block's FINAL tokens. The normal exit (M:1640: the
///             step changed nothing and no mask is left) means the last denoise pass ran
///             on exactly those tokens, so that pass's keep() is the block end and nothing
///             more runs. Every other exit (the post-step budget M:1489, max_steps_per_block
///             M:1497, an empty block M:1445) leaves the cache holding stale K/V: keep() stays
///             at the block start and one more pass runs the final tokens, no logits,
///             keep() = block end. It is skipped when no block follows.
///
/// Slot == pos for every row. Tokens are streamed when their block ends, cut at
/// max_new_tokens and after the first stop id (M:1771-1782).
///
/// Randomness lives in three places only, all drawn from mt19937_64 seeded at start():
/// the M2T token when temperature > 0, the final round's M2T re-pick, and the resample on
/// a repeated block state. At temperature 0 the only draw is which changed position to
/// resample, so a greedy request that never repeats a state uses no randomness at all.

#include "open_dllm/policy.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace open_dllm {

namespace {

/// <|mask|> (hf tokenizer.json); used when Common::mask_id is unset.
constexpr int32_t kLLaDA2MaskId = 156895;
/// _resample_to_escape_loop's max_iters (M:1344).
constexpr int kResampleIters = 5;

class LLaDA2Policy final : public Policy {
public:
    explicit LLaDA2Policy(const LLaDA2Config& c) : c_(c) {
        if (c_.block < 1) throw std::invalid_argument("make_llada2: block must be >= 1");
        // steps is only the length of the per-step minimum schedule; 0 is legal (an empty
        // schedule, M:1240: every step then fills every mask), and steps > block just
        // schedules zero-token minimums (M:1242-1245).
        if (c_.steps < 0) throw std::invalid_argument("make_llada2: steps must be >= 0");
        if (c_.common.mask_id < 0) c_.common.mask_id = kLLaDA2MaskId;
    }

    void start(const std::vector<int32_t>& prompt) override {
        const int B = c_.block;
        prompt_ = prompt;
        L_ = static_cast<int32_t>(prompt.size());
        gen_ = std::max(0, c_.common.max_new_tokens);
        num_blocks_ = (L_ + gen_ + B - 1) / B;  // M:1716
        prefill_blocks_ = L_ / B;               // M:1735: the tail rides with the masks
        rng_.seed(c_.common.sampling.seed);
        out_.clear();
        stopped_ = false;
        keep_ = 0;
        decode_passes_ = 0;
        prefill_next_ = 0;
        if (prefill_blocks_ >= num_blocks_)
            phase_ = Phase::Done;  // nothing to generate (M:1737's range is empty)
        else if (prefill_blocks_ > 0)
            phase_ = Phase::Prefill;
        else
            begin_block(0);
    }

    bool done() const override { return phase_ == Phase::Done; }

    Pass next() override {
        Pass p;
        p.attn = Attn::Block;
        switch (phase_) {
        case Phase::Prefill: {
            const int32_t b0 = prefill_next_ * c_.block;
            for (int32_t i = 0; i < c_.block; ++i) p.rows.push_back({prompt_[b0 + i], b0 + i, b0 + i});
            break;
        }
        case Phase::Denoise:
        case Phase::Commit:
            for (int32_t i = 0; i < c_.block; ++i) p.rows.push_back({blk_[i], bs_ + i, bs_ + i});
            if (phase_ == Phase::Denoise) p.logit_rows = logit_rows_;
            break;
        case Phase::Done:
            throw std::logic_error("open_dllm::LLaDA2: next() after done()");
        }
        return p;
    }

    void feed(const std::vector<const float*>& logits, size_t vocab) override {
        switch (phase_) {
        case Phase::Prefill:
            if (!logits.empty()) throw std::invalid_argument("open_dllm::LLaDA2: prefill takes no logits");
            keep_ = (prefill_next_ + 1) * c_.block;
            if (++prefill_next_ == prefill_blocks_) begin_block(prefill_blocks_);
            break;
        case Phase::Denoise:
            if (logits.size() != logit_rows_.size())
                throw std::invalid_argument("open_dllm::LLaDA2: feed() needs one logit row per logit_rows entry");
            ++decode_passes_;
            step(logits, vocab);
            break;
        case Phase::Commit:
            if (!logits.empty()) throw std::invalid_argument("open_dllm::LLaDA2: a commit pass takes no logits");
            ++decode_passes_;
            keep_ = bs_ + c_.block;
            begin_block(block_ + 1);
            break;
        case Phase::Done:
            throw std::logic_error("open_dllm::LLaDA2: feed() after done()");
        }
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

    /// Set up block b as _joint_decode_block's preamble does (M:1427-1453), then run its
    /// first loop head. A block that ends before any forward goes straight to end_block().
    void begin_block(int32_t b) {
        if (b >= num_blocks_) {
            phase_ = Phase::Done;
            return;
        }
        const int B = c_.block;
        const int32_t mask = c_.common.mask_id;
        block_ = b;
        bs_ = b * B;
        blk_.assign(static_cast<size_t>(B), mask);
        for (int32_t i = 0; i < B && bs_ + i < L_; ++i) blk_[i] = prompt_[bs_ + i];
        // prompt_mask_block (M:1427): every non-mask row at entry is prompt. It never moves:
        // the prompt is a prefix of the block and edits only touch rows after it.
        prompt_row_.assign(static_cast<size_t>(B), 0);
        orig_.assign(static_cast<size_t>(B), 0);
        logit_rows_.clear();
        int initial = 0;
        for (int i = 0; i < B; ++i) {
            prompt_row_[i] = blk_[i] != mask;
            orig_[i] = blk_[i] == mask;  // is_original_mask (M:1443)
            if (orig_[i]) {
                ++initial;
                logit_rows_.push_back(i);  // only these rows' logits are ever read
            }
        }
        seen_.clear();
        post_steps_ = 0;
        step_id_ = 0;
        forwarded_.clear();
        if (initial == 0) {  // M:1445: a block with nothing to decode
            end_block();
            return;
        }
        // _get_num_transfer_tokens(initial, steps) (M:1239-1246): spread the block's
        // ORIGINAL mask count, not the block length, over `steps`; remainder first.
        schedule_.assign(static_cast<size_t>(c_.steps), c_.steps ? initial / c_.steps : 0);
        for (int i = 0; c_.steps && i < initial % c_.steps; ++i) ++schedule_[i];
        if (loop_head()) {
            end_block();
            return;
        }
        phase_ = Phase::Denoise;
    }

    /// The loop head (M:1456-1503) for the step about to run: this step's mask rows, the
    /// masks an edit created, the post-step count, and whether the loop breaks here --
    /// before any forward, so the caller ends the block without a pass.
    bool loop_head() {
        const int32_t mask = c_.common.mask_id;
        masks_.assign(blk_.size(), 0);
        int n_masks = 0, orig_left = 0;
        for (size_t i = 0; i < blk_.size(); ++i) {
            masks_[i] = blk_[i] == mask && !prompt_row_[i];
            n_masks += masks_[i];
            orig_left += orig_[i] && blk_[i] == mask;
        }
        new_masks_ = n_masks - orig_left;
        // Post-steps count from the first head with no ORIGINAL mask left (M:1469-1472);
        // a mask an edit opened does not hold them back.
        post_steps_ = orig_left == 0 ? post_steps_ + 1 : 0;
        // Both breaks, in the reference's order. The first wants zero masks of either kind
        // (M:1484-1489); the final round below guarantees that one step after it runs.
        if (n_masks == 0 && post_steps_ > c_.max_post_steps) return true;
        return step_id_ >= c_.max_steps_per_block;
    }

    /// One iteration of the loop body (M:1505-1646) on the logits of the pass that just ran.
    void step(const std::vector<const float*>& logits, size_t vocab) {
        const int B = c_.block;
        const int32_t mask = c_.common.mask_id;
        const Sampling& smp = c_.common.sampling;
        const bool sampled = smp.temperature > 0.f;  // M:1515: temperature != 0.0

        // ---- 2. per-row token + confidence (M:1514-1518, _diffusion_sample M:1309-1331)
        // x_s/p_s is what M2T writes: drawn under temperature/top-k/top-p, but scored by the
        // UNFILTERED softmax at temperature 1. x0/p0 is the greedy token and its probability,
        // what T2T writes and gates on. At temperature 0 they are the same thing. Only mask
        // rows ever write x_s, so only they draw.
        std::vector<const float*> row(static_cast<size_t>(B), nullptr);
        for (size_t j = 0; j < logits.size(); ++j) row[logit_rows_[j]] = logits[j];
        std::vector<int32_t> xs(static_cast<size_t>(B), mask), x0(static_cast<size_t>(B), mask);
        std::vector<double> ps(static_cast<size_t>(B), 0.0), p0(static_cast<size_t>(B), 0.0);
        for (int i : logit_rows_) {
            softmax(row[i], vocab, probs_);
            x0[i] = static_cast<int32_t>(argmax(row[i], vocab));
            p0[i] = probs_[x0[i]];
            xs[i] = x0[i];
            ps[i] = p0[i];
            if (sampled && masks_[i]) {
                xs[i] = sample(row[i], vocab, smp, rng_);
                ps[i] = probs_[xs[i]];
            }
        }

        // ---- 3. M2T (M:1526-1540): every mask over `threshold` if that meets this step's
        // minimum (schedule + the masks edits opened), else the `need` most confident.
        std::vector<char> m2t(static_cast<size_t>(B), 0);
        int n_masks = 0;
        for (int i = 0; i < B; ++i) n_masks += masks_[i];
        if (n_masks > 0) {
            if (step_id_ < static_cast<int>(schedule_.size())) {
                const int need = schedule_[step_id_] + new_masks_;
                int high = 0;
                for (int i = 0; i < B; ++i) high += masks_[i] && ps[i] > c_.threshold;
                if (high >= need) {
                    for (int i = 0; i < B; ++i) m2t[i] = masks_[i] && ps[i] > c_.threshold;
                } else {
                    // torch.topk over the masked confidences; k never exceeds the mask count,
                    // so non-masks (-inf there) are never picked. Ties go to the lower row
                    // (torch leaves tie order unspecified).
                    std::vector<int> order;
                    for (int i = 0; i < B; ++i)
                        if (masks_[i]) order.push_back(i);
                    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ps[a] > ps[b]; });
                    const int k = std::min(need, n_masks);
                    for (int r = 0; r < k; ++r) m2t[order[r]] = 1;
                }
            } else {
                m2t = masks_;  // past the schedule: fill every mask
            }
        }

        // ---- 4. T2T (M:1543-1547): rewrite a generated token whose greedy prediction differs,
        // when that prediction clears editing_threshold. Prompt rows are never edited.
        std::vector<char> t2t(static_cast<size_t>(B), 0);
        for (int i = 0; i < B; ++i)
            t2t[i] = !masks_[i] && !prompt_row_[i] && p0[i] > c_.editing_threshold && blk_[i] != x0[i];

        // ---- final round (M:1560-1564): the step after which the block must end. With a
        // post-step budget it is the step that spends it; with none, it is the step whose
        // M2T covers every original mask still open.
        bool final_round;
        if (c_.max_post_steps > 0) {
            final_round = post_steps_ >= c_.max_post_steps;
        } else {
            bool any = false, all = true;
            for (int i = 0; i < B; ++i)
                if (masks_[i] && orig_[i]) {
                    any = true;
                    all = all && m2t[i];
                }
            final_round = any && all;
        }

        // ---- 5. on the final round fill every mask, and re-pick any DELETE/INSERT about to
        // be written with both at -inf (M:1570-1598), so no edit can reopen a mask. M2T
        // positions re-draw under the request's sampling, T2T positions greedily.
        if (final_round) {
            m2t = masks_;
            const Sampling greedy{};
            for (int i = 0; i < B; ++i) {
                if (m2t[i] && is_edit(xs[i])) xs[i] = redraw(row[i], vocab, smp, c_.delete_id, c_.insert_id);
            }
            for (int i = 0; i < B; ++i) {
                if (t2t[i] && is_edit(x0[i])) x0[i] = redraw(row[i], vocab, greedy, c_.delete_id, c_.insert_id);
            }
        }

        // ---- 6. writes (M:1601-1606), then the anti-loop resample (M:1610-1621)
        const std::vector<int32_t> old = blk_;
        const std::vector<char> snapshot = orig_;
        bool any_write = false;
        for (int i = 0; i < B; ++i) any_write = any_write || m2t[i] || t2t[i];
        if (any_write) {
            for (int i = 0; i < B; ++i)
                if (m2t[i]) blk_[i] = xs[i];
            for (int i = 0; i < B; ++i)
                if (t2t[i]) blk_[i] = x0[i];
            if (!final_round) resample(old, row, vocab, m2t, t2t);
        }
        // Pre-edit states, every step, whether or not anything was written (M:1623).
        seen_.insert(blk_);

        // ---- 7. DELETE / INSERT (M:1626-1637)
        apply_edits(old, snapshot);
        forwarded_ = old;
        ++step_id_;
        keep_ = bs_;

        // ---- 8. the normal exit (M:1640-1644): the step changed nothing and no mask is left,
        // so the pass that just ran saw the final tokens and its K/V can stay.
        if (blk_ == old && std::find(blk_.begin(), blk_.end(), mask) == blk_.end()) {
            end_block();
            return;
        }
        if (loop_head()) end_block();
    }

    /// _resample_to_escape_loop (M:1333-1386). If the pre-edit state was seen before in this
    /// block, re-draw one randomly chosen changed position (its current token at -inf) until
    /// the state is new, at most kResampleIters times.
    void resample(const std::vector<int32_t>& old, const std::vector<const float*>& row, size_t vocab,
                  const std::vector<char>& m2t, const std::vector<char>& t2t) {
        if (!seen_.count(blk_)) return;
        const Sampling greedy{};
        std::vector<int> changed;
        for (int it = 0; it < kResampleIters; ++it) {
            changed.clear();
            for (size_t i = 0; i < blk_.size(); ++i)
                if (blk_[i] != old[i]) changed.push_back(static_cast<int>(i));
            if (changed.empty()) break;
            std::uniform_int_distribution<size_t> pick(0, changed.size() - 1);
            const int pos = changed[pick(rng_)];
            if (m2t[pos])
                blk_[pos] = redraw(row[pos], vocab, c_.common.sampling, blk_[pos]);
            else if (t2t[pos])
                blk_[pos] = redraw(row[pos], vocab, greedy, blk_[pos]);
            else
                continue;  // unreachable: only M2T/T2T rows change (kept as the reference has it)
            if (!seen_.count(blk_)) break;
        }
    }

    /// _apply_edit_operations_with_tracking (M:1248-1307). DELETE drops its row; INSERT at
    /// row i becomes [MASK, old[i]] -- a new mask in front of the token the step overwrote
    /// (MASK again when an M2T row was the one written). The block is then cut or right-
    /// padded with masks back to `block`. Original-mask tracking travels with each token;
    /// the inserted and padded masks are not original.
    void apply_edits(const std::vector<int32_t>& old, const std::vector<char>& snapshot) {
        const int32_t mask = c_.common.mask_id;
        const size_t B = blk_.size();
        std::vector<int32_t> tok;
        std::vector<char> trk;
        tok.reserve(B + B);
        trk.reserve(B + B);
        for (size_t i = 0; i < B; ++i) {
            if (blk_[i] == c_.delete_id) continue;
            if (blk_[i] == c_.insert_id) {
                tok.push_back(mask);
                trk.push_back(0);
                tok.push_back(old[i]);
                trk.push_back(snapshot[i]);
            } else {
                tok.push_back(blk_[i]);
                trk.push_back(snapshot[i]);
            }
        }
        tok.resize(B, mask);
        trk.resize(B, 0);
        blk_.swap(tok);
        orig_.swap(trk);
    }

    /// The block is final: stream its generated tokens, then decide what runs next.
    void end_block() {
        const int32_t be = bs_ + c_.block;
        const int32_t first = std::max(L_, bs_);
        // generate()'s output is x[L : L + gen] cut after the first eos (M:1771-1782).
        for (int32_t p = first; p < std::min(be, L_ + gen_) && !stopped_; ++p) {
            out_.push_back(blk_[p - bs_]);
            stopped_ = is_stop(blk_[p - bs_]);
        }
        // eos_early_stop (M:1764-1769) looks at every generated token up to the block end,
        // past max_new_tokens included; an earlier block's stop would already have ended it.
        bool stop_here = false;
        for (int32_t p = first; p < be && c_.eos_early_stop; ++p) stop_here = stop_here || is_stop(blk_[p - bs_]);
        const bool exact = !forwarded_.empty() && forwarded_ == blk_;
        keep_ = exact ? be : bs_;
        if (block_ + 1 >= num_blocks_ || stop_here)
            phase_ = Phase::Done;  // no later block reads this K/V: no commit pass
        else if (exact)
            begin_block(block_ + 1);
        else
            phase_ = Phase::Commit;
    }

    /// The row's logits with `ban_a` / `ban_b` at -inf, drawn under `s` (greedy at T=0).
    int32_t redraw(const float* row, size_t vocab, const Sampling& s, int32_t ban_a, int32_t ban_b = -1) {
        scratch_.assign(row, row + vocab);
        for (int32_t b : {ban_a, ban_b})
            if (b >= 0 && static_cast<size_t>(b) < vocab) scratch_[b] = -std::numeric_limits<float>::infinity();
        return sample(scratch_.data(), vocab, s, rng_);
    }

    bool is_edit(int32_t t) const { return t == c_.delete_id || t == c_.insert_id; }

    bool is_stop(int32_t t) const {
        const auto& s = c_.common.stop_ids;
        return std::find(s.begin(), s.end(), t) != s.end();
    }

    LLaDA2Config c_;
    std::mt19937_64 rng_;
    std::vector<int32_t> prompt_;
    int32_t L_ = 0, gen_ = 0, num_blocks_ = 0, prefill_blocks_ = 0;
    Phase phase_ = Phase::Done;
    int32_t prefill_next_ = 0;  ///< next full prompt block to prefill
    int32_t block_ = 0;         ///< the block being decoded
    int32_t bs_ = 0;            ///< its first position (and slot)
    std::vector<int32_t> blk_;  ///< x[0, bs : bs + block]
    std::vector<char> prompt_row_, orig_, masks_;
    std::vector<int> logit_rows_, schedule_;
    std::set<std::vector<int32_t>> seen_;  ///< seen_block_results (M:1451)
    int post_steps_ = 0, step_id_ = 0, new_masks_ = 0;
    std::vector<int32_t> forwarded_;  ///< the tokens the last denoise pass of this block ran
    std::vector<int32_t> out_;
    bool stopped_ = false;
    int32_t keep_ = 0;
    int decode_passes_ = 0;
    std::vector<double> probs_;
    std::vector<float> scratch_;
};

}  // namespace

std::unique_ptr<Policy> make_llada2(const LLaDA2Config& c) { return std::make_unique<LLaDA2Policy>(c); }

}  // namespace open_dllm
