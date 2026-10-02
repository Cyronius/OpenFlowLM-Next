// Traces: OPEN-DLLM-WEDLM (canonical spec: specs/open-engine/spec.md; plan: .claude/plans/diffusion-lm-support.md)
/// \file wedlm_test.cpp
/// \brief Replays the WeDLM reference's recorded passes against make_wedlm().
///
/// The fixture (specs/open-engine/tests/fixtures/dllm_wedlm.json) comes from
/// utilities/dllm-ref/ref_wedlm.py, which runs tencent/WeDLM's own engine code with a
/// scripted model. For every pass this checks that the policy asks for the same rows
/// (token, RoPE position, cache slot), the same attention and the same logit rows; feeds
/// it the logits the reference saw; and checks keep() and take() after it. Then the whole
/// output and the pass count. The first mismatch stops the run with a non-zero exit.
/// No model, no hardware: `wedlm_test <fixtures/dllm_wedlm.json>`.
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_dllm/policy.hpp"

using nlohmann::json;
using namespace open_dllm;

namespace {

struct Mismatch {
    std::string what;
};

template <class T>
std::string str(const std::vector<T>& v) {
    std::ostringstream s;
    s << '[';
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    s << ']';
    return s.str();
}

std::string str(const Row& r) {
    return "{token " + std::to_string(r.token) + ", pos " + std::to_string(r.pos) + ", slot " +
           std::to_string(r.slot) + "}";
}

WeDLMConfig config_of(const json& j) {
    WeDLMConfig c;
    c.window = j.at("window").get<int>();
    // null is the reference's tau = None: one fill per pass. Nothing is under -inf.
    c.tau = j.at("tau").is_null() ? -std::numeric_limits<double>::infinity() : j.at("tau").get<double>();
    c.lambda = j.at("lambda").get<double>();
    c.common.mask_id = j.at("mask_id").get<int32_t>();
    c.common.stop_ids = j.at("stop_ids").get<std::vector<int32_t>>();
    c.common.max_new_tokens = j.at("max_new_tokens").get<int>();
    c.common.sampling.temperature = j.at("temperature").get<float>();
    c.common.sampling.top_k = j.at("top_k").get<int>();
    c.common.sampling.top_p = j.at("top_p").get<float>();
    c.common.sampling.seed = j.at("seed").get<uint64_t>();
    return c;
}

/// Runs one scenario; throws Mismatch at the first difference. Returns the passes checked.
int replay(const json& sc, size_t vocab) {
    auto policy = make_wedlm(config_of(sc.at("config")));
    policy->start(sc.at("prompt").get<std::vector<int32_t>>());
    const json& passes = sc.at("passes");
    std::vector<int32_t> output;
    for (size_t i = 0; i < passes.size(); ++i) {
        const json& want = passes[i];
        const std::string at = "pass " + std::to_string(i) + ": ";
        if (policy->done()) throw Mismatch{at + "policy is done, the reference ran this pass"};
        const Pass p = policy->next();

        const json& rows = want.at("rows");
        if (p.rows.size() != rows.size())
            throw Mismatch{at + std::to_string(p.rows.size()) + " rows, expected " + std::to_string(rows.size())};
        for (size_t r = 0; r < rows.size(); ++r) {
            const Row e{rows[r][0].get<int32_t>(), rows[r][1].get<int32_t>(), rows[r][2].get<int32_t>()};
            const Row& g = p.rows[r];
            if (g.token != e.token || g.pos != e.pos || g.slot != e.slot)
                throw Mismatch{at + "row " + std::to_string(r) + " is " + str(g) + ", expected " + str(e)};
        }
        if (want.at("attn").get<std::string>() != "causal" || p.attn != Attn::Causal)
            throw Mismatch{at + "attention is not causal"};
        const auto want_rows = want.at("logit_rows").get<std::vector<int>>();
        if (p.logit_rows != want_rows)
            throw Mismatch{at + "logit_rows " + str(p.logit_rows) + ", expected " + str(want_rows)};

        // The reference's logits for those rows, in that order.
        const json& lj = want.at("logits");
        if (lj.size() != want_rows.size()) throw Mismatch{at + "fixture: logits do not match logit_rows"};
        std::vector<std::vector<float>> store(lj.size());
        std::vector<const float*> ptrs(lj.size());
        for (size_t r = 0; r < lj.size(); ++r) {
            store[r] = lj[r].get<std::vector<float>>();
            if (store[r].size() != vocab) throw Mismatch{at + "fixture: a logits row is not vocab wide"};
            ptrs[r] = store[r].data();
        }
        policy->feed(ptrs, vocab);

        const int32_t want_keep = want.at("keep").get<int32_t>();
        if (policy->keep() != want_keep)
            throw Mismatch{at + "keep() " + std::to_string(policy->keep()) + ", expected " + std::to_string(want_keep)};
        const auto taken = policy->take();
        const auto want_taken = want.at("taken").get<std::vector<int32_t>>();
        if (taken != want_taken) throw Mismatch{at + "take() " + str(taken) + ", expected " + str(want_taken)};
        output.insert(output.end(), taken.begin(), taken.end());
    }
    if (!policy->done()) throw Mismatch{"the reference finished after " + std::to_string(passes.size()) + " passes, the policy did not"};
    const auto want_out = sc.at("output").get<std::vector<int32_t>>();
    if (output != want_out) throw Mismatch{"output " + str(output) + ", expected " + str(want_out)};
    const int want_passes = sc.at("decode_passes").get<int>();
    if (policy->decode_passes() != want_passes)
        throw Mismatch{"decode_passes() " + std::to_string(policy->decode_passes()) + ", expected " +
                       std::to_string(want_passes)};
    if (!policy->take().empty()) throw Mismatch{"take() after done() is not empty"};
    return static_cast<int>(passes.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: wedlm_test <specs/open-engine/tests/fixtures/dllm_wedlm.json>\n");
        return 2;
    }
    std::ifstream f(argv[1]);
    if (!f) {
        std::fprintf(stderr, "FAIL  cannot read %s\n", argv[1]);
        return 2;
    }
    const json fx = json::parse(f);
    const size_t vocab = fx.at("vocab").get<size_t>();
    int scenarios = 0, passes = 0;
    for (const json& sc : fx.at("scenarios")) {
        const std::string name = sc.at("name").get<std::string>();
        try {
            const int n = replay(sc, vocab);
            std::printf("ok    %-16s %2d passes, %2zu tokens  (%s)\n", name.c_str(), n, sc.at("output").size(),
                        sc.at("what").get<std::string>().c_str());
            ++scenarios;
            passes += n;
        } catch (const Mismatch& m) {
            std::printf("FAIL  %s: %s\n", name.c_str(), m.what.c_str());
            return 1;
        } catch (const std::exception& e) {
            std::printf("FAIL  %s: threw %s\n", name.c_str(), e.what());
            return 1;
        }
    }
    std::printf("OPEN-DLLM-WEDLM: %d scenarios, %d passes match the reference\n", scenarios, passes);
    return 0;
}
