// Traces: OPEN-DLLM-LLADA2 (plan: specs/open-engine/plans/diffusion-lm.md)
/// \file llada2_test.cpp
/// \brief Replays specs/open-engine/tests/fixtures/dllm_llada2.json against make_llada2().
///
/// The fixture is LLaDA2.2's own decode loop (modeling_llada2_moe.py, run by
/// utilities/dllm-ref/ref_llada2.py against a scripted model), turned into the passes a
/// cached backend runs. For every pass this checks, exactly: the rows (token, pos, slot),
/// the attention mode and the logit rows the policy asks for; then feeds the recorded
/// logits and checks keep() and take(). At the end: done(), the streamed output against
/// generate()'s return value, and decode_passes(). The first mismatch exits non-zero.
///
///   llada2_test.exe [fixture.json]

#include "open_dllm/policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

using json = nlohmann::json;
using namespace open_dllm;

namespace {

template <class T>
std::string fmt(const std::vector<T>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

std::string fmt(const Row& r) {
    return "(token " + std::to_string(r.token) + ", pos " + std::to_string(r.pos) + ", slot " +
           std::to_string(r.slot) + ")";
}

[[noreturn]] void fail(const std::string& where, const std::string& what) {
    std::fprintf(stderr, "FAIL %s: %s\n", where.c_str(), what.c_str());
    std::exit(1);
}

LLaDA2Config config_of(const json& j) {
    LLaDA2Config c;
    c.block = j.at("block");
    c.steps = j.at("steps");
    c.threshold = j.at("threshold");
    c.editing_threshold = j.at("editing_threshold");
    c.max_post_steps = j.at("max_post_steps");
    c.max_steps_per_block = j.at("max_steps_per_block");
    c.delete_id = j.at("delete_id");
    c.insert_id = j.at("insert_id");
    c.eos_early_stop = j.at("eos_early_stop");
    c.common.mask_id = j.at("mask_id");
    c.common.stop_ids = j.at("stop_ids").get<std::vector<int32_t>>();
    c.common.max_new_tokens = j.at("max_new_tokens");
    c.common.sampling.temperature = j.at("temperature");
    c.common.sampling.top_k = j.at("top_k");
    c.common.sampling.top_p = j.at("top_p");
    c.common.sampling.seed = j.at("seed");
    return c;
}

size_t run(const json& sc, size_t vocab) {
    const std::string name = sc.at("name");
    auto pol = make_llada2(config_of(sc.at("config")));
    pol->start(sc.at("prompt").get<std::vector<int32_t>>());
    std::vector<int32_t> streamed;
    const json& passes = sc.at("passes");
    std::vector<std::vector<float>> rows;
    std::vector<const float*> ptrs;
    for (size_t n = 0; n < passes.size(); ++n) {
        const json& e = passes[n];
        const std::string at = name + " pass " + std::to_string(n) + " (" + e.at("kind").get<std::string>() + ")";
        if (pol->done()) fail(at, "the policy is done; the reference runs this pass");
        const Pass p = pol->next();

        const json& er = e.at("rows");
        if (p.rows.size() != er.size())
            fail(at, "rows: got " + std::to_string(p.rows.size()) + ", expected " + std::to_string(er.size()));
        for (size_t i = 0; i < er.size(); ++i) {
            const Row want{er[i][0], er[i][1], er[i][2]};
            const Row& got = p.rows[i];
            if (got.token != want.token || got.pos != want.pos || got.slot != want.slot)
                fail(at, "row " + std::to_string(i) + ": got " + fmt(got) + ", expected " + fmt(want));
        }
        const Attn want_attn = e.at("attn") == "block" ? Attn::Block : Attn::Causal;
        if (p.attn != want_attn) fail(at, "attention mode differs");
        const auto want_lr = e.at("logit_rows").get<std::vector<int>>();
        if (p.logit_rows != want_lr) fail(at, "logit_rows: got " + fmt(p.logit_rows) + ", expected " + fmt(want_lr));

        const json& el = e.at("logits");
        rows.assign(el.size(), {});
        ptrs.clear();
        for (size_t j = 0; j < el.size(); ++j) {
            rows[j] = el[j].get<std::vector<float>>();
            if (rows[j].size() != vocab) fail(at, "fixture logit row " + std::to_string(j) + " is not vocab-wide");
            ptrs.push_back(rows[j].data());
        }
        pol->feed(ptrs, vocab);

        const int32_t want_keep = e.at("keep");
        if (pol->keep() != want_keep)
            fail(at, "keep(): got " + std::to_string(pol->keep()) + ", expected " + std::to_string(want_keep));
        const auto got_take = pol->take();
        const auto want_take = e.at("take").get<std::vector<int32_t>>();
        if (got_take != want_take) fail(at, "take(): got " + fmt(got_take) + ", expected " + fmt(want_take));
        streamed.insert(streamed.end(), got_take.begin(), got_take.end());
    }
    if (!pol->done()) fail(name, "the policy wants another pass; the reference stopped");
    const auto want_out = sc.at("output").get<std::vector<int32_t>>();
    if (streamed != want_out) fail(name, "output: got " + fmt(streamed) + ", expected " + fmt(want_out));
    const int want_dp = sc.at("decode_passes");
    if (pol->decode_passes() != want_dp)
        fail(name, "decode_passes(): got " + std::to_string(pol->decode_passes()) + ", expected " + std::to_string(want_dp));
    return passes.size();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "../../specs/open-engine/tests/fixtures/dllm_llada2.json";
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "llada2_test: cannot open %s\n", path.c_str());
        return 2;
    }
    json doc;
    try {
        doc = json::parse(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llada2_test: %s: %s\n", path.c_str(), e.what());
        return 2;
    }
    const size_t vocab = doc.at("vocab");
    size_t total = 0, n = 0;
    for (const json& sc : doc.at("scenarios")) {
        size_t k = 0;
        try {
            k = run(sc, vocab);
        } catch (const std::exception& e) {
            fail(sc.at("name").get<std::string>(), std::string("exception: ") + e.what());
        }
        std::printf("ok   %-28s %3zu passes, %2zu tokens\n", sc.at("name").get<std::string>().c_str(), k,
                    sc.at("output").size());
        total += k;
        ++n;
    }
    std::printf("llada2_test: %zu scenarios, %zu passes, every pass matches the reference\n", n, total);
    return 0;
}
