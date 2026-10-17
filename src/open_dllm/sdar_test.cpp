// Traces: OPEN-DLLM-SDAR (plan: specs/open-engine/plans/diffusion-lm.md)
// The SDAR policy against JetAstra/SDAR generate.py's own decode loop, run on a scripted
// model by utilities/dllm-ref/ref_sdar.py:
//   python utilities/dllm-ref/ref_sdar.py
//   sdar_test.exe specs/open-engine/tests/fixtures/dllm_sdar.json
// Every scenario is replayed pass by pass: the policy's rows (token, pos, slot), attention
// mode and logit rows must equal the reference's; it is fed the logits the reference saw
// for those rows; then keep() and take() must equal the reference's. At the end done(), the
// whole output and decode_passes() must match. The first mismatch ends the run, non-zero.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_dllm/policy.hpp"

using json = nlohmann::json;
using namespace open_dllm;

namespace {

struct Mismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <class T>
std::string show(const std::vector<T>& v) {
    std::ostringstream s;
    s << '[';
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    s << ']';
    return s.str();
}

template <class T>
void expect_eq(const T& want, const T& got, const std::string& what) {
    if (!(want == got)) {
        std::ostringstream s;
        s << what << ": reference " << want << ", policy " << got;
        throw Mismatch(s.str());
    }
}

template <class T>
void expect_eq(const std::vector<T>& want, const std::vector<T>& got, const std::string& what) {
    if (want != got) throw Mismatch(what + ": reference " + show(want) + ", policy " + show(got));
}

SDARConfig config_of(const json& c) {
    SDARConfig cfg;
    cfg.block = c.at("block").get<int>();
    cfg.steps = c.at("steps").get<int>();
    cfg.dynamic = c.at("dynamic").get<bool>();
    cfg.tau = c.at("tau").get<double>();
    cfg.common.mask_id = c.at("mask_id").get<int32_t>();
    cfg.common.stop_ids = c.at("stop_ids").get<std::vector<int32_t>>();
    cfg.common.max_new_tokens = c.at("max_new_tokens").get<int>();
    const json& s = c.at("sampling");
    cfg.common.sampling.temperature = s.at("temperature").get<float>();
    cfg.common.sampling.top_k = s.at("top_k").get<int>();
    cfg.common.sampling.top_p = s.at("top_p").get<float>();
    cfg.common.sampling.seed = s.at("seed").get<uint64_t>();
    return cfg;
}

/// Replays one scenario; returns the number of passes on success, throws Mismatch.
size_t replay(const json& sc, size_t vocab, float scale) {
    std::unique_ptr<Policy> pol = make_sdar(config_of(sc.at("config")));
    pol->start(sc.at("prompt").get<std::vector<int32_t>>());

    std::vector<int32_t> output;
    const json& passes = sc.at("passes");
    for (size_t i = 0; i < passes.size(); ++i) {
        const json& ep = passes[i];
        const std::string at = "pass " + std::to_string(i);
        if (pol->done()) throw Mismatch(at + ": the policy is done, the reference ran this pass");
        const Pass p = pol->next();

        const json& er = ep.at("rows");
        expect_eq(er.size(), p.rows.size(), at + " row count");
        for (size_t r = 0; r < er.size(); ++r) {
            const std::string row = at + " row " + std::to_string(r);
            expect_eq(er[r][0].get<int32_t>(), p.rows[r].token, row + " token");
            expect_eq(er[r][1].get<int32_t>(), p.rows[r].pos, row + " pos");
            expect_eq(er[r][2].get<int32_t>(), p.rows[r].slot, row + " slot");
        }
        const std::string want_attn = ep.at("attn").get<std::string>();
        expect_eq(want_attn, std::string(p.attn == Attn::Block ? "block" : "causal"), at + " attn");
        expect_eq(ep.at("logit_rows").get<std::vector<int>>(), p.logit_rows, at + " logit_rows");

        // the reference's logits for exactly those rows, integers in units of `scale` (exact in fp32)
        const json& el = ep.at("logits");
        std::vector<std::vector<float>> rows(el.size());
        std::vector<const float*> ptrs;
        for (size_t r = 0; r < el.size(); ++r) {
            if (el[r].size() != vocab) throw std::runtime_error(at + ": a logit row is not vocab wide");
            rows[r].reserve(vocab);
            for (const json& v : el[r]) rows[r].push_back(static_cast<float>(v.get<int>()) * scale);
            ptrs.push_back(rows[r].data());
        }
        pol->feed(ptrs, vocab);

        expect_eq(ep.at("keep").get<int32_t>(), pol->keep(), at + " keep()");
        const std::vector<int32_t> took = pol->take();
        expect_eq(ep.at("take").get<std::vector<int32_t>>(), took, at + " take()");
        output.insert(output.end(), took.begin(), took.end());
    }
    if (!pol->done()) throw Mismatch("after the reference's last pass the policy is not done");
    expect_eq(sc.at("output").get<std::vector<int32_t>>(), output, "output");
    expect_eq(sc.at("decode_passes").get<int>(), pol->decode_passes(), "decode_passes()");
    return passes.size();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: sdar_test <dllm_sdar.json>\n");
        return 2;
    }
    json fx;
    try {
        std::ifstream f(argv[1]);
        if (!f) throw std::runtime_error(std::string("cannot read ") + argv[1]);
        fx = json::parse(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL fixture: %s\n", e.what());
        return 2;
    }
    const size_t vocab = fx.at("vocab").get<size_t>();
    const float scale = fx.at("logit_scale").get<float>();
    size_t n_pass = 0, n_scen = 0;
    for (const json& sc : fx.at("scenarios")) {
        const std::string name = sc.at("name").get<std::string>();
        try {
            const size_t n = replay(sc, vocab, scale);
            std::printf("ok    %-30s %3zu passes, %3zu tokens out\n", name.c_str(), n, sc.at("output").size());
            n_pass += n;
            ++n_scen;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL  %s: %s\n", name.c_str(), e.what());
            return 1;
        }
    }
    std::printf("OPEN-DLLM-SDAR: %zu scenarios, %zu passes, all match generate.py\n", n_scen, n_pass);
    return 0;
}
