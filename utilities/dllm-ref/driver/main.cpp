/// \file main.cpp
/// \brief dllm-ref: runs the open_dllm decode policies (src/open_dllm) on llama.cpp, for
///        Phase 1 of .claude/plans/diffusion-lm-support.md -- quality, tokens per forward
///        and golden traces for WeDLM, SDAR and LLaDA2.2, with the same policy code the
///        open engine will run later.
///
///     dllm-ref --model m.gguf --policy wedlm|sdar|llada2|ar --in prompts.jsonl --out results.jsonl
///              [--trace DIR] [--ctx 8192] [--threads N] [--ngl 999] [--seed 0] [--max-new 512] ...
///     (dllm-ref --help lists every flag)
///
/// Prompts arrive as token ids and results leave as token ids: tokenisation and chat
/// templates are Python's job. The policy decides every pass; this file only runs them, in
/// the loop policy.hpp's header describes. llama.cpp keys its cache by position, not by
/// slot, which that contract allows: for every policy here a row's slot is >= keep() exactly
/// when its position is, so "drop every row at slot >= keep()" is
/// llama_memory_seq_rm(mem, 0, keep(), -1). Backend::trim checks that on every pass.
///
/// How a pass becomes llama.cpp calls (Backend::run):
///
///   Block                    one llama_batch, causal attention off: every row sees the kept
///                            cache and every row of the pass. The pass must be ONE ubatch
///                            (llama-context.cpp:1713 asserts it for non-causal batches).
///   Causal, slot == pos,     one llama_batch, causal on. Physical order is position order
///   consecutive positions    here, so llama.cpp's position-causal mask is the policy's
///                            physical-causal one. Every prompt prefill takes this route.
///   any other Causal pass    one llama_decode per row, in row order, causal off. Row j then
///   (WeDLM's reordered rows) sees the kept cache, rows 0..j-1 (already written) and itself:
///                            WeDLM's mask exactly. Its positions jump about the cache's
///                            last one, which llama-batch.cpp:295-319 refuses unless
///                            LLAMA_ALLOW_NONCONSECUTIVE_POS=1 (local patch 1).
///
/// Environment. The policy table says which local llama.cpp patches a policy needs, and the
/// driver sets their variables for its own process before llama.cpp loads anything:
/// LLAMA_ALLOW_NONCONSECUTIVE_POS=1 for wedlm, LLAMA_BLOCK_ROUTING=32:48 for llada2 (patch
/// 2: LLaDA2.2's per-32-token-block expert pool, top 48). A value already in the environment
/// wins, for A/B runs. Both patches read their variable once, on first use (a function-local
/// static in llama-batch.cpp / llama-graph.cpp), which is after main() has set it; one read
/// in a namespace-scope static initialiser would run before main() and miss it.
///
/// Outputs. --out gets one JSON line per prompt, flushed as it completes: the ids generated,
/// pass counts, timings and the settings in effect. --trace writes DIR/<id>.json per prompt:
/// every pass's rows, mode, logit rows, the top-8 logits of each logit row, keep() and the
/// tokens that became final -- the golden trace, and what hf_check.py holds against an HF
/// reference run (plan, Phase 1, "Checking the driver itself").
///
/// llama.cpp file:line citations are at bb4caa75, before the local patches.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "llama.h"
#include "nlohmann/json.hpp"
#include "open_dllm/policy.hpp"

namespace {

/// Insertion-ordered, so a result line reads id, policy, out_ids, ... as written.
using json = nlohmann::ordered_json;
using open_dllm::Attn;
using open_dllm::Pass;
using open_dllm::Policy;
using open_dllm::Row;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double round2(double v) { return std::round(v * 100.0) / 100.0; }

/// Every json::dump goes through here. A path from argv is in the ANSI code page on Windows,
/// not UTF-8, and the default (strict) dump throws on it; replace keeps the line.
std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

/// A float for JSON with the 9 significant digits that round-trip it. nlohmann prints the
/// double it widens to, which is 17 digits of which 8 are noise.
double jfloat(float v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v));
    return std::strtod(buf, nullptr);
}

// ---- the policies and their shipped settings -----------------------------------------

enum class Kind { WeDLM, SDAR, LLaDA2, AR };

/// What each policy runs with unless a flag says otherwise: the settings its model ships
/// with (plan, "The three models"), temperature 0 everywhere. Fields a policy does not use
/// are 0 / -1. ar's empty stop list means "the model's end-of-generation tokens", read from
/// the GGUF vocab once the model is loaded.
struct Preset {
    const char* name;
    Kind kind;
    int32_t mask_id;
    std::vector<int32_t> stop_ids;
    int window;              ///< wedlm W
    double lambda;           ///< wedlm distance penalty
    double tau;              ///< wedlm entropy threshold / sdar confidence threshold
    int block, steps;        ///< sdar, llada2
    double threshold;        ///< llada2 mask->token
    double edit_threshold;   ///< llada2 token->token
    int max_post_steps;      ///< llada2
    int32_t delete_id, insert_id;  ///< llada2 edit tokens
    bool nonconsecutive;     ///< needs LLAMA_ALLOW_NONCONSECUTIVE_POS=1 (patch 1)
    const char* block_routing;  ///< LLAMA_BLOCK_ROUTING value (patch 2), or nullptr
};

const Preset kPresets[] = {
    // name     kind          mask    stops              W  lambda tau  block steps thr  edit post delete  insert  noncons routing
    {"wedlm",  Kind::WeDLM,  151665, {151643, 151645}, 16, 0.02,  0.4,  0,    0,  0.0, 0.0,  0,     -1,     -1, true,   nullptr},
    {"sdar",   Kind::SDAR,   151669, {151645, 151643},  0, 0.0,   0.9, 16,   16,  0.0, 0.0,  0,     -1,     -1, false,  nullptr},
    {"llada2", Kind::LLaDA2, 156895, {156892},          0, 0.0,   0.0, 32,   32,  0.5, 0.0, 16, 156930, 156931, false,  "32:48"},
    {"ar",     Kind::AR,         -1, {},                0, 0.0,   0.0,  0,    0,  0.0, 0.0,  0,     -1,     -1, false,  nullptr},
};

const Preset* find_preset(const std::string& name) {
    for (const Preset& p : kPresets)
        if (name == p.name) return &p;
    return nullptr;
}

/// The autoregressive yardstick (Qwen3-8B) as a Policy, so it runs through the same loop,
/// the same backend and the same accounting as the diffusion policies: one causal prefill
/// pass whose last row's logits give the first token, then one row per pass. It samples
/// with open_dllm::sample in HF's order (temperature, then top-k / top-p).
class ArPolicy final : public Policy {
public:
    explicit ArPolicy(open_dllm::Common c) : c_(std::move(c)) {}

    void start(const std::vector<int32_t>& prompt) override {
        if (prompt.empty()) throw std::invalid_argument("ar: empty prompt");
        prompt_ = prompt;
        rng_.seed(c_.sampling.seed);
        n_ = 0;
        generated_ = 0;
        passes_ = 0;
        out_.clear();
        last_ = -1;
        done_ = c_.max_new_tokens <= 0;
    }

    bool done() const override { return done_; }

    Pass next() override {
        if (done_) throw std::logic_error("ar: next() after done()");
        Pass p;
        p.attn = Attn::Causal;
        if (n_ == 0) {
            p.rows.reserve(prompt_.size());
            for (size_t i = 0; i < prompt_.size(); ++i) {
                const int32_t at = static_cast<int32_t>(i);
                p.rows.push_back({prompt_[i], at, at});
            }
        } else {
            p.rows.push_back({last_, n_, n_});
        }
        p.logit_rows.push_back(static_cast<int>(p.rows.size()) - 1);
        return p;
    }

    void feed(const std::vector<const float*>& logits, size_t vocab) override {
        if (logits.size() != 1) throw std::invalid_argument("ar: feed() takes exactly one logit row");
        if (n_ > 0) ++passes_;  // the prefill pass is not a decode pass (policy.hpp)
        n_ += n_ == 0 ? static_cast<int32_t>(prompt_.size()) : 1;
        last_ = open_dllm::sample(logits[0], vocab, c_.sampling, rng_);
        out_.push_back(last_);
        ++generated_;
        const auto& s = c_.stop_ids;
        if (std::find(s.begin(), s.end(), last_) != s.end() || generated_ >= c_.max_new_tokens) done_ = true;
    }

    int32_t keep() const override { return n_; }

    std::vector<int32_t> take() override {
        std::vector<int32_t> t;
        t.swap(out_);
        return t;
    }

    int decode_passes() const override { return passes_; }

private:
    open_dllm::Common c_;
    std::vector<int32_t> prompt_;
    std::mt19937_64 rng_;
    int32_t n_ = 0;        ///< rows forwarded so far, all of them kept
    int32_t last_ = -1;    ///< the token the next pass forwards
    int generated_ = 0;
    int passes_ = 0;
    bool done_ = true;
    std::vector<int32_t> out_;
};

// ---- command line --------------------------------------------------------------------

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    std::string model, policy, in, out, trace_dir;
    int ctx = 8192;
    int threads = 0;  ///< 0: half the hardware threads
    int ngl = 999;
    int ubatch = 512;
    uint64_t seed = 0;
    int max_new = 512;
    double temperature = 0.0;
    int top_k = 0;
    double top_p = 1.0;
    std::optional<int> window, block, steps, max_post_steps;
    std::optional<double> tau, lambda, threshold, edit_threshold;
    std::optional<int32_t> mask_id;
    std::optional<std::vector<int32_t>> stop_ids;
    bool static_schedule = false;
    bool no_eos_early_stop = false;
    bool verbose = false;
    bool help = false;
    std::vector<std::string> given;  ///< policy-specific flags seen, to warn when ignored
};

/// Which policies a policy-specific flag applies to.
struct FlagScope {
    const char* flag;
    const char* policies;
};
const FlagScope kScopes[] = {
    {"--window", "wedlm"},
    {"--lambda", "wedlm"},
    {"--tau", "wedlm sdar"},
    {"--block", "sdar llada2"},
    {"--steps", "sdar llada2"},
    {"--static", "sdar"},
    {"--threshold", "llada2"},
    {"--edit-threshold", "llada2"},
    {"--max-post-steps", "llada2"},
    {"--no-eos-early-stop", "llada2"},
    {"--mask-id", "wedlm sdar llada2"},
};

void usage(std::FILE* f) {
    std::fputs(
        "usage: dllm-ref --model M.gguf --policy wedlm|sdar|llada2|ar --in PROMPTS.jsonl --out RESULTS.jsonl\n"
        "                [options]\n"
        "\n"
        "  --trace DIR            write DIR/<id>.json per prompt (every pass, top-8 logits per logit row)\n"
        "  --ctx N                context (KV cache) size, default 8192\n"
        "  --threads N            CPU threads, default half the hardware threads\n"
        "  --ngl N                layers offloaded to the GPU, default 999 (all)\n"
        "  --ubatch N             physical batch, default 512; rounded up to a multiple of 32, >= the block\n"
        "  --seed N               sampling seed, reset per prompt, default 0\n"
        "  --max-new N            tokens to generate, default 512; a prompt's own \"max_new\" wins\n"
        "  --temperature T        default 0 (greedy)\n"
        "  --top-k K              default 0 (off)\n"
        "  --top-p P              default 1 (off)\n"
        "  --mask-id ID           the mask token (wedlm 151665, sdar 151669, llada2 156895)\n"
        "  --stop-ids a,b,c       stop tokens (wedlm 151643,151645; sdar 151645,151643; llada2 156892;\n"
        "                         ar: the model's end-of-generation tokens)\n"
        "  wedlm:   --window W (16)  --tau T (0.4)  --lambda L (0.02)\n"
        "  sdar:    --block B (16)  --steps S (16)  --tau T (0.9)  --static (low_confidence_static)\n"
        "  llada2:  --block B (32)  --steps S (32)  --threshold T (0.5)  --edit-threshold E (0.0)\n"
        "           --max-post-steps N (16)  --no-eos-early-stop\n"
        "  --verbose              show llama.cpp's INFO log (model load, graph reserves)\n"
        "\n"
        "PROMPTS.jsonl: one {\"id\": str, \"prompt_ids\": [int...], \"max_new\": int (optional)} per line.\n",
        f);
}

int to_int(const std::string& flag, const std::string& v) {
    size_t used = 0;
    long long x = 0;
    try {
        x = std::stoll(v, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != v.size() || x < std::numeric_limits<int32_t>::min() ||
        x > std::numeric_limits<int32_t>::max())
        throw UsageError(flag + " needs an integer, got '" + v + "'");
    return static_cast<int>(x);
}

uint64_t to_u64(const std::string& flag, const std::string& v) {
    size_t used = 0;
    unsigned long long x = 0;
    try {
        if (!v.empty() && v[0] != '-') x = std::stoull(v, &used);  // stoull would wrap "-1"
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != v.size()) throw UsageError(flag + " needs a non-negative integer, got '" + v + "'");
    return static_cast<uint64_t>(x);
}

double to_double(const std::string& flag, const std::string& v) {
    size_t used = 0;
    double x = 0;
    try {
        x = std::stod(v, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != v.size()) throw UsageError(flag + " needs a number, got '" + v + "'");
    return x;
}

std::vector<int32_t> to_ids(const std::string& flag, const std::string& v) {
    std::vector<int32_t> r;
    size_t s = 0;
    while (true) {
        const size_t e = v.find(',', s);
        const std::string tok = v.substr(s, e == std::string::npos ? std::string::npos : e - s);
        if (!tok.empty()) r.push_back(to_int(flag, tok));
        if (e == std::string::npos) break;
        s = e + 1;
    }
    return r;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw UsageError(a + " needs a value");
            return argv[++i];
        };
        auto scoped = [&]() { o.given.push_back(a); };
        if (a == "--model") o.model = value();
        else if (a == "--policy") o.policy = value();
        else if (a == "--in") o.in = value();
        else if (a == "--out") o.out = value();
        else if (a == "--trace") o.trace_dir = value();
        else if (a == "--ctx") o.ctx = to_int(a, value());
        else if (a == "--threads") o.threads = to_int(a, value());
        else if (a == "--ngl") o.ngl = to_int(a, value());
        else if (a == "--ubatch") o.ubatch = to_int(a, value());
        else if (a == "--seed") o.seed = to_u64(a, value());
        else if (a == "--max-new") o.max_new = to_int(a, value());
        else if (a == "--temperature") o.temperature = to_double(a, value());
        else if (a == "--top-k") o.top_k = to_int(a, value());
        else if (a == "--top-p") o.top_p = to_double(a, value());
        else if (a == "--mask-id") { o.mask_id = to_int(a, value()); scoped(); }
        else if (a == "--stop-ids") o.stop_ids = to_ids(a, value());
        else if (a == "--window") { o.window = to_int(a, value()); scoped(); }
        else if (a == "--tau") { o.tau = to_double(a, value()); scoped(); }
        else if (a == "--lambda") { o.lambda = to_double(a, value()); scoped(); }
        else if (a == "--block") { o.block = to_int(a, value()); scoped(); }
        else if (a == "--steps") { o.steps = to_int(a, value()); scoped(); }
        else if (a == "--static") { o.static_schedule = true; scoped(); }
        else if (a == "--threshold") { o.threshold = to_double(a, value()); scoped(); }
        else if (a == "--edit-threshold") { o.edit_threshold = to_double(a, value()); scoped(); }
        else if (a == "--max-post-steps") { o.max_post_steps = to_int(a, value()); scoped(); }
        else if (a == "--no-eos-early-stop") { o.no_eos_early_stop = true; scoped(); }
        else if (a == "--verbose") o.verbose = true;
        else if (a == "-h" || a == "--help") o.help = true;
        else throw UsageError("unknown argument '" + a + "'");
    }
    if (o.help) return o;
    if (o.model.empty() || o.policy.empty() || o.in.empty() || o.out.empty())
        throw UsageError("--model, --policy, --in and --out are required");
    if (!find_preset(o.policy)) throw UsageError("--policy must be wedlm, sdar, llada2 or ar, got '" + o.policy + "'");
    if (o.ctx < 64) throw UsageError("--ctx must be >= 64");
    if (o.threads < 0) throw UsageError("--threads must be >= 0");
    if (o.ubatch < 1) throw UsageError("--ubatch must be >= 1");
    if (o.temperature < 0) throw UsageError("--temperature must be >= 0");
    if (o.top_k < 0) throw UsageError("--top-k must be >= 0");
    if (!(o.top_p > 0 && o.top_p <= 1)) throw UsageError("--top-p must be in (0, 1]");
    return o;
}

// ---- settings in effect ----------------------------------------------------------------

/// The preset, overridden by flags. Policy constructors validate the ranges.
struct Knobs {
    int32_t mask_id = -1;
    std::vector<int32_t> stop_ids;
    int window = 0;
    double tau = 0, lambda = 0;
    int block = 0, steps = 0;
    bool dynamic = true;
    double threshold = 0, edit_threshold = 0;
    int max_post_steps = 0;
    bool eos_early_stop = true;
    int32_t delete_id = -1, insert_id = -1;
    open_dllm::Sampling sampling;
};

Knobs resolve(const Options& o, const Preset& p) {
    Knobs k;
    k.mask_id = o.mask_id.value_or(p.mask_id);
    k.stop_ids = o.stop_ids.value_or(p.stop_ids);
    k.window = o.window.value_or(p.window);
    k.tau = o.tau.value_or(p.tau);
    k.lambda = o.lambda.value_or(p.lambda);
    k.block = o.block.value_or(p.block);
    k.steps = o.steps.value_or(p.steps);
    k.dynamic = !o.static_schedule;
    k.threshold = o.threshold.value_or(p.threshold);
    k.edit_threshold = o.edit_threshold.value_or(p.edit_threshold);
    k.max_post_steps = o.max_post_steps.value_or(p.max_post_steps);
    k.eos_early_stop = !o.no_eos_early_stop;
    k.delete_id = p.delete_id;
    k.insert_id = p.insert_id;
    k.sampling.temperature = static_cast<float>(o.temperature);
    k.sampling.top_k = o.top_k;
    k.sampling.top_p = static_cast<float>(o.top_p);
    k.sampling.seed = o.seed;
    return k;
}

/// One policy per prompt: max_new_tokens lives in the config, and a prompt may carry its own.
std::unique_ptr<Policy> make_policy(Kind kind, const Knobs& k, int max_new) {
    open_dllm::Common c;
    c.mask_id = k.mask_id;
    c.stop_ids = k.stop_ids;
    c.max_new_tokens = max_new;
    c.sampling = k.sampling;
    switch (kind) {
    case Kind::WeDLM: {
        open_dllm::WeDLMConfig w;
        w.common = c;
        w.window = k.window;
        w.tau = k.tau;
        w.lambda = k.lambda;
        return open_dllm::make_wedlm(w);
    }
    case Kind::SDAR: {
        open_dllm::SDARConfig s;
        s.common = c;
        s.block = k.block;
        s.steps = k.steps;
        s.dynamic = k.dynamic;
        s.tau = k.tau;
        return open_dllm::make_sdar(s);
    }
    case Kind::LLaDA2: {
        open_dllm::LLaDA2Config l;
        l.common = c;
        l.block = k.block;
        l.steps = k.steps;
        l.threshold = k.threshold;
        l.editing_threshold = k.edit_threshold;
        l.max_post_steps = k.max_post_steps;
        l.delete_id = k.delete_id;
        l.insert_id = k.insert_id;
        l.eos_early_stop = k.eos_early_stop;
        return open_dllm::make_llada2(l);
    }
    case Kind::AR:
        return std::make_unique<ArPolicy>(c);
    }
    throw std::logic_error("make_policy: unknown policy kind");
}

/// The furthest cache position a request can reach, so a prompt that cannot fit is refused
/// before it runs for minutes and fails inside llama_decode.
int64_t extent(Kind kind, const Knobs& k, int64_t prompt_len, int64_t max_new) {
    const int64_t gen = std::max<int64_t>(1, max_new);
    switch (kind) {
    case Kind::WeDLM:
        return prompt_len + gen + 2 * k.window;  // a window pass holds W + p <= 2W rows past ctx
    case Kind::SDAR:
    case Kind::LLaDA2:
        return (prompt_len + gen + k.block - 1) / k.block * k.block;  // whole aligned blocks
    case Kind::AR:
        return prompt_len + gen;
    }
    return prompt_len + gen;
}

// ---- llama.cpp -------------------------------------------------------------------------

bool g_verbose = false;

/// llama.cpp and ggml both log through this. Model loading and every graph re-reserve log at
/// INFO; keep warnings and errors unless --verbose. A CONT record continues the one before
/// it, so it takes that one's level.
void log_cb(ggml_log_level level, const char* text, void* /*user*/) {
    static ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (g_verbose || last >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

/// Sets a variable for this process unless it is already set, and returns the value in
/// effect. Called before llama_backend_init, so llama.cpp sees it from its first getenv.
std::string ensure_env(const char* name, const char* value) {
    if (const char* cur = std::getenv(name)) return cur;
#ifdef _WIN32
    if (_putenv_s(name, value) != 0) throw std::runtime_error(std::string("cannot set ") + name);
#else
    if (setenv(name, value, 0) != 0) throw std::runtime_error(std::string("cannot set ") + name);
#endif
    return value;
}

struct ModelFree {
    void operator()(llama_model* m) const { llama_model_free(m); }
};
struct ContextFree {
    void operator()(llama_context* c) const { llama_free(c); }
};

/// A llama_batch for sequence 0, owned. llama_batch_init leaves every field uninitialised;
/// add() writes all five per row.
class Batch {
public:
    explicit Batch(int32_t capacity) : b_(llama_batch_init(capacity, 0, 1)), cap_(capacity) {}
    ~Batch() { llama_batch_free(b_); }
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    void clear() { b_.n_tokens = 0; }
    void add(llama_token token, llama_pos pos, bool logits) {
        if (b_.n_tokens >= cap_) throw std::logic_error("Batch: over capacity");
        const int32_t i = b_.n_tokens++;
        b_.token[i] = token;
        b_.pos[i] = pos;
        b_.n_seq_id[i] = 1;
        b_.seq_id[i][0] = 0;
        b_.logits[i] = static_cast<int8_t>(logits ? 1 : 0);
    }
    const llama_batch& get() const { return b_; }

private:
    llama_batch b_;
    int32_t cap_;
};

enum class Route { Batch, Rows };

const char* route_name(Route r) { return r == Route::Batch ? "batch" : "rows"; }

/// Block passes are one batch. A Causal pass is one batch when every row's slot is its
/// position and the positions are consecutive: physical order is then position order, and
/// llama.cpp's mask (position-causal, llama-kv-cache.cpp:1647) is the policy's (physical-
/// causal). Anything else -- WeDLM moves filled tokens ahead of masks -- runs row by row.
Route route_of(const Pass& p) {
    if (p.attn == Attn::Block) return Route::Batch;
    for (size_t j = 0; j < p.rows.size(); ++j) {
        const Row& r = p.rows[j];
        if (r.slot != r.pos) return Route::Rows;
        if (j > 0 && r.pos != p.rows[j - 1].pos + 1) return Route::Rows;
    }
    return Route::Batch;
}

/// Runs passes on one llama_context, sequence 0, and holds the logits a pass asked for
/// until the next pass (llama.cpp's own buffer is overwritten by every llama_decode, and the
/// row-by-row route makes several per pass).
class Backend {
public:
    Backend(llama_context* ctx, int32_t n_vocab)
        : ctx_(ctx),
          mem_(llama_get_memory(ctx)),
          n_vocab_(n_vocab),
          n_batch_(static_cast<int32_t>(llama_n_batch(ctx))),
          n_ubatch_(static_cast<int32_t>(llama_n_ubatch(ctx))),
          batch_(static_cast<int32_t>(llama_n_batch(ctx))) {}

    /// Between prompts. The cells go; their data stays, as it does after every seq_rm --
    /// an empty cell is masked whatever it holds.
    void reset() {
        llama_memory_clear(mem_, false);
        kept_ = 0;
    }

    /// Runs one pass and returns the llama_decode calls it took. Synchronises at the end,
    /// so the caller's clock measures the pass and not just its submission.
    int run(const Pass& p, Route route) {
        check(p);
        logits_.resize(p.logit_rows.size() * static_cast<size_t>(n_vocab_));
        const int calls = route == Route::Rows ? run_rows(p) : run_batch(p, p.attn == Attn::Causal);
        llama_synchronize(ctx_);
        return calls;
    }

    /// The pass's logits, in logit_rows order, valid until the next run().
    void logit_ptrs(size_t n, std::vector<const float*>& out) const {
        out.resize(n);
        for (size_t k = 0; k < n; ++k) out[k] = logits_.data() + k * static_cast<size_t>(n_vocab_);
    }

    /// Drops every cache row at slot >= keep. The cache is keyed by position, so this is a
    /// position trim, exact only while slot >= keep <=> pos >= keep for this pass's rows:
    /// policy.hpp promises it for these policies, and this checks it.
    void trim(const Pass& p, int32_t keep) {
        for (const Row& r : p.rows)
            if ((r.slot >= keep) != (r.pos >= keep))
                throw std::logic_error("row at slot " + std::to_string(r.slot) + ", position " + std::to_string(r.pos) +
                                       " straddles keep() = " + std::to_string(keep) +
                                       ": a position-keyed cache cannot drop it by slot");
        if (!llama_memory_seq_rm(mem_, 0, keep, -1))
            throw std::runtime_error("llama_memory_seq_rm refused to trim at " + std::to_string(keep));
        kept_ = keep;
    }

private:
    /// What llama.cpp would otherwise assert on (killing the whole run) or silently get
    /// wrong, refused per pass instead.
    void check(const Pass& p) {
        const size_t n = p.rows.size();
        if (n == 0) throw std::logic_error("a pass with no rows");
        if (n > static_cast<size_t>(n_batch_))  // llama-context.cpp:1711 asserts
            throw std::runtime_error("a pass of " + std::to_string(n) + " rows exceeds n_batch " +
                                     std::to_string(n_batch_) + " (raise --ctx)");
        if (p.attn == Attn::Block && n > static_cast<size_t>(n_ubatch_))  // llama-context.cpp:1713 asserts
            throw std::runtime_error("a block pass of " + std::to_string(n) + " rows needs n_ubatch >= " +
                                     std::to_string(n) + " (raise --ubatch)");
        want_.assign(n, -1);
        for (size_t k = 0; k < p.logit_rows.size(); ++k) {
            const int r = p.logit_rows[k];
            if (r < 0 || static_cast<size_t>(r) >= n) throw std::logic_error("logit row " + std::to_string(r) + " is not a row");
            if (want_[static_cast<size_t>(r)] >= 0) throw std::logic_error("logit row " + std::to_string(r) + " listed twice");
            want_[static_cast<size_t>(r)] = static_cast<int32_t>(k);
        }
        pos_.clear();
        for (const Row& r : p.rows) {
            if (r.token < 0 || r.token >= n_vocab_)
                throw std::logic_error("token " + std::to_string(r.token) + " is outside the vocab");
            // A row below the kept length would share a position with a kept cell: two
            // cells, one position, and the next trim cannot tell them apart.
            if (r.pos < kept_)
                throw std::logic_error("row at position " + std::to_string(r.pos) + " is inside the kept cache (keep " +
                                       std::to_string(kept_) + ")");
            pos_.push_back(r.pos);
        }
        std::sort(pos_.begin(), pos_.end());
        if (std::adjacent_find(pos_.begin(), pos_.end()) != pos_.end())
            throw std::logic_error("two rows of one pass share a position");
    }

    /// One llama_decode. A Block pass goes in position order: the mask is symmetric, so
    /// order does not change the result, and llama.cpp refuses positions that decrease
    /// within a sequence (llama-batch.cpp:379).
    int run_batch(const Pass& p, bool causal) {
        const size_t n = p.rows.size();
        order_.resize(n);
        std::iota(order_.begin(), order_.end(), size_t{0});
        if (!causal)
            std::stable_sort(order_.begin(), order_.end(),
                             [&](size_t a, size_t b) { return p.rows[a].pos < p.rows[b].pos; });
        set_causal(causal);
        batch_.clear();
        for (size_t i : order_) batch_.add(p.rows[i].token, p.rows[i].pos, want_[i] >= 0);
        decode();
        for (size_t b = 0; b < n; ++b)
            if (want_[order_[b]] >= 0) copy_logits(static_cast<int32_t>(b), want_[order_[b]]);
        return 1;
    }

    /// One llama_decode per row, causal off: each row sees every cell of the sequence, which
    /// at that moment is the kept cache, the rows before it in this pass, and itself.
    int run_rows(const Pass& p) {
        set_causal(false);
        for (size_t i = 0; i < p.rows.size(); ++i) {
            batch_.clear();
            batch_.add(p.rows[i].token, p.rows[i].pos, want_[i] >= 0);
            decode();
            if (want_[i] >= 0) copy_logits(0, want_[i]);
        }
        return static_cast<int>(p.rows.size());
    }

    /// Switches llama.cpp's attention mode only when it changes. Each switch makes the next
    /// llama_decode build a new scheduler and re-reserve every graph (set_causal_attn sets
    /// sched_need_reserve, llama-context.cpp:1184-1193; sched_reserve, :581), so restoring
    /// causal after every pass would cost SDAR and LLaDA2 -- Block passes only -- two
    /// re-reserves per pass for nothing. Every route sets the mode it needs before it
    /// decodes, so a mode left over from the previous pass never reaches the next.
    void set_causal(bool on) {
        if (on == causal_) return;
        llama_set_causal_attn(ctx_, on);
        causal_ = on;
    }

    void decode() {
        const int32_t rc = llama_decode(ctx_, batch_.get());
        if (rc == 0) return;
        throw std::runtime_error("llama_decode returned " + std::to_string(rc) +
                                 (rc == 1 ? " (no KV cache slot: raise --ctx)" : ""));
    }

    void copy_logits(int32_t batch_index, int32_t k) {
        const float* src = llama_get_logits_ith(ctx_, batch_index);
        if (!src) throw std::runtime_error("llama_get_logits_ith returned no logits for batch row " + std::to_string(batch_index));
        std::copy(src, src + n_vocab_, logits_.data() + static_cast<size_t>(k) * static_cast<size_t>(n_vocab_));
    }

    llama_context* ctx_;
    llama_memory_t mem_;
    int32_t n_vocab_;
    int32_t n_batch_, n_ubatch_;
    Batch batch_;
    bool causal_ = true;  ///< the context is created causal (LLAMA_ATTENTION_TYPE_CAUSAL)
    int32_t kept_ = 0;    ///< keep() of the last trim: the cache holds positions below it
    std::vector<float> logits_;    ///< logit_rows.size() x n_vocab
    std::vector<int32_t> want_;    ///< per row: its index in logit_rows, or -1
    std::vector<size_t> order_;    ///< batch order of a one-batch pass
    std::vector<int32_t> pos_;     ///< scratch for the duplicate check
};

// ---- one prompt ------------------------------------------------------------------------

/// The k largest logits of a row, largest first, ties to the lower id: one linear scan with
/// a small sorted buffer, not a sort of the whole vocab per row.
json top_logits(const float* x, int32_t n, size_t k) {
    std::vector<std::pair<float, int32_t>> best;
    best.reserve(k + 1);
    for (int32_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (best.size() == k && !(v > best.back().first)) continue;
        auto at = std::find_if(best.begin(), best.end(), [v](const std::pair<float, int32_t>& e) { return v > e.first; });
        best.insert(at, std::make_pair(v, i));
        if (best.size() > k) best.pop_back();
    }
    json a = json::array();
    for (const auto& e : best) a.push_back(json::array({e.second, jfloat(e.first)}));
    return a;
}

json trace_pass(const Pass& p, Route route, const std::vector<const float*>& logits, int32_t n_vocab, int32_t keep,
                const std::vector<int32_t>& taken) {
    json rows = json::array();
    for (const Row& r : p.rows) rows.push_back(json::array({r.token, r.pos, r.slot}));
    json top = json::array();
    for (const float* l : logits) top.push_back(top_logits(l, n_vocab, 8));
    json e;
    e["rows"] = std::move(rows);
    e["attn"] = p.attn == Attn::Block ? "block" : "causal";
    e["route"] = route_name(route);
    e["logit_rows"] = p.logit_rows;
    e["top8"] = std::move(top);
    e["keep"] = keep;
    e["taken"] = taken;
    return e;
}

struct Outcome {
    std::vector<int32_t> out;
    int policy_passes = 0;   ///< next() calls
    int llama_calls = 0;     ///< llama_decode calls
    int decode_passes = 0;   ///< policy->decode_passes()
    double ms_prefill = 0;   ///< passes that did not advance decode_passes()
    double ms_decode = 0;
    double ms_policy = 0;    ///< host time inside next() and feed(), included in the two above
};

/// policy.hpp's backend loop. A pass counts as prefill when it leaves decode_passes()
/// where it was -- the contract's own definition -- so the policy, not the driver, decides
/// which passes are prompt work.
Outcome run_prompt(Policy& policy, Backend& be, const std::vector<int32_t>& prompt, int32_t n_vocab, json* trace) {
    Outcome o;
    be.reset();
    policy.start(prompt);
    std::vector<const float*> logits;
    while (!policy.done()) {
        const auto t0 = Clock::now();
        const Pass p = policy.next();
        const auto t1 = Clock::now();
        const Route route = route_of(p);
        o.llama_calls += be.run(p, route);
        be.logit_ptrs(p.logit_rows.size(), logits);
        const auto t2 = Clock::now();
        const int before = policy.decode_passes();
        policy.feed(logits, static_cast<size_t>(n_vocab));
        const int32_t keep = policy.keep();
        const std::vector<int32_t> taken = policy.take();
        const auto t3 = Clock::now();
        be.trim(p, keep);
        const auto t4 = Clock::now();

        ++o.policy_passes;
        (policy.decode_passes() == before ? o.ms_prefill : o.ms_decode) += ms_between(t0, t4);
        o.ms_policy += ms_between(t0, t1) + ms_between(t2, t3);
        if (trace) (*trace)["passes"].push_back(trace_pass(p, route, logits, n_vocab, keep, taken));
        o.out.insert(o.out.end(), taken.begin(), taken.end());
    }
    o.decode_passes = policy.decode_passes();
    return o;
}

// ---- input / output --------------------------------------------------------------------

struct Prompt {
    std::string id;
    std::vector<int32_t> ids;
    std::optional<int> max_new;
};

std::vector<Prompt> read_prompts(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<Prompt> r;
    std::string line;
    int n = 0;
    while (std::getline(f, line)) {
        ++n;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        try {
            const json j = json::parse(line);
            Prompt p;
            const json& id = j.at("id");
            p.id = id.is_string() ? id.get<std::string>() : dump(id);
            p.ids = j.at("prompt_ids").get<std::vector<int32_t>>();
            if (j.contains("max_new") && !j.at("max_new").is_null()) p.max_new = j.at("max_new").get<int>();
            r.push_back(std::move(p));
        } catch (const std::exception& e) {
            throw std::runtime_error(path + ":" + std::to_string(n) + ": " + e.what());
        }
    }
    return r;
}

/// A prompt id as a file name: anything but [A-Za-z0-9._-] becomes '_'.
std::string file_safe(const std::string& id) {
    std::string s;
    for (char c : id) s += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.') ? c : '_';
    return s.empty() ? std::string("prompt") : s;
}

/// ar stops where the model's own vocab says generation ends (Qwen3: <|im_end|>,
/// <|endoftext|> and the few llama.cpp also marks), which is what a yardstick should do
/// whatever model it is pointed at.
std::vector<int32_t> eog_tokens(const llama_vocab* vocab) {
    std::vector<int32_t> r;
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (int32_t t = 0; t < n; ++t)
        if (llama_vocab_is_eog(vocab, t)) r.push_back(t);
    return r;
}

json settings_of(const Preset& ps, const Knobs& k, const Options& o, int32_t n_ctx, int32_t n_batch, int32_t n_ubatch,
                 int threads, const json& env) {
    json s;
    s["model"] = o.model;
    s["ctx"] = n_ctx;
    s["n_batch"] = n_batch;
    s["n_ubatch"] = n_ubatch;
    s["ngl"] = o.ngl;
    s["threads"] = threads;
    s["seed"] = o.seed;
    s["temperature"] = o.temperature;
    s["top_k"] = o.top_k;
    s["top_p"] = o.top_p;
    if (ps.kind != Kind::AR) s["mask_id"] = k.mask_id;
    s["stop_ids"] = k.stop_ids;
    switch (ps.kind) {
    case Kind::WeDLM:
        s["window"] = k.window;
        s["tau"] = k.tau;
        s["lambda"] = k.lambda;
        break;
    case Kind::SDAR:
        s["block"] = k.block;
        s["steps"] = k.steps;
        s["tau"] = k.tau;
        s["dynamic"] = k.dynamic;
        break;
    case Kind::LLaDA2:
        s["block"] = k.block;
        s["steps"] = k.steps;
        s["threshold"] = k.threshold;
        s["edit_threshold"] = k.edit_threshold;
        s["max_post_steps"] = k.max_post_steps;
        s["eos_early_stop"] = k.eos_early_stop;
        s["delete_id"] = k.delete_id;
        s["insert_id"] = k.insert_id;
        break;
    case Kind::AR:
        break;
    }
    s["env"] = env;
    return s;
}

int32_t round_up(int32_t v, int32_t m) { return (v + m - 1) / m * m; }

// ---- the run ---------------------------------------------------------------------------

int run(const Options& opt, const Preset& ps, const json& env) {
    Knobs k = resolve(opt, ps);
    // Fail on bad knobs now, not after a minute of model loading: the policy constructors
    // are the validators.
    try {
        (void)make_policy(ps.kind, k, opt.max_new);
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "dllm-ref: %s\n", e.what());
        return 2;
    }
    if (ps.kind == Kind::LLaDA2 && k.block % 32 != 0)
        std::fprintf(stderr, "dllm-ref: warning: --block %d is not a multiple of 32; LLAMA_BLOCK_ROUTING routes whole "
                             "32-token blocks only, so these passes use token-level routing\n", k.block);
    const std::vector<Prompt> prompts = read_prompts(opt.in);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = opt.ngl;
    std::unique_ptr<llama_model, ModelFree> model(llama_model_load_from_file(opt.model.c_str(), mp));
    if (!model) {
        std::fprintf(stderr, "dllm-ref: cannot load %s\n", opt.model.c_str());
        return 2;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (ps.kind == Kind::AR && !opt.stop_ids) k.stop_ids = eog_tokens(vocab);
    if (ps.kind != Kind::AR && (k.mask_id < 0 || k.mask_id >= n_vocab)) {
        std::fprintf(stderr, "dllm-ref: mask id %d is outside this model's vocab (%d tokens): wrong --policy for the model?\n",
                     k.mask_id, n_vocab);
        return 2;
    }

    // n_batch is the context: llama_decode asserts a batch fits n_batch (llama-context.cpp:
    // 1711), the longest pass is a whole prompt prefilled in one causal pass (wedlm, ar), and
    // n_batch costs nothing by itself (the logits buffer grows to the outputs actually asked
    // for). n_ubatch is the physical split and sizes the compute buffers: --ubatch, at least
    // the block, rounded up to 32. A Block pass must be ONE ubatch (llama-context.cpp:1713).
    // With one KV stream llama.cpp splits a batch with split_simple (llama-kv-cache.cpp:709):
    // in order, n_ubatch rows at a time, never evenly. So a 32-row LLaDA2 pass is one 32-row
    // ubatch, and any longer batch is cut on 32-row boundaries -- what LLAMA_BLOCK_ROUTING
    // needs, and a multiple of 32 also keeps the reserve graph on the routed topology.
    // kv_unified makes an iSWA cache split the same way (llama-kv-cache-iswa.cpp:164 would
    // otherwise use split_equal); with one sequence it changes nothing else.
    const bool block_policy = ps.kind == Kind::SDAR || ps.kind == Kind::LLaDA2;
    const int32_t ubatch = round_up(std::max<int32_t>(opt.ubatch, block_policy ? k.block : 1), 32);
    const unsigned hw = std::thread::hardware_concurrency();
    const int threads = opt.threads > 0 ? opt.threads : (hw == 0 ? 4 : std::max(1, static_cast<int>(hw / 2)));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(opt.ctx);
    cp.n_batch = static_cast<uint32_t>(opt.ctx);
    cp.n_ubatch = static_cast<uint32_t>(ubatch);
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;  // Backend::set_causal starts from here
    cp.kv_unified = true;
    cp.no_perf = true;
    std::unique_ptr<llama_context, ContextFree> ctx(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        std::fprintf(stderr, "dllm-ref: cannot create a context (ctx %d, ubatch %d)\n", opt.ctx, ubatch);
        return 2;
    }
    if (!llama_get_memory(ctx.get())) {
        // llama.cpp builds no KV cache for its own diffusion archs (llama-model.cpp:2114-2120).
        std::fprintf(stderr, "dllm-ref: this model has no KV cache in llama.cpp; convert WeDLM/SDAR as qwen3 and "
                             "LLaDA2.2 as bailingmoe2 (plan, Phase 1 'Weights')\n");
        return 2;
    }
    const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(ctx.get()));
    const int32_t n_batch = static_cast<int32_t>(llama_n_batch(ctx.get()));
    const int32_t n_ubatch = static_cast<int32_t>(llama_n_ubatch(ctx.get()));
    if (block_policy && n_ubatch < k.block) {
        std::fprintf(stderr, "dllm-ref: n_ubatch %d < block %d\n", n_ubatch, k.block);
        return 2;
    }

    char desc[256] = {0};
    llama_model_desc(model.get(), desc, sizeof desc);
    std::fprintf(stderr, "[dllm-ref] %s, %.2fB params, vocab %d | ctx %d, batch %d, ubatch %d, threads %d | policy %s",
                 desc, static_cast<double>(llama_model_n_params(model.get())) / 1e9, n_vocab, n_ctx, n_batch, n_ubatch,
                 threads, ps.name);
    for (auto it = env.begin(); it != env.end(); ++it)
        std::fprintf(stderr, ", %s=%s", it.key().c_str(), it.value().get<std::string>().c_str());
    std::fprintf(stderr, " | %zu prompts\n", prompts.size());
    if (g_verbose) std::fprintf(stderr, "[dllm-ref] %s\n", llama_print_system_info());

    std::ofstream out(opt.out, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "dllm-ref: cannot write %s\n", opt.out.c_str());
        return 2;
    }
    if (!opt.trace_dir.empty()) {
        std::error_code ec;
        fs::create_directories(opt.trace_dir, ec);
        if (ec) {
            std::fprintf(stderr, "dllm-ref: cannot create %s: %s\n", opt.trace_dir.c_str(), ec.message().c_str());
            return 2;
        }
    }

    const json settings = settings_of(ps, k, opt, n_ctx, n_batch, n_ubatch, threads, env);
    Backend be(ctx.get(), n_vocab);
    int failed = 0;
    long long total_out = 0, total_dp = 0;
    double total_ms_decode = 0;
    for (const Prompt& pr : prompts) {
        const int max_new = pr.max_new.value_or(opt.max_new);
        json st = settings;
        st["max_new"] = max_new;
        json line;
        line["id"] = pr.id;
        line["policy"] = ps.name;
        try {
            const int64_t need = extent(ps.kind, k, static_cast<int64_t>(pr.ids.size()), max_new);
            if (need > n_ctx)
                throw std::runtime_error("needs up to " + std::to_string(need) + " cache positions, --ctx is " +
                                         std::to_string(n_ctx));
            std::unique_ptr<Policy> policy = make_policy(ps.kind, k, max_new);
            json trace;
            if (!opt.trace_dir.empty()) {
                trace["id"] = pr.id;
                trace["policy"] = ps.name;
                trace["prompt_ids"] = pr.ids;
                trace["settings"] = st;
                trace["passes"] = json::array();
            }
            const Outcome o = run_prompt(*policy, be, pr.ids, n_vocab, opt.trace_dir.empty() ? nullptr : &trace);

            const int committed = static_cast<int>(o.out.size());
            // ar's prefill pass yields its first token, so a one-token-per-forward yardstick
            // reads decode_passes = tokens generated.
            const int decode_passes = ps.kind == Kind::AR ? committed : o.decode_passes;
            line["out_ids"] = o.out;
            line["prompt_len"] = pr.ids.size();
            line["passes"] = o.llama_calls;
            line["decode_passes"] = decode_passes;
            line["policy_passes"] = o.policy_passes;
            line["committed"] = committed;
            line["ms_prefill"] = round2(o.ms_prefill);
            line["ms_decode"] = round2(o.ms_decode);
            line["ms_policy"] = round2(o.ms_policy);
            line["settings"] = st;

            if (!opt.trace_dir.empty()) {
                trace["output"] = o.out;
                trace["decode_passes"] = decode_passes;
                trace["llama_decode_calls"] = o.llama_calls;
                const fs::path tp = fs::path(opt.trace_dir) / (file_safe(pr.id) + ".json");
                std::ofstream tf(tp, std::ios::binary | std::ios::trunc);
                if (tf) tf << dump(trace) << '\n';
                if (!tf) std::fprintf(stderr, "dllm-ref: warning: cannot write %s\n", tp.string().c_str());
            }

            total_out += committed;
            total_dp += decode_passes;
            total_ms_decode += o.ms_decode;
            std::fprintf(stderr,
                         "[dllm-ref] %s: prompt %zu, out %d, decode passes %d (%.2f tok/pass), llama_decode %d, "
                         "prefill %.2f s, decode %.2f s (%.2f tok/s)\n",
                         pr.id.c_str(), pr.ids.size(), committed, decode_passes,
                         decode_passes > 0 ? static_cast<double>(committed) / decode_passes : 0.0, o.llama_calls,
                         o.ms_prefill / 1000.0, o.ms_decode / 1000.0,
                         o.ms_decode > 0 ? committed / (o.ms_decode / 1000.0) : 0.0);
        } catch (const std::exception& e) {
            ++failed;
            line["error"] = e.what();
            std::fprintf(stderr, "[dllm-ref] %s: FAILED: %s\n", pr.id.c_str(), e.what());
        }
        out << dump(line) << '\n';
        out.flush();
    }

    std::fprintf(stderr, "[dllm-ref] %zu prompts, %d failed; %lld tokens over %lld decode passes (%.2f tok/pass), %.2f tok/s decode\n",
                 prompts.size(), failed, total_out, total_dp,
                 total_dp > 0 ? static_cast<double>(total_out) / static_cast<double>(total_dp) : 0.0,
                 total_ms_decode > 0 ? static_cast<double>(total_out) / (total_ms_decode / 1000.0) : 0.0);
    return failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        opt = parse(argc, argv);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "dllm-ref: %s\n\n", e.what());
        usage(stderr);
        return 2;
    }
    if (opt.help) {
        usage(stdout);
        return 0;
    }
    const Preset& ps = *find_preset(opt.policy);
    for (const std::string& flag : opt.given)
        for (const FlagScope& s : kScopes)
            if (flag == s.flag && (std::string(" ") + s.policies + " ").find(" " + opt.policy + " ") == std::string::npos)
                std::fprintf(stderr, "dllm-ref: warning: %s applies to %s only; ignored for %s\n", s.flag, s.policies,
                             opt.policy.c_str());
    g_verbose = opt.verbose;

    int rc = 2;
    try {
        // Before llama.cpp loads anything: the patches read these from the process environment.
        json env = json::object();
        if (ps.nonconsecutive)
            env["LLAMA_ALLOW_NONCONSECUTIVE_POS"] = ensure_env("LLAMA_ALLOW_NONCONSECUTIVE_POS", "1");
        if (ps.block_routing) env["LLAMA_BLOCK_ROUTING"] = ensure_env("LLAMA_BLOCK_ROUTING", ps.block_routing);

        llama_log_set(log_cb, nullptr);
        llama_backend_init();
        rc = run(opt, ps, env);
        llama_backend_free();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dllm-ref: %s\n", e.what());
        rc = 2;
    }
    return rc;
}
