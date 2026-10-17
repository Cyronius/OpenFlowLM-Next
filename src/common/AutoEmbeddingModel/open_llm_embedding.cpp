/// \file open_llm_embedding.cpp
/// \brief OpenLlmEmbedding over open_qwen36::Core: last-token pooling, and the reranker's yes/no score.
#include "AutoEmbeddingModel/open_llm_embedding.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "AutoEmbeddingModel/llm_conventions.hpp"
#include "AutoModel/request_error.hpp"
#include "utils/utils.hpp"

#ifdef OFLM_USE_OPEN_QWEN36
#include "open_qwen36/core.hpp"
#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/manifest.hpp"
#include "tokenizers_cpp.h"
#endif

namespace fs = std::filesystem;

#ifdef OFLM_USE_OPEN_QWEN36

namespace {

bool complete_kernel_set(const fs::path& dir) {
    std::error_code ec;
    if (!fs::is_regular_file(dir / "manifest.json", ec)) return false;
    try {
        const open_qwen36::Manifest m = open_qwen36::Manifest::load((dir / "manifest.json").string());
        for (const auto& f : m.files())
            if (!fs::is_regular_file(dir / f, ec)) return false;
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

/// OFLM_OPEN_KERNELS_DIR, then the link oflm-add makes beside the model, then every xclbins root.
std::string find_kernel_set(const fs::path& model_dir) {
    if (const char* env = std::getenv("OFLM_OPEN_KERNELS_DIR"))
        if (complete_kernel_set(env)) return env;
    if (complete_kernel_set(model_dir / "open_kernels")) return (model_dir / "open_kernels").string();
    for (const std::string& root : utils::xclbin_roots()) {
        const fs::path cand = fs::path(root) / "xclbins" / model_dir.filename() / "open_kernels";
        if (complete_kernel_set(cand)) return cand.string();
    }
    return {};
}

std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("OpenLlmEmbedding: cannot read " + p.string());
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}

}  // namespace

struct OpenLlmEmbedding::Impl {
    std::unique_ptr<open_qwen36::Core> core;
    std::unique_ptr<tokenizers::Tokenizer> tok;
    bool add_eos = false;
    int eos = -1;
    std::vector<float> yes_row, no_row;

    std::vector<int> encode(const std::string& s) const {
        const std::vector<int32_t> v = tok->Encode(s);
        return std::vector<int>(v.begin(), v.end());
    }

    std::vector<float> hidden(const std::vector<int>& ids) const {
        if (ids.size() > core->max_ctx())
            throw request_error("the input is " + std::to_string(ids.size()) + " tokens; this model's context is " +
                                std::to_string(core->max_ctx()));
        return core->prefill_hidden(ids);
    }
};

OpenLlmEmbedding::OpenLlmEmbedding(oflm_rt::device* npu_device_inst, std::string tag, Mode mode)
    : AutoEmbeddingModel(npu_device_inst, std::move(tag)), impl_(new Impl), mode_(mode) {}

OpenLlmEmbedding::~OpenLlmEmbedding() = default;

void OpenLlmEmbedding::load_model(std::string model_path, json model_info, bool enable_preemption) {
    (void)enable_preemption;
    this->model_path = std::move(model_path);
    const fs::path dir(this->model_path);
    open_qwen36::CoreConfig cfg;
    cfg.model_dir = dir.string();
    cfg.kernel_dir = find_kernel_set(dir);
    if (cfg.kernel_dir.empty())
        throw std::runtime_error("OpenLlmEmbedding: no open kernel set for " + dir.string() +
                                 " (oflm-add links one at <model>/open_kernels; or set OFLM_OPEN_KERNELS_DIR)");
    cfg.max_ctx = model_info.contains("default_context_length") && model_info["default_context_length"].is_number_integer()
                      ? model_info["default_context_length"].get<size_t>()
                      : 8192;
    cfg.verbose = std::getenv("OFLM_OPEN_VERBOSE") != nullptr;
    cfg.lm_head = false;
    if (const char* tm = std::getenv("OFLM_OPEN_TIMEOUT_MS")) cfg.timeout_ms = static_cast<unsigned>(std::strtoul(tm, nullptr, 10));
    impl_->core = std::make_unique<open_qwen36::Core>(cfg, npu_device_inst);
    impl_->core->load_weights();
    impl_->core->reset();
    impl_->tok = tokenizers::Tokenizer::FromBlobJSON(read_text(dir / "tokenizer.json"));
    if (!impl_->tok) throw std::runtime_error("OpenLlmEmbedding: tokenizer.json in " + dir.string() + " did not load");
    if (const auto* g = dynamic_cast<const open_qwen36::GgufFile*>(&impl_->core->file())) {
        impl_->add_eos = g->kv_has("tokenizer.ggml.add_eos_token") && g->kv_str("tokenizer.ggml.add_eos_token") == "1";
        if (g->kv_has("tokenizer.ggml.eos_token_id")) impl_->eos = static_cast<int>(g->kv_u64("tokenizer.ggml.eos_token_id"));
    }
    if (mode_ == Mode::Embed && (!impl_->add_eos || impl_->eos < 0))
        header_print("WARNING", "OpenLlmEmbedding: the GGUF does not ask for an end-of-text token; texts are pooled at their own last token");
    if (mode_ == Mode::Rerank) {
        const int yes = impl_->tok->TokenToId("yes"), no = impl_->tok->TokenToId("no");
        if (yes < 0 || no < 0) throw std::runtime_error("OpenLlmEmbedding: the tokenizer has no \"yes\" / \"no\" token");
        const size_t h = impl_->core->hidden();
        impl_->yes_row.resize(h);
        impl_->no_row.resize(h);
        impl_->core->file().embed_row("lm_head.weight", static_cast<size_t>(yes), h, impl_->yes_row.data());
        impl_->core->file().embed_row("lm_head.weight", static_cast<size_t>(no), h, impl_->no_row.data());
    }
    this->is_model_loaded = true;
    header_print("OFLM", current_model + (mode_ == Mode::Rerank ? " (reranker)" : " (embedder)") + " on the open kernels (" +
                             cfg.kernel_dir + "), hidden " + std::to_string(impl_->core->hidden()) + ", context " +
                             std::to_string(cfg.max_ctx));
}

std::vector<float> OpenLlmEmbedding::embed(std::string& text, embedding_task_type_t task_type) {
    if (mode_ == Mode::Rerank)
        throw std::runtime_error("OpenLlmEmbedding: " + current_model + " is a reranker; it answers /v1/rerank");
    if (!impl_->core) throw std::runtime_error("OpenLlmEmbedding: embed() before load_model()");
    const std::string prompt = llm_conventions::qwen3_embed_text(task_type, text);
    std::vector<float> h = impl_->hidden(llm_conventions::pooled_ids(impl_->encode(prompt), impl_->add_eos, impl_->eos));
    llm_conventions::l2_normalize(h);
    return h;
}

std::vector<float> OpenLlmEmbedding::embed_batch(const std::vector<std::string>& texts, embedding_task_type_t task_type,
                                                 int64_t* tokens) {
    if (mode_ == Mode::Rerank)
        throw std::runtime_error("OpenLlmEmbedding: " + current_model + " is a reranker; it answers /v1/rerank");
    if (!impl_->core) throw std::runtime_error("OpenLlmEmbedding: embed_batch() before load_model()");
    std::vector<float> out;
    int64_t n = 0;
    for (const std::string& t : texts) {
        const std::string prompt = llm_conventions::qwen3_embed_text(task_type, t);
        const std::vector<int> ids = llm_conventions::pooled_ids(impl_->encode(prompt), impl_->add_eos, impl_->eos);
        n += static_cast<int64_t>(ids.size());
        std::vector<float> h = impl_->hidden(ids);
        llm_conventions::l2_normalize(h);
        out.insert(out.end(), h.begin(), h.end());
    }
    if (tokens) *tokens = n;
    return out;
}

size_t OpenLlmEmbedding::embedding_dim() const { return impl_->core ? impl_->core->hidden() : 0; }

size_t OpenLlmEmbedding::min_dimensions() const { return mode_ == Mode::Embed ? 32 : 0; }

std::vector<double> OpenLlmEmbedding::rerank(const std::string& query, const std::vector<std::string>& documents,
                                             const std::string& instruction, int64_t* tokens) {
    if (mode_ != Mode::Rerank)
        throw std::runtime_error("OpenLlmEmbedding: " + current_model + " is an embedder, not a reranker");
    if (!impl_->core) throw std::runtime_error("OpenLlmEmbedding: rerank() before load_model()");
    std::vector<double> scores;
    int64_t n = 0;
    for (const std::string& d : documents) {
        const llm_conventions::RerankPrompt p = llm_conventions::qwen3_rerank_prompt(instruction, query, d);
        std::vector<int> ids = impl_->encode(p.prefix);
        for (const auto& part : {p.body, p.suffix}) {
            const std::vector<int> v = impl_->encode(part);
            ids.insert(ids.end(), v.begin(), v.end());
        }
        n += static_cast<int64_t>(ids.size());
        const std::vector<float> h = impl_->hidden(ids);
        scores.push_back(llm_conventions::yes_no_score(dot(h, impl_->yes_row), dot(h, impl_->no_row)));
    }
    if (tokens) *tokens = n;
    return scores;
}

#else  // no open_qwen36 in this build (the HRX backend)

struct OpenLlmEmbedding::Impl {};

OpenLlmEmbedding::OpenLlmEmbedding(oflm_rt::device* npu_device_inst, std::string tag, Mode mode)
    : AutoEmbeddingModel(npu_device_inst, std::move(tag)), impl_(new Impl), mode_(mode) {}
OpenLlmEmbedding::~OpenLlmEmbedding() = default;
void OpenLlmEmbedding::load_model(std::string, json, bool) {
    throw std::runtime_error("OpenLlmEmbedding: this build has no open_qwen36 engine");
}
std::vector<float> OpenLlmEmbedding::embed(std::string&, embedding_task_type_t) { return {}; }
std::vector<float> OpenLlmEmbedding::embed_batch(const std::vector<std::string>&, embedding_task_type_t, int64_t*) { return {}; }
size_t OpenLlmEmbedding::embedding_dim() const { return 0; }
size_t OpenLlmEmbedding::min_dimensions() const { return 0; }
std::vector<double> OpenLlmEmbedding::rerank(const std::string&, const std::vector<std::string>&, const std::string&, int64_t*) {
    return {};
}

#endif
