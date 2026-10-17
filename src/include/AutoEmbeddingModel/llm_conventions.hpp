/// \file llm_conventions.hpp
/// \brief Text and score rules of the decoder-LLM embedder and reranker, testable without a device.
/// \note Traces: SERVER-EMBED-QWEN3, SERVER-RERANK (specs/server-api/spec.md)
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "auto_embedding_model.hpp"

namespace llm_conventions {

/// The instruction Qwen's cards use when a caller gives none (both models).
inline const char* kWebSearchInstruction =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Qwen3-Embedding: a query carries the instruction, a document is embedded as written.
inline std::string qwen3_embed_text(embedding_task_type_t task, const std::string& text,
                                    const std::string& instruction = kWebSearchInstruction) {
    switch (task) {
        case task_query:
            return "Instruct: " + instruction + "\nQuery:" + text;
        case task_document:
        case task_search_result:
            return text;
        default:
            throw TaskPromptUnavailable(
                "this model declares task prompts [query, document] and none of them matches the requested "
                "task. Refusing to pick one: a wrongly-prefixed embedding is correctly shaped and correctly "
                "normed, so nothing downstream could tell the answer is for a different task.");
    }
}

/// The tokenizer's ids, then the end-of-text id when the GGUF says add_eos_token: the pooling reads its state.
inline std::vector<int> pooled_ids(std::vector<int> ids, bool add_eos, int eos) {
    if (add_eos) ids.push_back(eos);
    return ids;
}

/// Qwen3-Reranker's prompt, in the three pieces its card tokenizes separately.
struct RerankPrompt {
    std::string prefix, body, suffix;
    std::string whole() const { return prefix + body + suffix; }
};

inline RerankPrompt qwen3_rerank_prompt(const std::string& instruction, const std::string& query,
                                        const std::string& document) {
    RerankPrompt p;
    p.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
               "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
    p.body = "<Instruct>: " + (instruction.empty() ? std::string(kWebSearchInstruction) : instruction) +
             "\n<Query>: " + query + "\n<Document>: " + document;
    p.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return p;
}

/// The relevance score: the softmax over the (no, yes) pair at the last position, i.e. sigmoid(yes - no).
inline double yes_no_score(double logit_yes, double logit_no) {
    return 1.0 / (1.0 + std::exp(logit_no - logit_yes));
}

inline void l2_normalize(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    if (s <= 0.0) throw std::runtime_error("embedding has zero norm");
    const double inv = 1.0 / std::sqrt(s);
    for (float& x : v) x = static_cast<float>(x * inv);
}

/// Matryoshka truncation: the first `dims` entries, scaled back to unit length.
inline std::vector<float> truncate_renormalize(const std::vector<float>& v, size_t dims) {
    if (dims == 0 || dims > v.size())
        throw std::runtime_error("dimensions " + std::to_string(dims) + " is outside 1.." + std::to_string(v.size()));
    std::vector<float> out(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(dims));
    l2_normalize(out);
    return out;
}

}  // namespace llm_conventions
