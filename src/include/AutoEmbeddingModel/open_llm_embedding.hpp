/// \file open_llm_embedding.hpp
/// \brief OpenLlmEmbedding: embeddings and rerank scores from a decoder LLM on the open kernels (open_qwen36).
/// \note Traces: OPEN-HIDDEN-STATE (specs/open-engine), SERVER-EMBED-QWEN3, SERVER-RERANK (specs/server-api)
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "auto_embedding_model.hpp"

class OpenLlmEmbedding : public AutoEmbeddingModel {
public:
    enum class Mode { Embed, Rerank };

    OpenLlmEmbedding(oflm_rt::device* npu_device_inst, std::string tag, Mode mode);
    ~OpenLlmEmbedding() override;

    /// `model_path` holds model.gguf, tokenizer.json and an open_kernels link (what oflm-add installs).
    void load_model(std::string model_path, json model_info, bool enable_preemption = false) override;

    std::vector<float> embed(std::string& text, embedding_task_type_t task_type) override;
    std::vector<float> embed_batch(const std::vector<std::string>& texts, embedding_task_type_t task_type,
                                   int64_t* tokens = nullptr) override;
    size_t embedding_dim() const override;
    size_t min_dimensions() const override;
    bool supports_task_prompts() const override { return mode_ == Mode::Embed; }
    embedding_task_type_t default_task() const override { return task_document; }
    bool is_reranker() const override { return mode_ == Mode::Rerank; }
    std::vector<double> rerank(const std::string& query, const std::vector<std::string>& documents,
                               const std::string& instruction, int64_t* tokens = nullptr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Mode mode_;
};
