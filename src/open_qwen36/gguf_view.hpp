/// \file gguf_view.hpp
/// \brief A GGUF served as the `.q4nx` container its kernel set's plan was written for (OPEN-GGUF-Q4NX-VIEW).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/weight_file.hpp"

namespace open_qwen36 {

/// The config.json fields a kernel set checks, from a GGUF's metadata (#24; qwen35: OPEN-GGUF-QWEN35-SPEC).
nlohmann::json derive_config(const GgufFile& g);
/// Transformer blocks: block_count less the trailing MTP blocks `nextn_predict_layers` declares.
uint64_t gguf_layer_count(const GgufFile& g);

struct GgufQwen35Geometry {
    uint64_t hidden = 0, heads = 0, kv_heads = 0, head_dim = 0;
    uint64_t lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv = 0;
    uint64_t blocks = 0;
};

class GgufQ4nxView final : public WeightFile {
public:
    explicit GgufQ4nxView(const std::string& path);
    explicit GgufQ4nxView(std::unique_ptr<GgufFile> g);
    static bool supports(const std::string& arch);

    bool has(const std::string& name) const override;
    const uint8_t* raw(const std::string& name, size_t* nbytes = nullptr) const override;
    size_t chunk_bytes(const std::string& name) const override;
    void embed_row(const std::string& name, size_t row, size_t dim, float* out) const override;
    std::vector<float> bf16(const std::string& name) const override;
    std::vector<float> f32(const std::string& name) const override;
    const TensorMeta& meta(const std::string& name) const override;
    void bf16_row(const std::string& name, size_t row, size_t dim, float* out) const override;
    void drop_pages() override;

    const GgufFile& gguf() const { return *g_; }
    const GgufQwen35Geometry& geometry() const { return geo_; }
    std::string source_name(const std::string& name) const;

private:
    struct Tensor {
        TensorMeta meta;
        std::vector<uint8_t> bytes;
        uint64_t used = 0;
    };
    const Tensor& get(const std::string& name) const;
    Tensor build(const std::string& name) const;

    std::unique_ptr<GgufFile> g_;
    GgufQwen35Geometry geo_;
    mutable std::map<std::string, Tensor> cache_;
    mutable uint64_t clock_ = 0;
};

namespace gguf_view {

size_t to_q4nx_chunks(const GgufFile& g, const std::string& gguf_name, uint64_t rows, uint64_t cols,
                      const std::vector<uint64_t>* row_of, const std::vector<uint64_t>* blk_of,
                      bool head, std::vector<uint8_t>& out);
void requant_q8_block(const float* v, uint16_t* scale, int8_t* codes);
uint64_t file_head(uint64_t h, uint64_t num_k, uint64_t num_v);

}  // namespace gguf_view
}  // namespace open_qwen36
