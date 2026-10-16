// open_whisper: the container's tensors from model.open.safetensors, OpenAI's model.safetensors or a GGUF (2026-10-16). SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ow {

struct ModelShape {
  int64_t d_model, n_enc, n_dec, n_heads, ffn, n_mel, max_src, max_tgt, vocab;
};
ModelShape turbo_shape();

enum class WeightKind { Container, Stock, Gguf };

const char *weight_file_name(WeightKind kind);

// The converted container wins, then the 16-bit checkpoint, then a GGUF.
std::vector<WeightKind> weights_present(const std::string &model_dir);

struct TensorDesc {
  std::string dtype;            // "BF16" or "F32"
  std::vector<size_t> shape;
};

class WeightSource {
public:
  virtual ~WeightSource() = default;
  virtual WeightKind kind() const = 0;
  virtual std::string describe() const = 0;
  virtual bool has(const std::string &name) const = 0;
  virtual TensorDesc desc(const std::string &name) const = 0;
  virtual std::vector<uint16_t> bf16(const std::string &name) const = 0;
  virtual std::vector<float> f32(const std::string &name) const = 0;
};

std::unique_ptr<WeightSource> open_weight_file(const std::string &path, WeightKind kind,
                                               const ModelShape &shape);
std::unique_ptr<WeightSource> open_weight_source(const std::string &model_dir,
                                                 const ModelShape &shape);
std::vector<std::string> container_tensor_names(const ModelShape &shape);

// Empty when the transformers name has no tensor in the stt.whisper GGUF layout.
std::string gguf_tensor_name(const std::string &hf_name);

}  // namespace ow
