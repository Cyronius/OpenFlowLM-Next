//===- weight_source.hpp --------------------------------------*- C++ -*-===//
//
// open_whisper -- the container's tensors (OPEN-WHISPER-CONTAINER: the same
// names, dtypes, shapes and bits) from whichever weight file a model
// directory holds: the model.open.safetensors q4nx-build writes, OpenAI's own
// model.safetensors (OPEN-WHISPER-STOCK), or a GGUF of the model
// (OPEN-WHISPER-GGUF). The last two are rearranged at load the way
// q4nx-build's whisper_tensors() does it, so Weights and Decoder see the same
// bits whichever file is there.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ow {

// The shape a weight file is read against: whisper-large-v3-turbo's in the
// engine (turbo_shape()), a tiny one in weight_source_test.
struct ModelShape {
  int64_t d_model, n_enc, n_dec, n_heads, ffn, n_mel, max_src, max_tgt, vocab;
};
ModelShape turbo_shape();

enum class WeightKind { Container, Stock, Gguf };

// The file each kind is read from, inside a model directory.
const char *weight_file_name(WeightKind kind);

// The weight files `model_dir` holds, in the order the engine prefers them:
// the container (built for this engine), then OpenAI's 16-bit checkpoint, then
// a GGUF. Empty when there are none.
std::vector<WeightKind> weights_present(const std::string &model_dir);

struct TensorDesc {
  std::string dtype;            // "BF16" or "F32"
  std::vector<size_t> shape;
};

class WeightSource {
public:
  virtual ~WeightSource() = default;
  virtual WeightKind kind() const = 0;
  // The file and, for a GGUF, its tensor types -- for the load log.
  virtual std::string describe() const = 0;
  virtual bool has(const std::string &name) const = 0;
  // Throws when the tensor is absent.
  virtual TensorDesc desc(const std::string &name) const = 0;
  // A BF16 tensor's bits. Throws unless the tensor is BF16.
  virtual std::vector<uint16_t> bf16(const std::string &name) const = 0;
  // An F32 tensor. Throws unless the tensor is F32.
  virtual std::vector<float> f32(const std::string &name) const = 0;
};

// Opens `path` as `kind`, checked against `shape` (a GGUF's metadata and every
// source tensor's shape). The container is read as it is; its own header
// carries the shapes Weights and Decoder check.
std::unique_ptr<WeightSource> open_weight_file(const std::string &path, WeightKind kind,
                                               const ModelShape &shape);

// The first file weights_present() finds in `model_dir`; throws when there is none.
std::unique_ptr<WeightSource> open_weight_source(const std::string &model_dir,
                                                 const ModelShape &shape);

// Every tensor the container holds for `shape`, sorted (481 for turbo).
std::vector<std::string> container_tensor_names(const ModelShape &shape);

// A transformers tensor name -> its name in handy-computer's GGUF of the model
// ("model.encoder.layers.3.fc1.weight" -> "enc.blocks.3.ffn.fc1.weight"); empty
// when the name has no counterpart.
std::string gguf_tensor_name(const std::string &hf_name);

}  // namespace ow
