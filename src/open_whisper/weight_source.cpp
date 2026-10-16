//===- weight_source.cpp --------------------------------------*- C++ -*-===//
// open_whisper -- see weight_source.hpp. SPDX-License-Identifier: MIT
#include "weight_source.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <tuple>

#include "decoder.hpp"
#include "kernels.hpp"
#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/q4nx_file.hpp"
#include "weights.hpp"

namespace ow {
namespace {

using Shape = std::vector<size_t>;

std::string shape_str(const Shape &s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
  return "[" + out + "]";
}

Shape dims(std::initializer_list<int64_t> d) {
  Shape s;
  for (int64_t v : d) s.push_back(static_cast<size_t>(v));
  return s;
}

size_t count(const Shape &s) {
  size_t n = 1;
  for (size_t v : s) n *= v;
  return n;
}

bool starts_with(const std::string &s, const std::string &prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

uint16_t round_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof u);
  return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// Every container tensor with its dtype and shape: whisper_tensors() in
// utilities/q4nx-build/q4nx/open_whisper.py, the layout OPEN-WHISPER-CONTAINER states.
std::map<std::string, TensorDesc> container_layout(const ModelShape &s) {
  const int64_t D = s.d_model, F = s.ffn, X = 2 * s.n_dec * s.d_model;
  std::map<std::string, TensorDesc> m;
  auto bf = [&](const std::string &n, Shape sh) { m[n] = {"BF16", std::move(sh)}; };
  auto f32 = [&](const std::string &n, Shape sh) { m[n] = {"F32", std::move(sh)}; };
  bf("enc.conv1.B", dims({3 * s.n_mel, D}));
  f32("enc.conv1.bias", dims({D}));
  bf("enc.conv2.B", dims({3 * D, D}));
  f32("enc.conv2.bias", dims({D}));
  f32("enc.pos", dims({s.max_src, D}));
  f32("enc.ln.w", dims({D}));
  f32("enc.ln.b", dims({D}));
  for (int64_t i = 0; i < s.n_enc; ++i) {
    const std::string q = "enc." + std::to_string(i) + ".";
    bf(q + "qkv.B", dims({D, 3 * D}));
    f32(q + "qkv.bias", dims({3 * D}));
    bf(q + "o.B", dims({D, D}));
    f32(q + "o.bias", dims({D}));
    bf(q + "fc1.B", dims({D, F}));
    f32(q + "fc1.bias", dims({F}));
    bf(q + "fc2.B", dims({F, D}));
    f32(q + "fc2.bias", dims({D}));
    for (const char *n : {"ln1.w", "ln1.b", "ln2.w", "ln2.b"}) f32(q + n, dims({D}));
  }
  bf("dec.xkv.B", dims({D, X}));
  f32("dec.xkv.bias", dims({X}));
  bf("decoder.embed_tokens.weight", dims({s.vocab, D}));
  f32("decoder.embed_positions.weight", dims({s.max_tgt, D}));
  f32("decoder.layer_norm.weight", dims({D}));
  f32("decoder.layer_norm.bias", dims({D}));
  for (int64_t l = 0; l < s.n_dec; ++l) {
    const std::string p = "decoder.layers." + std::to_string(l) + ".";
    for (const char *n : {"q", "k", "v", "out"}) bf(p + "self_attn." + n + "_proj.weight", dims({D, D}));
    for (const char *n : {"q", "v", "out"}) f32(p + "self_attn." + n + "_proj.bias", dims({D}));
    // encoder_attn's k_proj/v_proj live only in dec.xkv
    for (const char *n : {"q", "out"}) {
      bf(p + "encoder_attn." + n + "_proj.weight", dims({D, D}));
      f32(p + "encoder_attn." + n + "_proj.bias", dims({D}));
    }
    bf(p + "fc1.weight", dims({F, D}));
    f32(p + "fc1.bias", dims({F}));
    bf(p + "fc2.weight", dims({D, F}));
    f32(p + "fc2.bias", dims({D}));
    for (const char *n : {"self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"}) {
      f32(p + n + ".weight", dims({D}));
      f32(p + n + ".bias", dims({D}));
    }
  }
  return m;
}

// The transformers tensors the container is made from, with their shapes.
std::map<std::string, Shape> source_layout(const ModelShape &s) {
  const int64_t D = s.d_model, F = s.ffn;
  std::map<std::string, Shape> m;
  const std::string e = "model.encoder.", d = "model.decoder.";
  m[e + "conv1.weight"] = dims({D, s.n_mel, 3});
  m[e + "conv1.bias"] = dims({D});
  m[e + "conv2.weight"] = dims({D, D, 3});
  m[e + "conv2.bias"] = dims({D});
  m[e + "embed_positions.weight"] = dims({s.max_src, D});
  m[e + "layer_norm.weight"] = dims({D});
  m[e + "layer_norm.bias"] = dims({D});
  m[d + "embed_tokens.weight"] = dims({s.vocab, D});
  m[d + "embed_positions.weight"] = dims({s.max_tgt, D});
  m[d + "layer_norm.weight"] = dims({D});
  m[d + "layer_norm.bias"] = dims({D});
  auto block = [&](const std::string &p, std::initializer_list<const char *> attns,
                   std::initializer_list<const char *> norms) {
    for (const char *a : attns)
      for (const char *n : {"q", "k", "v", "out"}) {
        m[p + a + "." + n + "_proj.weight"] = dims({D, D});
        if (std::strcmp(n, "k") != 0) m[p + a + "." + n + "_proj.bias"] = dims({D});   // k_proj has no bias
      }
    for (const char *n : norms) {
      m[p + n + ".weight"] = dims({D});
      m[p + n + ".bias"] = dims({D});
    }
    m[p + "fc1.weight"] = dims({F, D});
    m[p + "fc1.bias"] = dims({F});
    m[p + "fc2.weight"] = dims({D, F});
    m[p + "fc2.bias"] = dims({D});
  };
  for (int64_t i = 0; i < s.n_enc; ++i)
    block(e + "layers." + std::to_string(i) + ".", {"self_attn"},
          {"self_attn_layer_norm", "final_layer_norm"});
  for (int64_t l = 0; l < s.n_dec; ++l)
    block(d + "layers." + std::to_string(l) + ".", {"self_attn", "encoder_attn"},
          {"self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"});
  return m;
}

// out[c * ld + col0 + r] = in[r * cols + c]: `in` [rows, cols] lands transposed in
// columns [col0, col0 + rows) of the row-major [cols, ld] matrix `out`.
void transpose_into(const float *in, size_t rows, size_t cols, float *out, size_t ld, size_t col0) {
  constexpr size_t B = 64;
  for (size_t r0 = 0; r0 < rows; r0 += B)
    for (size_t c0 = 0; c0 < cols; c0 += B) {
      const size_t r1 = std::min(rows, r0 + B), c1 = std::min(cols, c0 + B);
      for (size_t r = r0; r < r1; ++r)
        for (size_t c = c0; c < c1; ++c) out[c * ld + col0 + r] = in[r * cols + c];
    }
}

// Conv1d weight [C_out, C_in, 3] -> im2col operand [3 * C_in, C_out], K = tap * C_in + c.
std::vector<float> conv_operand(const std::vector<float> &w, size_t c_out, size_t c_in) {
  std::vector<float> out(3 * c_in * c_out);
  for (size_t o = 0; o < c_out; ++o)
    for (size_t c = 0; c < c_in; ++c)
      for (size_t t = 0; t < 3; ++t) out[(t * c_in + c) * c_out + o] = w[(o * c_in + c) * 3 + t];
  return out;
}

// Transformers-named tensors as f32, from one file.
class SourceTensors {
public:
  virtual ~SourceTensors() = default;
  virtual bool has(const std::string &hf) const = 0;
  virtual Shape shape(const std::string &hf) const = 0;
  virtual std::vector<float> read(const std::string &hf) const = 0;
  virtual std::string type_of(const std::string &hf) const = 0;
};

class StockTensors final : public SourceTensors {
public:
  explicit StockTensors(const std::string &path) : f_(path) {}
  bool has(const std::string &hf) const override { return f_.has(hf); }
  Shape shape(const std::string &hf) const override { return f_.meta(hf).shape; }
  std::string type_of(const std::string &hf) const override { return f_.meta(hf).dtype; }
  std::vector<float> read(const std::string &hf) const override {
    const auto &m = f_.meta(hf);
    size_t nbytes = 0;
    const uint8_t *p = f_.raw(hf, &nbytes);
    const size_t n = count(m.shape);
    const size_t width = m.dtype == "F32" ? 4 : (m.dtype == "F16" || m.dtype == "BF16") ? 2 : 0;
    if (!width)
      throw std::runtime_error("model.safetensors: '" + hf + "' is " + m.dtype +
                               "; this reader takes F16, BF16 or F32");
    if (nbytes != n * width)
      throw std::runtime_error("model.safetensors: '" + hf + "' holds " + std::to_string(nbytes) +
                               " bytes, its shape " + shape_str(m.shape) + " needs " +
                               std::to_string(n * width));
    std::vector<float> out(n);
    if (width == 4) {
      std::memcpy(out.data(), p, n * 4);
      return out;
    }
    const bool f16 = m.dtype == "F16";
    for (size_t i = 0; i < n; ++i) {
      uint16_t u;
      std::memcpy(&u, p + 2 * i, 2);
      out[i] = f16 ? open_qwen36::fp16_to_f32(u) : open_qwen36::bf16_to_f32(u);
    }
    return out;
  }

private:
  open_qwen36::Q4nxFile f_;
};

class GgufTensors final : public SourceTensors {
public:
  GgufTensors(const std::string &path, const ModelShape &s) : path_(path), g_(path) {
    const std::string arch =
        g_.kv_has("general.architecture") ? g_.kv_str("general.architecture") : std::string();
    if (arch != "whisper")
      throw std::runtime_error(path + ": general.architecture is '" + arch + "', not 'whisper'");
    auto want = [&](const char *key, int64_t v) {
      if (!g_.kv_has(key))
        throw std::runtime_error(path + ": no " + key +
                                 " -- not the stt.whisper GGUF layout this reader knows");
      const uint64_t got = g_.kv_u64(key);
      if (got != static_cast<uint64_t>(v))
        throw std::runtime_error(path + ": " + key + " is " + std::to_string(got) +
                                 ", this engine implements " + std::to_string(v));
    };
    want("stt.whisper.encoder.n_layers", s.n_enc);
    want("stt.whisper.decoder.n_layers", s.n_dec);
    want("stt.whisper.encoder.d_model", s.d_model);
    want("stt.whisper.decoder.d_model", s.d_model);
    want("stt.whisper.encoder.n_heads", s.n_heads);
    want("stt.whisper.decoder.n_heads", s.n_heads);
    want("stt.whisper.encoder.ffn_dim", s.ffn);
    want("stt.whisper.decoder.ffn_dim", s.ffn);
    want("stt.whisper.encoder.num_mel_bins", s.n_mel);
    want("stt.whisper.encoder.max_source_positions", s.max_src);
    want("stt.whisper.decoder.max_target_positions", s.max_tgt);
    want("stt.whisper.decoder.vocab_size", s.vocab);
    // The decoder adds the raw embedding and reuses it as the output head.
    if (g_.kv_has("stt.whisper.decoder.scale_embedding") && g_.kv_u64("stt.whisper.decoder.scale_embedding"))
      throw std::runtime_error(path + ": stt.whisper.decoder.scale_embedding is true; this decoder does not scale");
    if (g_.kv_has("stt.whisper.decoder.tie_word_embeddings") &&
        !g_.kv_u64("stt.whisper.decoder.tie_word_embeddings"))
      throw std::runtime_error(path + ": the output head is not tied to the token embedding");
  }
  bool has(const std::string &hf) const override {
    const std::string n = gguf_tensor_name(hf);
    return !n.empty() && g_.has(n);
  }
  Shape shape(const std::string &hf) const override { return g_.meta(gguf_tensor_name(hf)).shape; }
  std::string type_of(const std::string &hf) const override {
    return open_qwen36::GgufFile::type_name(g_.tensor(gguf_tensor_name(hf)).type);
  }
  std::vector<float> read(const std::string &hf) const override { return g_.f32(gguf_tensor_name(hf)); }

private:
  std::string path_;
  open_qwen36::GgufFile g_;
};

// A stock checkpoint or a GGUF, presented as the container.
class DerivedSource final : public WeightSource {
public:
  DerivedSource(WeightKind kind, const std::string &path, const ModelShape &s)
      : kind_(kind), file_(weight_file_name(kind)), s_(s), layout_(container_layout(s)),
        sources_(source_layout(s)) {
    if (kind == WeightKind::Gguf) src_ = std::make_unique<GgufTensors>(path, s);
    else src_ = std::make_unique<StockTensors>(path);
    std::vector<std::string> missing;
    std::map<std::string, int> types;
    for (const auto &kv : sources_) {
      if (!src_->has(kv.first)) {
        missing.push_back(kind == WeightKind::Gguf ? gguf_tensor_name(kv.first) : kv.first);
        continue;
      }
      const Shape got = src_->shape(kv.first);
      if (got != kv.second)
        throw std::runtime_error(file_ + ": '" + kv.first + "' has shape " + shape_str(got) +
                                 ", expected " + shape_str(kv.second));
      ++types[src_->type_of(kv.first)];
    }
    if (!missing.empty())
      throw std::runtime_error(file_ + ": " + std::to_string(missing.size()) +
                               " tensors missing, the first '" + missing.front() + "'");
    for (const auto &kv : types)
      types_ += (types_.empty() ? "" : ", ") + kv.first + " x" + std::to_string(kv.second);
  }

  WeightKind kind() const override { return kind_; }
  std::string describe() const override {
    if (kind_ == WeightKind::Gguf)
      return file_ + " (GGUF: " + types_ + "; widened to the container's bf16/f32 at load)";
    return file_ + " (transformers checkpoint: " + types_ + "; rearranged at load)";
  }
  bool has(const std::string &name) const override { return layout_.count(name) != 0; }
  TensorDesc desc(const std::string &name) const override {
    auto it = layout_.find(name);
    if (it == layout_.end()) throw std::runtime_error(file_ + ": missing tensor '" + name + "'");
    return it->second;
  }
  std::vector<uint16_t> bf16(const std::string &name) const override {
    if (desc(name).dtype != "BF16")
      throw std::runtime_error(file_ + ": '" + name + "' is F32, expected BF16");
    const std::vector<float> v = values(name);
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = round_bf16(v[i]);
    return out;
  }
  std::vector<float> f32(const std::string &name) const override {
    if (desc(name).dtype != "F32")
      throw std::runtime_error(file_ + ": '" + name + "' is BF16, expected F32");
    return values(name);
  }

private:
  std::vector<float> read(const std::string &hf) const { return src_->read(hf); }

  // `name`'s values in f32, before the container's bf16 rounding.
  std::vector<float> values(const std::string &name) const {
    const size_t D = static_cast<size_t>(s_.d_model), F = static_cast<size_t>(s_.ffn);
    const size_t MEL = static_cast<size_t>(s_.n_mel), NDEC = static_cast<size_t>(s_.n_dec);
    const std::string e = "model.encoder.";
    if (name == "enc.conv1.B") return conv_operand(read(e + "conv1.weight"), D, MEL);
    if (name == "enc.conv2.B") return conv_operand(read(e + "conv2.weight"), D, D);
    if (name == "enc.conv1.bias") return read(e + "conv1.bias");
    if (name == "enc.conv2.bias") return read(e + "conv2.bias");
    if (name == "enc.pos") return read(e + "embed_positions.weight");
    if (name == "enc.ln.w") return read(e + "layer_norm.weight");
    if (name == "enc.ln.b") return read(e + "layer_norm.bias");
    if (name == "dec.xkv.B") {
      std::vector<float> out(D * 2 * NDEC * D);
      for (size_t l = 0; l < NDEC; ++l)
        for (size_t kv = 0; kv < 2; ++kv) {
          const std::vector<float> w = read("model.decoder.layers." + std::to_string(l) +
                                            ".encoder_attn." + (kv ? "v" : "k") + "_proj.weight");
          transpose_into(w.data(), D, D, out.data(), 2 * NDEC * D, (2 * l + kv) * D);
        }
      return out;
    }
    if (name == "dec.xkv.bias") {
      std::vector<float> out(2 * NDEC * D, 0.f);   // k_proj has no bias: zero thirds
      for (size_t l = 0; l < NDEC; ++l) {
        const std::vector<float> b =
            read("model.decoder.layers." + std::to_string(l) + ".encoder_attn.v_proj.bias");
        std::copy(b.begin(), b.end(), out.begin() + static_cast<std::ptrdiff_t>((2 * l + 1) * D));
      }
      return out;
    }
    if (starts_with(name, "decoder.")) return read("model." + name);
    if (starts_with(name, "enc.")) {
      const size_t dot = name.find('.', 4);
      const std::string layer = name.substr(4, dot - 4), field = name.substr(dot + 1);
      const std::string p = e + "layers." + layer + ".";
      if (field == "qkv.B") {
        std::vector<float> out(D * 3 * D);
        const char *proj[] = {"q", "k", "v"};
        for (size_t j = 0; j < 3; ++j) {
          const std::vector<float> w = read(p + "self_attn." + proj[j] + "_proj.weight");
          transpose_into(w.data(), D, D, out.data(), 3 * D, j * D);
        }
        return out;
      }
      if (field == "qkv.bias") {
        std::vector<float> out(3 * D, 0.f);        // q | 0 | v: k_proj has no bias
        const std::vector<float> q = read(p + "self_attn.q_proj.bias");
        const std::vector<float> v = read(p + "self_attn.v_proj.bias");
        std::copy(q.begin(), q.end(), out.begin());
        std::copy(v.begin(), v.end(), out.begin() + static_cast<std::ptrdiff_t>(2 * D));
        return out;
      }
      auto transposed = [&](const std::string &hf, size_t rows, size_t cols) {
        const std::vector<float> w = read(hf);
        std::vector<float> out(rows * cols);
        transpose_into(w.data(), rows, cols, out.data(), rows, 0);
        return out;
      };
      if (field == "o.B") return transposed(p + "self_attn.out_proj.weight", D, D);
      if (field == "o.bias") return read(p + "self_attn.out_proj.bias");
      if (field == "fc1.B") return transposed(p + "fc1.weight", F, D);
      if (field == "fc1.bias") return read(p + "fc1.bias");
      if (field == "fc2.B") return transposed(p + "fc2.weight", D, F);
      if (field == "fc2.bias") return read(p + "fc2.bias");
      if (field == "ln1.w") return read(p + "self_attn_layer_norm.weight");
      if (field == "ln1.b") return read(p + "self_attn_layer_norm.bias");
      if (field == "ln2.w") return read(p + "final_layer_norm.weight");
      if (field == "ln2.b") return read(p + "final_layer_norm.bias");
    }
    throw std::runtime_error(file_ + ": no rule makes container tensor '" + name + "'");
  }

  WeightKind kind_;
  std::string file_;
  ModelShape s_;
  std::map<std::string, TensorDesc> layout_;
  std::map<std::string, Shape> sources_;
  std::unique_ptr<SourceTensors> src_;
  std::string types_;
};

// model.open.safetensors as q4nx-build wrote it: read, never rearranged.
class ContainerSource final : public WeightSource {
public:
  explicit ContainerSource(const std::string &path) : f_(path) {}
  WeightKind kind() const override { return WeightKind::Container; }
  std::string describe() const override { return "model.open.safetensors (q4nx-build's container)"; }
  bool has(const std::string &name) const override { return f_.has(name); }
  TensorDesc desc(const std::string &name) const override {
    if (!f_.has(name))
      throw std::runtime_error("model.open.safetensors: missing tensor '" + name + "'");
    const auto &m = f_.meta(name);
    return {m.dtype, m.shape};
  }
  std::vector<uint16_t> bf16(const std::string &name) const override {
    require_bf16(f_, name);
    size_t nbytes = 0;
    const uint8_t *raw = f_.raw(name, &nbytes);
    const size_t n = count(f_.meta(name).shape);
    if (nbytes != n * 2)
      throw std::runtime_error("model.open.safetensors: '" + name + "' is " + std::to_string(nbytes) +
                               " bytes, expected " + std::to_string(n * 2) + " (bf16)");
    std::vector<uint16_t> v(n);
    std::memcpy(v.data(), raw, n * 2);
    return v;
  }
  std::vector<float> f32(const std::string &name) const override { return f_.f32(name); }

private:
  open_qwen36::Q4nxFile f_;
};

}  // namespace

ModelShape turbo_shape() {
  return {Geometry::d_model,     Geometry::n_enc,   DecoderGeometry::n_layers,
          Geometry::n_heads,     Geometry::ffn,     Geometry::n_mel,
          Geometry::max_src_pos, DecoderGeometry::max_target_positions,
          DecoderGeometry::vocab};
}

const char *weight_file_name(WeightKind kind) {
  switch (kind) {
    case WeightKind::Container: return "model.open.safetensors";
    case WeightKind::Stock: return "model.safetensors";
    case WeightKind::Gguf: return "model.gguf";
  }
  return "?";
}

std::vector<WeightKind> weights_present(const std::string &model_dir) {
  std::vector<WeightKind> out;
  for (WeightKind k : {WeightKind::Container, WeightKind::Stock, WeightKind::Gguf}) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(std::filesystem::path(model_dir) / weight_file_name(k), ec))
      out.push_back(k);
  }
  return out;
}

std::unique_ptr<WeightSource> open_weight_file(const std::string &path, WeightKind kind,
                                               const ModelShape &shape) {
  if (kind == WeightKind::Container) return std::make_unique<ContainerSource>(path);
  return std::make_unique<DerivedSource>(kind, path, shape);
}

std::unique_ptr<WeightSource> open_weight_source(const std::string &model_dir, const ModelShape &shape) {
  const std::vector<WeightKind> present = weights_present(model_dir);
  if (present.empty())
    throw std::runtime_error("no Whisper weights in " + model_dir +
                             ": expected model.open.safetensors, model.safetensors or model.gguf");
  return open_weight_file(model_dir + "/" + weight_file_name(present.front()), present.front(), shape);
}

std::vector<std::string> container_tensor_names(const ModelShape &shape) {
  std::vector<std::string> out;
  for (const auto &kv : container_layout(shape)) out.push_back(kv.first);
  return out;
}

std::string gguf_tensor_name(const std::string &hf) {
  static const std::map<std::string, std::string> top = {
      {"model.encoder.conv1", "enc.conv.0"},
      {"model.encoder.conv2", "enc.conv.1"},
      {"model.encoder.embed_positions", "enc.pos_emb"},
      {"model.encoder.layer_norm", "enc.final_norm"},
      {"model.decoder.embed_tokens", "dec.token_embd"},
      {"model.decoder.embed_positions", "dec.pos_emb"},
      {"model.decoder.layer_norm", "dec.final_norm"}};
  static const std::map<std::string, std::string> enc_block = {
      {"self_attn.q_proj", "attn.q"},   {"self_attn.k_proj", "attn.k"},
      {"self_attn.v_proj", "attn.v"},   {"self_attn.out_proj", "attn.out"},
      {"self_attn_layer_norm", "norm_attn"}, {"final_layer_norm", "norm_ffn"},
      {"fc1", "ffn.fc1"},               {"fc2", "ffn.fc2"}};
  static const std::map<std::string, std::string> dec_block = {
      {"self_attn.q_proj", "self_attn.q"},       {"self_attn.k_proj", "self_attn.k"},
      {"self_attn.v_proj", "self_attn.v"},       {"self_attn.out_proj", "self_attn.out"},
      {"encoder_attn.q_proj", "cross_attn.q"},   {"encoder_attn.k_proj", "cross_attn.k"},
      {"encoder_attn.v_proj", "cross_attn.v"},   {"encoder_attn.out_proj", "cross_attn.out"},
      {"self_attn_layer_norm", "norm_self"},     {"encoder_attn_layer_norm", "norm_cross"},
      {"final_layer_norm", "norm_ffn"},          {"fc1", "ffn.fc1"},
      {"fc2", "ffn.fc2"}};
  const size_t dot = hf.rfind('.');
  if (dot == std::string::npos) return {};
  const std::string stem = hf.substr(0, dot), kind = hf.substr(dot + 1);
  if (kind != "weight" && kind != "bias") return {};
  if (auto it = top.find(stem); it != top.end()) return it->second + "." + kind;
  for (const auto &[side, pre, table] :
       {std::make_tuple(std::string("model.encoder.layers."), std::string("enc.blocks."), &enc_block),
        std::make_tuple(std::string("model.decoder.layers."), std::string("dec.blocks."), &dec_block)}) {
    if (!starts_with(stem, side)) continue;
    const std::string rest = stem.substr(side.size());
    const size_t d = rest.find('.');
    if (d == std::string::npos || d == 0) return {};
    const std::string idx = rest.substr(0, d), layer = rest.substr(d + 1);
    if (!std::all_of(idx.begin(), idx.end(), [](char c) { return c >= '0' && c <= '9'; })) return {};
    auto it = table->find(layer);
    if (it == table->end()) return {};
    return pre + idx + "." + it->second + "." + kind;
  }
  return {};
}

}  // namespace ow
