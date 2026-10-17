// Traces: OPEN-WHISPER-GGUF, OPEN-WHISPER-STOCK (canonical spec: specs/open-whisper/spec.md). SPDX-License-Identifier: MIT
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_qwen36/gguf_file.hpp"
#include "weight_source.hpp"

namespace fs = std::filesystem;
using ow::WeightKind;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

void expect_throw(const std::function<void()> &fn, const std::string &needle, const std::string &what) {
  try {
    fn();
  } catch (const std::exception &e) {
    const bool ok = std::string(e.what()).find(needle) != std::string::npos;
    check(ok, what + (ok ? "" : std::string(" (threw: ") + e.what() + ")"));
    return;
  }
  check(false, what + " (did not throw)");
}

std::string fnv1a64(const void *data, size_t n) {
  uint64_t h = 0xCBF29CE484222325ull;
  const auto *p = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

nlohmann::json read_json(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  nlohmann::json j;
  f >> j;
  return j;
}

ow::ModelShape shape_from(const nlohmann::json &j) {
  return {j.at("d_model"), j.at("n_enc"), j.at("n_dec"), j.at("n_heads"), j.at("ffn"),
          j.at("n_mel"), j.at("max_src"), j.at("max_tgt"), j.at("vocab")};
}

void against_oracle(const ow::WeightSource &src, const ow::ModelShape &shape, const nlohmann::json &want,
                    const std::string &label) {
  const std::vector<std::string> names = ow::container_tensor_names(shape);
  std::set<std::string> got_names(names.begin(), names.end()), want_names;
  for (const auto &kv : want.items()) want_names.insert(kv.key());
  check(got_names == want_names, label + ": the converter's " + std::to_string(want_names.size()) +
                                     " tensor names, no more and no fewer");
  size_t same = 0;
  std::string first_bad;
  for (const std::string &n : names) {
    if (!want.contains(n)) continue;
    const nlohmann::json &w = want[n];
    const ow::TensorDesc d = src.desc(n);
    std::string hash;
    if (d.dtype == "BF16") {
      const std::vector<uint16_t> v = src.bf16(n);
      hash = fnv1a64(v.data(), v.size() * 2);
    } else {
      const std::vector<float> v = src.f32(n);
      hash = fnv1a64(v.data(), v.size() * 4);
    }
    const bool ok = d.dtype == w["dtype"].get<std::string>() &&
                    d.shape == w["shape"].get<std::vector<size_t>>() && hash == w["fnv1a64"].get<std::string>();
    if (ok) ++same;
    else if (first_bad.empty()) first_bad = n;
  }
  check(same == names.size(), label + ": " + std::to_string(same) + "/" + std::to_string(names.size()) +
                                  " tensors bit-identical to the converter's" +
                                  (first_bad.empty() ? "" : ", first difference '" + first_bad + "'"));
}

int unit(const fs::path &testdata) {
  const fs::path tiny = testdata / "weights_tiny";
  const nlohmann::json want = read_json(tiny / "expected.json");
  const ow::ModelShape s = shape_from(want["shape"]);

  against_oracle(*ow::open_weight_file((tiny / "model.safetensors").string(), WeightKind::Stock, s), s,
                 want["f16"], "stock checkpoint (F16)");
  against_oracle(*ow::open_weight_file((tiny / "model-F16.gguf").string(), WeightKind::Gguf, s), s,
                 want["f16"], "GGUF F16");
  against_oracle(*ow::open_weight_file((tiny / "model-Q8_0.gguf").string(), WeightKind::Gguf, s), s,
                 want["q8_0"], "GGUF Q8_0");

  ow::ModelShape wrong = s;
  wrong.n_enc += 1;
  expect_throw([&] { ow::open_weight_file((tiny / "model-F16.gguf").string(), WeightKind::Gguf, wrong); },
               "stt.whisper.encoder.n_layers", "a GGUF of another geometry is refused, naming the key");
  wrong = s;
  wrong.ffn *= 2;
  expect_throw([&] { ow::open_weight_file((tiny / "model.safetensors").string(), WeightKind::Stock, wrong); },
               "has shape", "a checkpoint whose tensors have other shapes is refused");
  expect_throw([&] { ow::open_weight_file((tiny / "kquants.gguf").string(), WeightKind::Gguf, s); },
               "stt.whisper", "a GGUF without the stt.whisper metadata is refused");
  for (const auto &c : {std::make_pair("model-F16.gguf", WeightKind::Gguf),
                        std::make_pair("model.safetensors", WeightKind::Stock)}) {
    const fs::path cut = fs::temp_directory_path() / (std::string("ow_weight_source_test_cut_") + c.first);
    fs::copy_file(tiny / c.first, cut, fs::copy_options::overwrite_existing);
    fs::resize_file(cut, fs::file_size(cut) - 64);
    const WeightKind kind = c.second;
    expect_throw([&] { ow::open_weight_file(cut.string(), kind, s); }, "past EOF",
                 std::string(c.first) + " cut short by 64 bytes is refused, not read past its end");
    std::error_code ec;  // Q4nxFile keeps its mapping when its constructor throws, so this delete can fail
    fs::remove(cut, ec);
  }
  {
    const auto src = ow::open_weight_file((tiny / "model-F16.gguf").string(), WeightKind::Gguf, s);
    expect_throw([&] { src->bf16("enc.conv1.bias"); }, "expected BF16", "an F32 tensor is not handed out as bf16");
    expect_throw([&] { src->desc("enc.99.qkv.B"); }, "missing tensor", "an unknown tensor name is refused");
  }

  {
    open_qwen36::GgufFile g((tiny / "kquants.gguf").string());
    std::ifstream f(tiny / "kquants_expected.bin", std::ios::binary);
    std::vector<float> ref(3 * 512);
    f.read(reinterpret_cast<char *>(ref.data()), static_cast<std::streamsize>(ref.size() * 4));
    const char *names[] = {"q4_k", "q5_k", "q6_k"};
    for (int t = 0; t < 3; ++t) {
      const std::vector<float> v = g.f32(names[t]);
      const bool ok = v.size() == 512 && std::memcmp(v.data(), ref.data() + t * 512, 512 * 4) == 0;
      check(ok, std::string(names[t]) + " rows match the gguf package's dequant bit for bit");
    }
  }

  // Checked against the real file's list, not the map itself: a round trip is blind to a wrong entry.
  {
    const nlohmann::json real = read_json(testdata / "whisper_turbo_gguf_tensors.json");
    const ow::ModelShape t = ow::turbo_shape();
    std::set<std::string> mapped;
    size_t bad = 0;
    std::string first_bad;
    const int64_t D = t.d_model;
    std::vector<std::pair<std::string, std::vector<size_t>>> sources = {
        {"model.encoder.conv1.weight", {size_t(D), size_t(t.n_mel), 3}},
        {"model.encoder.conv2.weight", {size_t(D), size_t(D), 3}},
        {"model.encoder.embed_positions.weight", {size_t(t.max_src), size_t(D)}},
        {"model.decoder.embed_tokens.weight", {size_t(t.vocab), size_t(D)}},
        {"model.decoder.embed_positions.weight", {size_t(t.max_tgt), size_t(D)}},
        {"model.decoder.layers.3.fc1.weight", {size_t(t.ffn), size_t(D)}},
        {"model.encoder.layers.31.fc2.weight", {size_t(D), size_t(t.ffn)}},
        {"model.decoder.layers.0.encoder_attn.v_proj.bias", {size_t(D)}}};
    for (const auto &[hf, shape] : sources) {
      const std::string g = ow::gguf_tensor_name(hf);
      if (!real.contains(g) || real[g].get<std::vector<size_t>>() != shape) {
        ++bad;
        if (first_bad.empty()) first_bad = hf;
      }
    }
    check(bad == 0, "the map's names and shapes match the real GGUF" + (first_bad.empty() ? "" : ", first '" + first_bad + "'"));
    for (int64_t i = 0; i < t.n_enc; ++i)
      for (const char *n : {"self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight",
                            "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.out_proj.weight",
                            "self_attn.out_proj.bias", "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
                            "final_layer_norm.weight", "final_layer_norm.bias", "fc1.weight", "fc1.bias",
                            "fc2.weight", "fc2.bias"})
        mapped.insert(ow::gguf_tensor_name("model.encoder.layers." + std::to_string(i) + "." + n));
    for (int64_t l = 0; l < t.n_dec; ++l) {
      const std::string p = "model.decoder.layers." + std::to_string(l) + ".";
      for (const char *a : {"self_attn", "encoder_attn"})
        for (const char *n : {"q_proj.weight", "q_proj.bias", "k_proj.weight", "v_proj.weight", "v_proj.bias",
                              "out_proj.weight", "out_proj.bias"})
          mapped.insert(ow::gguf_tensor_name(p + a + "." + n));
      for (const char *n : {"self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"})
        for (const char *k : {".weight", ".bias"}) mapped.insert(ow::gguf_tensor_name(p + n + k));
      for (const char *n : {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"})
        mapped.insert(ow::gguf_tensor_name(p + n));
    }
    for (const char *n : {"model.encoder.conv1.weight", "model.encoder.conv1.bias", "model.encoder.conv2.weight",
                          "model.encoder.conv2.bias", "model.encoder.embed_positions.weight",
                          "model.encoder.layer_norm.weight", "model.encoder.layer_norm.bias",
                          "model.decoder.embed_tokens.weight", "model.decoder.embed_positions.weight",
                          "model.decoder.layer_norm.weight", "model.decoder.layer_norm.bias"})
      mapped.insert(ow::gguf_tensor_name(n));
    std::set<std::string> want_names;
    for (const auto &kv : real.items())
      if (kv.key().rfind("frontend.", 0) != 0) want_names.insert(kv.key());
    check(mapped == want_names, "every tensor of the real GGUF but the frontend is read (" +
                                    std::to_string(mapped.size()) + " of " + std::to_string(want_names.size()) + ")");
    check(ow::gguf_tensor_name("model.encoder.layers.x.fc1.weight").empty() &&
              ow::gguf_tensor_name("proj_out.weight").empty(),
          "names with no GGUF counterpart map to nothing");
  }

  {
    const fs::path dir = fs::temp_directory_path() / "ow_weight_source_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    auto touch = [&](const char *n) { std::ofstream(dir / n).put('x'); };
    check(ow::weights_present(dir.string()).empty(), "an empty directory holds no weights");
    touch("model.gguf");
    touch("model.safetensors");
    auto p = ow::weights_present(dir.string());
    check(p == std::vector<WeightKind>{WeightKind::Stock, WeightKind::Gguf},
          "the 16-bit checkpoint is read before a GGUF");
    touch("model.open.safetensors");
    p = ow::weights_present(dir.string());
    check(p.size() == 3 && p.front() == WeightKind::Container, "the converted container is read first");
    fs::remove_all(dir);
  }

  std::printf("%s\n", failures ? "weight_source_test FAILED" : "weight_source_test passed");
  return failures ? 1 : 0;
}

int compare(const std::string &container, const std::string &weights) {
  const std::string ext = fs::path(weights).extension().string();
  const WeightKind kind = ext == ".gguf" ? WeightKind::Gguf : WeightKind::Stock;
  const ow::ModelShape t = ow::turbo_shape();
  const auto ref = ow::open_weight_file(container, WeightKind::Container, t);
  const auto src = ow::open_weight_file(weights, kind, t);
  std::printf("reference %s\nsource    %s\n", container.c_str(), src->describe().c_str());
  size_t same = 0, total = 0, values = 0, diff_values = 0;
  for (const std::string &n : ow::container_tensor_names(t)) {
    ++total;
    const ow::TensorDesc a = ref->desc(n), b = src->desc(n);
    bool ok = a.dtype == b.dtype && a.shape == b.shape;
    size_t diff = 0, count = 0;
    if (ok && a.dtype == "BF16") {
      const auto x = ref->bf16(n), y = src->bf16(n);
      count = x.size();
      for (size_t i = 0; i < x.size(); ++i) diff += x[i] != y[i];
    } else if (ok) {
      const auto x = ref->f32(n), y = src->f32(n);
      count = x.size();
      for (size_t i = 0; i < x.size(); ++i) diff += std::memcmp(&x[i], &y[i], 4) != 0;
    }
    values += count;
    diff_values += diff;
    ok = ok && diff == 0;
    if (ok) ++same;
    else std::printf("DIFF %s: %zu of %zu values\n", n.c_str(), diff, count);
  }
  std::printf("%zu/%zu tensors identical, %zu of %zu values differ\n", same, total, diff_values, values);
  return same == total ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--compare") return compare(argv[2], argv[3]);
    if (argc == 2) return unit(argv[1]);
    std::fprintf(stderr, "usage: weight_source_test TESTDATA_DIR | --compare CONTAINER WEIGHTS_FILE\n");
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "weight_source_test: %s\n", e.what());
    return 1;
  }
}
