// Minimal read-only safetensors file: header index plus widening reads (2026-10-16).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace open_tts {

struct TensorInfo {
    std::string dtype;               // "F32", "F16", "BF16"
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;     // byte range inside the data section
    int64_t numel() const;
};

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path);
    const std::string& path() const { return path_; }
    bool has(const std::string& name) const { return index_.count(name) != 0; }
    const TensorInfo& info(const std::string& name) const;
    const std::map<std::string, TensorInfo>& tensors() const { return index_; }
    // F32/F16/BF16 widened exactly to f32 (F64 rounded).
    std::vector<float> f32(const std::string& name) const;
    double f64_scalar(const std::string& name) const;
    // dtype of the bulk of the weights, for the load log
    std::string main_dtype() const;

private:
    std::string path_;
    std::vector<uint8_t> data_;
    std::map<std::string, TensorInfo> index_;
};

float f16_to_f32(uint16_t h);
float bf16_to_f32(uint16_t b);

}  // namespace open_tts
