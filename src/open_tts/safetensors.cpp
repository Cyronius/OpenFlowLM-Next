// Minimal read-only safetensors file (2026-10-16).
#include "safetensors.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace open_tts {

int64_t TensorInfo::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 113;  // subnormal: renormalise
            while (!(man & 0x400)) { man <<= 1; --exp; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

SafetensorsFile::SafetensorsFile(const std::string& path) : path_(path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    in.seekg(0, std::ios::end);
    const uint64_t size = (uint64_t)in.tellg();
    in.seekg(0);
    uint64_t hlen = 0;
    if (size < 8 || !in.read(reinterpret_cast<char*>(&hlen), 8) || hlen > size - 8)
        throw std::runtime_error(path + ": not a safetensors file");
    std::string header(hlen, '\0');
    in.read(header.data(), (std::streamsize)hlen);
    const uint64_t data_size = size - 8 - hlen;
    data_.resize(data_size);
    if (!in.read(reinterpret_cast<char*>(data_.data()), (std::streamsize)data_size))
        throw std::runtime_error(path + ": short read");
    const auto j = nlohmann::json::parse(header);
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        TensorInfo t;
        t.dtype = it.value().at("dtype").get<std::string>();
        t.shape = it.value().at("shape").get<std::vector<int64_t>>();
        const auto off = it.value().at("data_offsets").get<std::vector<uint64_t>>();
        if (off.size() != 2 || off[0] > off[1] || off[1] > data_size)
            throw std::runtime_error(path + ": tensor " + it.key() + " lies outside the file");
        t.begin = off[0];
        t.end = off[1];
        const int64_t esz = t.dtype == "F64" ? 8 : t.dtype == "F32" ? 4 : (t.dtype == "F16" || t.dtype == "BF16") ? 2 : 0;
        if (esz && (int64_t)(t.end - t.begin) != t.numel() * esz)
            throw std::runtime_error(path + ": tensor " + it.key() + " has the wrong byte size");
        index_[it.key()] = std::move(t);
    }
}

const TensorInfo& SafetensorsFile::info(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error(path_ + ": no tensor " + name);
    return it->second;
}

std::vector<float> SafetensorsFile::f32(const std::string& name) const {
    const TensorInfo& t = info(name);
    const uint8_t* p = data_.data() + t.begin;
    const int64_t n = t.numel();
    std::vector<float> out((size_t)n);
    if (t.dtype == "F32") {
        std::memcpy(out.data(), p, (size_t)n * 4);
    } else if (t.dtype == "F64") {
        for (int64_t i = 0; i < n; ++i) {
            double v;
            std::memcpy(&v, p + 8 * i, 8);
            out[(size_t)i] = (float)v;
        }
    } else if (t.dtype == "F16" || t.dtype == "BF16") {
        const bool f16 = t.dtype == "F16";
        for (int64_t i = 0; i < n; ++i) {
            uint16_t v;
            std::memcpy(&v, p + 2 * i, 2);
            out[(size_t)i] = f16 ? f16_to_f32(v) : bf16_to_f32(v);
        }
    } else {
        throw std::runtime_error(path_ + ": tensor " + name + " has dtype " + t.dtype +
                                 "; the TTS engine reads F32, F16 or BF16");
    }
    return out;
}

double SafetensorsFile::f64_scalar(const std::string& name) const {
    const TensorInfo& t = info(name);
    if (t.dtype != "F64" || t.numel() < 1) return (double)f32(name).at(0);
    double v;
    std::memcpy(&v, data_.data() + t.begin, 8);
    return v;
}

std::string SafetensorsFile::main_dtype() const {
    std::map<std::string, int64_t> bytes;
    for (const auto& [name, t] : index_) bytes[t.dtype] += (int64_t)(t.end - t.begin);
    std::string best;
    int64_t most = -1;
    for (const auto& [d, b] : bytes)
        if (b > most) { most = b; best = d; }
    return best;
}

}  // namespace open_tts
