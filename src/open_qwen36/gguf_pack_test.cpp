/// \file gguf_pack_test.cpp
/// \brief No-XRT unit test for the GGUF-direct path: a synthetic GGUF goes
///        through pools::apply's std_perm_gguf and the packed pool bytes must
///        dequantize BIT-EXACTLY to the GGUF blocks' values (the pack is a byte
///        permutation plus an exact fp16 -> f32 widening of the scales;
///        open_kernels/gguf_pool.py holds the same law in NumPy).
///
/// Also covers GgufFile::embed_row (Q8_0) and pack_norm's f32 -> bf16.
///
///     cmake -S . -B build && cmake --build build && ctest --test-dir build
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/pools.hpp"

namespace fs = std::filesystem;

using namespace open_qwen36;

namespace {

void put_u32(std::vector<uint8_t>& f, uint32_t v) {
    f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 4);
}
void put_u64(std::vector<uint8_t>& f, uint64_t v) {
    f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 8);
}
void put_str(std::vector<uint8_t>& f, const std::string& s) {
    put_u64(f, s.size());
    f.insert(f.end(), s.begin(), s.end());
}
void put_kv_str(std::vector<uint8_t>& f, const std::string& k, const std::string& v) {
    put_str(f, k);
    put_u32(f, 8);                                   // GGUF_TYPE_STRING
    put_str(f, v);
}
void put_kv_u32(std::vector<uint8_t>& f, const std::string& k, uint32_t v) {
    put_str(f, k);
    put_u32(f, 4);                                   // GGUF_TYPE_UINT32
    put_u32(f, v);
}

uint16_t f32_to_f16(float f) {                       // truncating (test values only)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>(((u >> 16) & 0x8000) | ((((u >> 23) - 127 + 15) & 0x1F) << 10) |
                                 ((u >> 13) & 0x3FF));
}

/// GGUF blocks for a [rows, cols] tensor and the reference dequant (computed
/// from the EXACT fp16 scales that hit the file).
struct Blocks {
    std::vector<uint8_t> bytes;
    std::vector<float> values;                       // rows x cols
    unsigned rows, cols, blk, type;
};

Blocks make_blocks(unsigned rows, unsigned cols, const std::string& type, std::mt19937_64& rng) {
    const unsigned nb = cols / 32;
    const bool q4 = type != "Q8_0";
    const bool has_min = type == "Q4_1";
    const unsigned blk = q4 ? (has_min ? 20u : 18u) : 34u;
    Blocks out;
    out.rows = rows;
    out.cols = cols;
    out.blk = blk;
    out.type = type == "Q4_0" ? 2u : type == "Q4_1" ? 3u : 8u;
    out.bytes.assign(static_cast<size_t>(rows) * nb * blk, 0);
    out.values.resize(static_cast<size_t>(rows) * cols);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned b = 0; b < nb; ++b) {
            uint8_t* p = out.bytes.data() + (static_cast<size_t>(r) * nb + b) * blk;
            const uint16_t dh = f32_to_f16(0.01f + 0.02f * static_cast<float>(rng() % 1000) / 1000.f);
            const uint16_t mh = has_min ? f32_to_f16(-0.05f + 0.04f * static_cast<float>(rng() % 1000) / 1000.f) : 0;
            std::memcpy(p, &dh, 2);
            if (has_min) std::memcpy(p + 2, &mh, 2);
            const float df = fp16_to_f32(dh);        // the exact value that hits the file
            const float mf = has_min ? fp16_to_f32(mh) : 0.f;
            for (unsigned j = 0; j < 32; ++j) {
                if (q4) {
                    const uint8_t q = static_cast<uint8_t>(rng() % 16);
                    if (j < 16) p[has_min ? 4 + j : 2 + j] |= q;
                    else p[has_min ? 4 + (j - 16) : 2 + (j - 16)] |= static_cast<uint8_t>(q << 4);
                    out.values[static_cast<size_t>(r) * cols + b * 32 + j] = df *
                        (has_min ? static_cast<float>(q) : static_cast<float>(static_cast<int>(q) - 8)) + mf;
                } else {
                    const int8_t q = static_cast<int8_t>(rng() % 256) - 128;
                    p[2 + j] = static_cast<uint8_t>(q);
                    out.values[static_cast<size_t>(r) * cols + b * 32 + j] = df * q;
                }
            }
        }
    }
    return out;
}

/// Q6_K tensor blocks and the reference dequant. Q6_K is the one super-block
/// quant whose embed_row had a real bug (the `qh` pointer advanced 16 B per
/// 128-value half instead of 32, skewing every second half of a super-block),
/// so this test is deliberately self-contained: it synthesizes the block with a
/// known code layout and checks GgufFile::embed_row against it.
struct BlocksQ6K {
    unsigned rows, cols;                 // cols is a multiple of 256
    std::vector<uint8_t> bytes;          // rows * (cols/256) * 210
    std::vector<float> values;           // rows * cols
};

BlocksQ6K make_blocks_q6k(unsigned rows, unsigned cols, std::mt19937_64& rng) {
    const unsigned nbs = cols / 256;                 // super-blocks per row
    BlocksQ6K b;
    b.rows = rows;
    b.cols = cols;
    b.bytes.assign(static_cast<size_t>(rows) * nbs * 210, 0);
    b.values.resize(static_cast<size_t>(rows) * cols);
    const uint16_t dh = f32_to_f16(0.5f);            // exactly representable
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned sb = 0; sb < nbs; ++sb) {
            uint8_t* p = b.bytes.data() + (static_cast<size_t>(r) * nbs + sb) * 210;
            std::memcpy(p + 208, &dh, 2);            // d
            std::memset(p + 192, 1, 16);             // scales: all 1 (int8), so y = d * 1 * (code-32)
            float* y = b.values.data() + static_cast<size_t>(r) * cols + sb * 256;
            for (int v = 0; v < 256; ++v) {
                const uint8_t code = static_cast<uint8_t>(rng() % 64);
                // ql: one byte per value's LOW 4 bits; low nibble for the first 64 of a
                // 128-half, high nibble for the next 64 (the dequant's q1/q2 vs q3/q4).
                const unsigned ql_byte = (v >= 128 ? 64u : 0u) + static_cast<unsigned>(v & 63);
                if ((v % 128) >= 64) p[ql_byte] |= static_cast<uint8_t>((code & 0xF) << 4);
                else p[ql_byte] |= static_cast<uint8_t>(code & 0xF);
                // qh: 32 bytes per 128-half, each covers 4 values (bits 0/2/4/6).
                const unsigned rel = static_cast<unsigned>((v >= 128 ? v - 128 : v) % 128);
                const unsigned l = rel % 32, which = rel / 32;
                p[128 + (v >= 128 ? 32u + l : l)] |=
                    static_cast<uint8_t>(((code >> 4) & 0x3) << (2 * which));
                y[v] = 0.5f * static_cast<float>(static_cast<int>(code) - 32);
            }
        }
    }
    return b;
}

/// A one-tensor GGUF carrying a single Q6_K tensor (rows x 256).
std::string write_gguf_q6k(const fs::path& dir, const BlocksQ6K& b) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);
    put_u64(f, 1);                                    // tensors
    put_u64(f, 1);                                    // kv pairs
    put_kv_str(f, "general.architecture", "llama");
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    put_str(f, "blk.0.attn_q.weight");
    put_u32(f, 2);
    put_u64(f, b.cols);                               // cols (fastest)
    put_u64(f, b.rows);
    put_u32(f, 14);                                   // GGML_TYPE_Q6_K
    put_u64(f, 0);                                    // RELATIVE offset (0: data sits at the data base)
    const size_t data_off = align32(f.size());        // == the GGUF's data base
    while (f.size() < data_off) f.push_back(0);
    f.insert(f.end(), b.bytes.begin(), b.bytes.end());
    const fs::path out = dir / "test-q6k.gguf";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

std::string write_gguf(const fs::path& dir, const Blocks& mm, const Blocks& emb, const std::vector<float>& norm) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);                                   // v3: no alignment field, fixed 32
    put_u64(f, 3);                                   // tensors
    put_u64(f, 2);                                   // kv pairs
    put_kv_str(f, "general.architecture", "llama");
    put_kv_u32(f, "llama.embedding_length", 512);
    // tensor infos with placeholder offsets (patched once the data layout is known)
    const auto put_tensor = [&](const std::string& name, const std::vector<uint64_t>& dims, uint32_t type) {
        put_str(f, name);
        put_u32(f, static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) put_u64(f, d);
        put_u32(f, type);
        put_u64(f, 0);
    };
    put_tensor("blk.0.attn_q.weight", {512, 64}, mm.type);
    put_tensor("token_embd.weight", {256, 32}, 8);           // Q8_0
    put_tensor("blk.0.attn_norm.weight", {512}, 0);          // F32
    // data section: 32-aligned right after the header; the tensor offsets are
    // RELATIVE to it (GGUF semantics)
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    const size_t off = align32(f.size());
    std::vector<uint64_t> offs;                  // relative offsets, patched into the header
    size_t rel = 0;
    auto emit = [&](const std::vector<uint8_t>& b) {
        offs.push_back(rel);
        rel = align32(rel + b.size());
    };
    emit(mm.bytes);
    emit(emb.bytes);
    offs.push_back(rel);
    // patch the three tensor offsets
    const size_t n_off_fields = 3;
    for (size_t t = 0; t < n_off_fields; ++t) {
        // the t-th tensor info's offset field: walk the header
        size_t p = 4 + 4 + 8 + 8;
        // skip the kv pairs
        for (int kv = 0; kv < 2; ++kv) {
            uint64_t ln;
            std::memcpy(&ln, f.data() + p, 8);
            p += 8 + ln;                             // key
            uint32_t ty;
            std::memcpy(&ty, f.data() + p, 4);
            p += 4;
            if (ty == 8) {                           // string value
                std::memcpy(&ln, f.data() + p, 8);
                p += 8 + ln;
            } else {                                 // u32 value
                p += 4;
            }
        }
        for (size_t i = 0; i < t; ++i) {
            uint64_t ln;
            uint32_t nd;
            std::memcpy(&ln, f.data() + p, 8);
            p += 8 + ln;                             // name
            std::memcpy(&nd, f.data() + p, 4);
            p += 4 + 8 * nd;                         // dims
            p += 4;                                  // type
            p += 8;                                  // the offset field itself
        }
        uint64_t ln;
        uint32_t nd;
        std::memcpy(&ln, f.data() + p, 8);
        p += 8 + ln;                                 // name
        std::memcpy(&nd, f.data() + p, 4);
        p += 4 + 8 * nd + 4;                         // dims, type -> the offset field
        std::memcpy(f.data() + p, &offs[t], 8);
    }
    while (f.size() < off + offs[0]) f.push_back(0);   // pad to the first tensor
    f.insert(f.end(), mm.bytes.begin(), mm.bytes.end());
    while (f.size() < off + offs[1]) f.push_back(0);
    f.insert(f.end(), emb.bytes.begin(), emb.bytes.end());
    while (f.size() < off + offs[2]) f.push_back(0);
    for (float v : norm) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        f.insert(f.end(), b, b + 4);
    }
    const fs::path out = dir / ("test-" + std::to_string(mm.type) + ".gguf");
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;                      // fastest first
    uint32_t type;
    std::vector<uint8_t> bytes;
};

/// A GGUF holding `ts` (architecture "gemma3"), 32-aligned; `n_tokens` adds a tokenizer.ggml.tokens array.
std::string write_gguf_tensors(const fs::path& dir, const std::string& fname, const std::vector<Tensor>& ts,
                               uint64_t n_tokens = 0) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);
    put_u64(f, ts.size());
    put_u64(f, n_tokens ? 2 : 1);
    put_kv_str(f, "general.architecture", "gemma3");
    if (n_tokens) {
        put_str(f, "tokenizer.ggml.tokens");
        put_u32(f, 9);                                   // an array ...
        put_u32(f, 8);                                   // ... of strings
        put_u64(f, n_tokens);
        for (uint64_t i = 0; i < n_tokens; ++i) put_str(f, "");
    }
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    size_t rel = 0;
    std::vector<size_t> offs;
    for (const Tensor& t : ts) {
        put_str(f, t.name);
        put_u32(f, static_cast<uint32_t>(t.dims.size()));
        for (uint64_t d : t.dims) put_u64(f, d);
        put_u32(f, t.type);
        put_u64(f, rel);
        offs.push_back(rel);
        rel = align32(rel + t.bytes.size());
    }
    const size_t base = align32(f.size());
    for (size_t i = 0; i < ts.size(); ++i) {
        while (f.size() < base + offs[i]) f.push_back(0);
        f.insert(f.end(), ts[i].bytes.begin(), ts[i].bytes.end());
    }
    const fs::path out = dir / fname;
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

/// One value of an f32-scale pool chunk (6144 B): row r, column k of its 32 x 256 tile.
float pool_value(const uint8_t* d, size_t r, size_t k) {
    const size_t kb = k / 32;
    float df, mf;
    std::memcpy(&df, d + 4 * (kb * 32 + r), 4);
    std::memcpy(&mf, d + 1024 + 4 * (kb * 32 + r), 4);
    const size_t p = (r / 16) * 4096 + k * 16 + (r % 16);
    const uint8_t byte = d[2048 + (p >> 1)];
    return ((p & 1) ? byte >> 4 : byte & 0xF) * df + mf;
}

/// OPEN-PACK-PAD: the source reads back exactly, and every pad value is +0.
int check_padded(const PackOp& op, const std::vector<uint8_t>& pool, const std::vector<float>& want,
                 size_t src_rows, size_t src_cols, const char* what) {
    const size_t per_band = op.in_dim / 128;
    size_t bad = 0, zeros = 0;
    for (size_t c = 0; c < op.nch; ++c) {
        const size_t rows0 = 64 * (c / per_band) + 32 * (c % 2), cols0 = 256 * ((c % per_band) / 2);
        for (size_t r = 0; r < 32; ++r)
            for (size_t k = 0; k < 256; ++k) {
                const size_t R = rows0 + r, K = cols0 + k;
                const float got = pool_value(pool.data() + c * 6144, r, k);
                const bool real = R < src_rows && K < src_cols;
                if (!real) ++zeros;
                if (f32_bits(got) != f32_bits(real ? want[R * src_cols + K] : 0.f)) {
                    if (bad < 5) std::printf("  %s: row %zu col %zu got %.9g\n", what, R, K, got);
                    ++bad;
                }
            }
    }
    std::printf("%s %s (%zu values differ; %zu pad values all +0)\n", bad ? "FAIL" : "ok   ", what, bad, zeros);
    return bad ? 1 : 0;
}

}  // namespace

int main() {
    std::mt19937_64 rng(42);
    const Blocks mm = make_blocks(64, 512, "Q4_1", rng);
    std::vector<float> norm(512);
    for (float& v : norm) v = 0.9f + 0.1f * static_cast<float>(rng() % 1000) / 1000.f;
    const Blocks emb = make_blocks(32, 256, "Q8_0", rng);
    const fs::path dir = fs::temp_directory_path() / "gguf_pack_test";
    fs::create_directories(dir);
    GgufFile g(write_gguf(dir, mm, emb, norm));

    // ---- header sanity
    if (g.kv_str("general.architecture") != "llama" || g.kv_u64("llama.embedding_length") != 512 ||
        !g.has("model.layers.0.self_attn.q_proj.weight")) {   // found through the HF-name map
        std::printf("FAIL header\n");
        return 1;
    }
    if (GgufFile::gguf_name("model.embed_tokens.weight") != "token_embd.weight" ||
        GgufFile::gguf_name("lm_head.weight") != "output.weight" ||
        GgufFile::gguf_name("model.layers.3.mlp.down_proj.weight") != "blk.3.ffn_down.weight" ||
        GgufFile::gguf_name("model.layers.1.input_layernorm.weight") != "blk.1.attn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.self_attn.q_norm.weight") != "blk.2.attn_q_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.pre_feedforward_layernorm.weight") != "blk.2.ffn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.post_feedforward_layernorm.weight") != "blk.2.post_ffw_norm.weight" ||
        GgufFile::gguf_name("model.norm.weight") != "output_norm.weight") {
        std::printf("FAIL name map\n");
        return 1;
    }
    std::printf("ok    header (kv, tensor index, HF-name map)\n");

    // ---- std_perm_gguf: the pool bytes dequantize to the GGUF values bit-exactly
    PackOp op;
    op.op = "std_perm_gguf";
    op.tensor = "model.layers.0.self_attn.q_proj.weight";   // the manifest HF-style name
    op.in_dim = 512;
    op.nch = 64ull * 512 / (32 * 256);               // 4 chunks of 32x256
    std::vector<uint8_t> pool(static_cast<size_t>(op.nch) * 6144, 0);
    pools::apply(op, g, 0, pool.data(), pool.size(), 6144);

    size_t bad = 0;
    for (size_t c = 0; c < op.nch; ++c) {
        const uint8_t* d = pool.data() + c * 6144;
        // the band law: rows0 = 64*(c/per_band) + 32*(c%2), cols0 = 256*((c%per_band)/2)
        const size_t per_band = 4;
        const size_t rows0 = 64 * (c / per_band) + 32 * (c % 2), cols0 = 256 * ((c % per_band) / 2);
        for (size_t r = 0; r < 32; ++r) {
            for (size_t k = 0; k < 256; ++k) {
                const size_t kb = k / 32;
                const float df = *reinterpret_cast<const float*>(d + 4 * (kb * 32 + r));
                const float mf = *reinterpret_cast<const float*>(d + 1024 + 4 * (kb * 32 + r));
                const size_t p = (r / 16) * 4096 + k * 16 + (r % 16);
                const uint8_t byte = d[2048 + (p >> 1)];
                const uint8_t nib = (p & 1) ? byte >> 4 : byte & 0xF;
                const float got = nib * df + mf;
                const float want = mm.values[(rows0 + r) * mm.cols + cols0 + k];
                if (f32_bits(got) != f32_bits(want)) {
                    if (bad < 5)
                        std::printf("  chunk %zu r=%zu k=%zu: got %.9g want %.9g\n", c, r, k, got, want);
                    ++bad;
                }
            }
        }
    }
    std::printf("%s std_perm_gguf (%zu of %u values differ)\n", bad ? "FAIL" : "ok", bad, op.nch * 32u * 256u);
    if (bad) return 1;

    // Q4_0 has a two-byte header (d only), unlike Q4_1's d+m header. Exercise
    // the same pack path separately so its code offset cannot drift to +4.
    {
        const Blocks mm0 = make_blocks(64, 512, "Q4_0", rng);
        GgufFile g0(write_gguf(dir, mm0, emb, norm));
        std::vector<uint8_t> p0(static_cast<size_t>(op.nch) * 6144, 0);
        pools::apply(op, g0, 0, p0.data(), p0.size(), 6144);
        size_t bad0 = 0;
        for (size_t c = 0; c < op.nch; ++c) {
            const uint8_t* d = p0.data() + c * 6144;
            const size_t per_band = 4;
            const size_t rows0 = 64 * (c / per_band) + 32 * (c % 2);
            const size_t cols0 = 256 * ((c % per_band) / 2);
            for (size_t r = 0; r < 32; ++r)
                for (size_t k = 0; k < 256; ++k) {
                    const size_t kb = k / 32;
                    const float df = *reinterpret_cast<const float*>(d + 4 * (kb * 32 + r));
                    const size_t p = (r / 16) * 4096 + k * 16 + (r % 16);
                    const uint8_t byte = d[2048 + (p >> 1)];
                    const uint8_t nib = (p & 1) ? byte >> 4 : byte & 0xF;
                    if (f32_bits(nib * df - 8.f * df) != f32_bits(mm0.values[(rows0 + r) * mm0.cols + cols0 + k]))
                        ++bad0;
                }
        }
        std::printf("%s std_perm_gguf Q4_0 (%zu values differ)\n", bad0 ? "FAIL" : "ok", bad0);
        if (bad0) return 1;
    }

    // ---- pack_norm: f32 -> bf16 (RNE)
    {
        std::vector<uint8_t> nb(norm.size() * 2);
        pools::pack_norm(g, "blk.0.attn_norm.weight", nb.size(), nb.data());
        for (size_t i = 0; i < norm.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &norm[i], 4);
            const uint16_t want = static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
            uint16_t got;
            std::memcpy(&got, nb.data() + 2 * i, 2);
            if (got != want) {
                std::printf("FAIL pack_norm[%zu]: got %04x want %04x\n", i, got, want);
                return 1;
            }
        }
        std::printf("ok    pack_norm f32 -> bf16 (RNE)\n");
    }

    // ---- embed_row (Q8_0, bit-exact)
    {
        std::vector<float> row(256);
        g.embed_row("token_embd.weight", 7, 256, row.data());
        for (size_t i = 0; i < 256; ++i) {
            if (f32_bits(row[i]) != f32_bits(emb.values[7 * 256 + i])) {
                std::printf("FAIL embed_row(Q8_0)[%zu]: got %.9g want %.9g\n", i, row[i], emb.values[7 * 256 + i]);
                return 1;
            }
        }
        std::printf("ok    embed_row Q8_0 (bit-exact)\n");
    }

    // ---- embed_row (Q6_K, the qh-advance regression: every super-block half)
    {
        BlocksQ6K q6 = make_blocks_q6k(4, 512, rng);      // 4 rows x 2 super-blocks
        GgufFile g6(write_gguf_q6k(dir, q6));
        std::vector<float> row(512);
        bool bad6 = false;
        for (unsigned r = 0; r < 4 && !bad6; ++r) {
            g6.embed_row("blk.0.attn_q.weight", r, 512, row.data());
            for (size_t i = 0; i < 512; ++i)
                if (f32_bits(row[i]) != f32_bits(q6.values[static_cast<size_t>(r) * 512 + i])) {
                    std::printf("FAIL embed_row(Q6_K)[r%u i%zu]: got %.9g want %.9g\n", r, i, row[i],
                                q6.values[static_cast<size_t>(r) * 512 + i]);
                    bad6 = true;
                    break;
                }
        }
        std::printf("%s embed_row Q6_K (both super-block halves bit-exact)\n", bad6 ? "FAIL" : "ok");
        if (bad6) return 1;
    }

    // ---- OPEN-PACK-PAD: a 96 x 288 Q4_0 source in a 128 x 512 pool, rows and columns padded
    {
        const Blocks src = make_blocks(96, 288, "Q4_0", rng);
        GgufFile gp(write_gguf_tensors(dir, "test-pad-q4_0.gguf",
                                       {{"blk.0.attn_output.weight", {288, 96}, 2, src.bytes}}));
        PackOp p;
        p.op = "std_perm_gguf";
        p.tensor = "model.layers.0.self_attn.o_proj.weight";
        p.in_dim = 512;
        p.nch = 128ull * 512 / 8192;
        p.src_dim = 288;
        p.src_rows = 96;
        std::vector<uint8_t> pool(static_cast<size_t>(p.nch) * 6144, 0xA5);   // the op owns every byte
        pools::apply(p, gp, 0, pool.data(), pool.size(), 6144);
        if (check_padded(p, pool, src.values, 96, 288, "std_perm_gguf padded Q4_0 96x288 -> 128x512")) return 1;
        bool refused = false;
        try {
            PackOp q = p;
            q.src_rows = 95;                         // the file holds 96: the plan must say so
            pools::apply(q, gp, 0, pool.data(), pool.size(), 6144);
        } catch (const std::exception&) {
            refused = true;
        }
        std::printf("%s std_perm_gguf refuses a source whose shape the plan does not name\n", refused ? "ok   " : "FAIL");
        if (!refused) return 1;
        bool q4nx_refused = false;
        try {
            PackOp q = p;
            q.op = "std_perm";                        // a .q4nx container cannot be padded
            std::vector<uint8_t> p5(static_cast<size_t>(q.nch) * 5120);
            pools::apply(q, gp, 0, p5.data(), p5.size(), 5120);
        } catch (const std::exception& e) {
            q4nx_refused = std::string(e.what()).find("OPEN-PACK-PAD") != std::string::npos;
        }
        std::printf("%s std_perm refuses a padded op by name\n", q4nx_refused ? "ok   " : "FAIL");
        if (!q4nx_refused) return 1;
    }

    // ---- without src_rows the file holds the plan's rows exactly; only a head may fall short, by under a band
    {
        const Blocks k96 = make_blocks(96, 512, "Q4_0", rng), v128 = make_blocks(128, 512, "Q4_0", rng);
        const Blocks h64 = make_blocks(64, 512, "Q4_0", rng);
        GgufFile gr(write_gguf_tensors(dir, "test-rows.gguf",
                                       {{"blk.0.attn_k.weight", {512, 96}, 2, k96.bytes},
                                        {"blk.0.attn_v.weight", {512, 128}, 2, v128.bytes},
                                        {"output.weight", {512, 96}, 2, k96.bytes}}));
        GgufFile gh(write_gguf_tensors(dir, "test-head64.gguf", {{"output.weight", {512, 64}, 2, h64.bytes}}));
        const auto plan = [](const std::string& tensor, uint64_t rows) {
            PackOp p;
            p.op = "std_perm_gguf";
            p.tensor = tensor;
            p.in_dim = 512;
            p.nch = rows * 512 / 8192;
            return p;
        };
        const auto refused = [](const PackOp& p, const GgufFile& gf) {
            std::vector<uint8_t> pool(static_cast<size_t>(p.nch) * 6144);
            try {
                pools::apply(p, gf, 0, pool.data(), pool.size(), 6144);
            } catch (const std::exception& e) {
                return std::string(e.what()).find("the file holds") != std::string::npos;
            }
            return false;
        };
        const bool ok = refused(plan("model.layers.0.self_attn.k_proj.weight", 128), gr) &&
                        refused(plan("model.layers.0.self_attn.v_proj.weight", 64), gr) &&
                        refused(plan("lm_head.weight", 128), gh);
        std::printf("%s std_perm_gguf refuses a file with fewer or more rows than the plan, and a head a band short\n",
                    ok ? "ok   " : "FAIL");
        if (!ok) return 1;
        const PackOp v = plan("model.layers.0.self_attn.v_proj.weight", 128), h = plan("lm_head.weight", 128);
        std::vector<uint8_t> pv(static_cast<size_t>(v.nch) * 6144, 0xA5), ph(static_cast<size_t>(h.nch) * 6144, 0xA5);
        pools::apply(v, gr, 0, pv.data(), pv.size(), 6144);
        pools::apply(h, gr, 0, ph.data(), ph.size(), 6144);
        if (check_padded(v, pv, v128.values, 128, 512, "std_perm_gguf Q4_0 128x512 at the plan's own rows") ||
            check_padded(h, ph, k96.values, 96, 512, "std_perm_gguf head of 96 rows in a 128-row plan"))
            return 1;
        // Traces: OPEN-VISION-GEMMA3 -- a head that holds the file's whole vocabulary, a band short of the plan
        GgufFile gv(write_gguf_tensors(dir, "test-head-vocab.gguf", {{"output.weight", {512, 64}, 2, h64.bytes}}, 64));
        GgufFile gw(write_gguf_tensors(dir, "test-head-vocab96.gguf", {{"output.weight", {512, 64}, 2, h64.bytes}}, 96));
        std::vector<uint8_t> pt(static_cast<size_t>(h.nch) * 6144, 0xA5);
        pools::apply(h, gv, 0, pt.data(), pt.size(), 6144);
        if (check_padded(h, pt, h64.values, 64, 512, "std_perm_gguf head of the file's whole 64-token vocabulary in a 128-row plan"))
            return 1;
        const bool other = refused(h, gw);
        std::printf("%s std_perm_gguf still refuses a band-short head that is not the file's whole vocabulary\n",
                    other ? "ok   " : "FAIL");
        if (!other) return 1;
    }

    // ---- an F16 head (Gemma 3's tied embedding) whose blocks are exactly q4_1 re-quantizes back exactly
    {
        const unsigned rows = 64, cols = 288;
        std::vector<float> vals(static_cast<size_t>(rows) * cols);
        std::vector<uint8_t> bytes(vals.size() * 2);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned b = 0; b < cols / 32; ++b) {
                const float d = 0.25f, m = -1.0f + 0.5f * static_cast<float>((r + b) % 3);
                for (unsigned j = 0; j < 32; ++j) {
                    const unsigned q = j == 0 ? 0u : j == 1 ? 15u : static_cast<unsigned>(rng() % 16);
                    const size_t i = static_cast<size_t>(r) * cols + b * 32 + j;
                    vals[i] = m + d * static_cast<float>(q);
                    const uint16_t h = f32_to_fp16(vals[i]);   // exact: multiples of 0.25 in [-1, 4.75]
                    std::memcpy(bytes.data() + 2 * i, &h, 2);
                }
            }
        GgufFile gf(write_gguf_tensors(dir, "test-f16.gguf", {{"token_embd.weight", {cols, rows}, 1, bytes}}));
        PackOp p;
        p.op = "std_perm_gguf";
        p.tensor = "lm_head.weight";                 // no output.weight: the tied embedding serves
        p.in_dim = 512;
        p.nch = 64ull * 512 / 8192;
        p.src_dim = cols;
        std::vector<uint8_t> pool(static_cast<size_t>(p.nch) * 6144, 0xA5);
        pools::apply(p, gf, 0, pool.data(), pool.size(), 6144);
        if (check_padded(p, pool, vals, rows, cols, "std_perm_gguf F16 head 64x288 -> 64x512 (exact re-quantization)"))
            return 1;
    }

    // ---- OPEN-EMBED-STRIDE on a GGUF: a row read into a wider buffer is zero-extended
    {
        std::vector<float> row(320, -1.f);
        g.embed_row("token_embd.weight", 7, 320, row.data());
        bool ok = true;
        for (size_t i = 0; i < 320; ++i)
            ok = ok && f32_bits(row[i]) == f32_bits(i < 256 ? emb.values[7 * 256 + i] : 0.f);
        bool refused = false;
        try {
            g.embed_row("token_embd.weight", 7, 128, row.data());
        } catch (const std::exception&) {
            refused = true;
        }
        std::printf("%s embed_row zero-extends to a padded width and refuses a narrower one\n",
                    ok && refused ? "ok   " : "FAIL");
        if (!(ok && refused)) return 1;
    }

    // ---- pack_norm into a padded slot: the real values, then zeros
    {
        std::vector<uint8_t> nb(600 * 2, 0xA5);
        pools::pack_norm(g, "blk.0.attn_norm.weight", nb.size(), nb.data());
        bool ok = true;
        for (size_t i = 0; i < 600; ++i) {
            uint16_t got, want = 0;
            std::memcpy(&got, nb.data() + 2 * i, 2);
            if (i < norm.size()) {
                const uint32_t u = f32_bits(norm[i]);
                want = static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
            }
            ok = ok && got == want;
        }
        bool refused = false;
        try {
            std::vector<uint8_t> small(500 * 2);
            pools::pack_norm(g, "blk.0.attn_norm.weight", small.size(), small.data());
        } catch (const std::exception&) {
            refused = true;
        }
        std::printf("%s pack_norm zero-extends into a padded slot and refuses a narrower one\n",
                    ok && refused ? "ok   " : "FAIL");
        if (!(ok && refused)) return 1;
    }

    // ---- sandwich norms: a Gemma GGUF's ffn_norm is HF's pre-FFN norm, a llama GGUF's its post-attention one
    {
        auto norm_t = [](const std::string& name, float v) {
            std::vector<uint8_t> b(8 * 4);
            for (int i = 0; i < 8; ++i) std::memcpy(b.data() + 4 * i, &v, 4);
            return Tensor{name, {8}, 0, b};
        };
        GgufFile gs(write_gguf_tensors(dir, "test-sandwich.gguf",
                                       {norm_t("blk.0.attn_norm.weight", 1.f), norm_t("blk.0.post_attention_norm.weight", 2.f),
                                        norm_t("blk.0.ffn_norm.weight", 3.f), norm_t("blk.0.post_ffw_norm.weight", 4.f)}));
        GgufFile gl(write_gguf_tensors(dir, "test-llama-norms.gguf",
                                       {norm_t("blk.0.attn_norm.weight", 1.f), norm_t("blk.0.ffn_norm.weight", 3.f)}));
        const std::string pre = "model.layers.0.";
        const bool ok = gs.bf16(pre + "input_layernorm.weight")[0] == 1.f &&
                        gs.bf16(pre + "post_attention_layernorm.weight")[0] == 2.f &&
                        gs.bf16(pre + "pre_feedforward_layernorm.weight")[0] == 3.f &&
                        gs.bf16(pre + "post_feedforward_layernorm.weight")[0] == 4.f &&
                        gl.bf16(pre + "post_attention_layernorm.weight")[0] == 3.f;
        std::printf("%s sandwich-norm names resolve by what the GGUF holds\n", ok ? "ok   " : "FAIL");
        if (!ok) return 1;
    }

    // ---- a short numeric array reads as its joined values, and is refused as a number
    {
        std::vector<uint8_t> f;
        f.insert(f.end(), {'G', 'G', 'U', 'F'});
        put_u32(f, 3);
        put_u64(f, 0);
        put_u64(f, 2);
        put_kv_str(f, "general.architecture", "gemma3");
        put_str(f, "gemma3.attention.head_count_kv");
        put_u32(f, 9);
        put_u32(f, 4);                                   // UINT32 elements
        put_u64(f, 3);
        for (uint32_t v : {8u, 0u, 8u}) put_u32(f, v);
        while (f.size() % 32) f.push_back(0);
        const fs::path p = dir / "test-array.gguf";
        {
            std::ofstream of(p, std::ios::binary);
            of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
        }
        GgufFile ga(p.string());
        const std::string k = "gemma3.attention.head_count_kv";
        bool u_refused = false, f_refused = false;
        try { (void)ga.kv_u64(k); } catch (const std::exception&) { u_refused = true; }
        try { (void)ga.kv_f64(k); } catch (const std::exception&) { f_refused = true; }
        const bool ok = ga.kv_str(k) == "8,0,8" && ga.kv_array_len(k) == 3 && u_refused && f_refused;
        std::printf("%s a per-layer array is not read as a number\n", ok ? "ok   " : "FAIL");
        if (!ok) return 1;
    }

    std::printf("GGUF-PACK OK\n");
    return 0;
}
