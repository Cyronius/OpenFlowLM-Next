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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/gguf_view.hpp"
#include "open_qwen36/manifest.hpp"
#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace fs = std::filesystem;

using namespace open_qwen36;

namespace {

// ---- OPEN-GGUF-Q4NX-VIEW: a synthetic qwen35 GGUF, built the same way by
// specs/open-engine/tests/test_gguf_q4nx_view.py, whose view must hash the same.
constexpr uint64_t VIEW_FNV1A = 0xbbe4818c9288d59cull;

uint64_t fnv1a(uint64_t h, const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

struct Lcg {
    uint64_t x;
    uint32_t next() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(x >> 33);
    }
};

struct SynTensor {
    std::string name;
    uint32_t type;                    // ggml type id
    std::vector<uint64_t> dims;       // fastest first
    std::vector<uint8_t> bytes;
};

/// The tensor's bytes from an LCG seeded by its name: random bytes, then every fp16 field patched finite.
SynTensor syn_tensor(const std::string& name, uint32_t type, std::vector<uint64_t> dims) {
    SynTensor t{name, type, dims, {}};
    uint64_t nv = 1;
    for (uint64_t d : dims) nv *= d;
    Lcg g{fnv1a(0xcbf29ce484222325ull, name.data(), name.size())};
    if (type == 0) {                                      // F32: k / 32768 - 1, exact in f32
        t.bytes.resize(nv * 4);
        for (uint64_t i = 0; i < nv; ++i) {
            const float v = static_cast<float>(static_cast<int>(g.next() & 0xFFFF)) * (1.0f / 32768.0f) - 1.0f;
            std::memcpy(t.bytes.data() + 4 * i, &v, 4);
        }
        return t;
    }
    if (type == 1) {                                      // F16
        t.bytes.resize(nv * 2);
        for (uint64_t i = 0; i < nv; ++i) {
            const uint32_t sign = (g.next() & 1u) << 15;
            const uint32_t man = g.next() & 0x3FFu;
            const uint16_t h = static_cast<uint16_t>(sign | 0x3000u | man);
            std::memcpy(t.bytes.data() + 2 * i, &h, 2);
        }
        return t;
    }
    const uint64_t bpb = type == 2 ? 18 : type == 3 ? 20 : type == 8 ? 34 : type == 12 ? 144 : 210;
    const uint64_t vpb = (type == 12 || type == 14) ? 256 : 32;
    const uint64_t nb = nv / vpb;
    t.bytes.resize(nb * bpb);
    for (auto& b : t.bytes) b = static_cast<uint8_t>(g.next() & 0xFF);
    auto put16 = [&](uint64_t at, uint16_t v) { std::memcpy(t.bytes.data() + at, &v, 2); };
    for (uint64_t b = 0; b < nb; ++b) {
        const uint64_t o = b * bpb;
        if (type == 2 || type == 3 || type == 8) put16(o, static_cast<uint16_t>(0x2000u | (g.next() & 0x3FFu)));
        if (type == 3) put16(o + 2, static_cast<uint16_t>(0xA000u | (g.next() & 0x3FFu)));
        if (type == 12) {
            put16(o, static_cast<uint16_t>(0x1800u | (g.next() & 0x3FFu)));
            put16(o + 2, static_cast<uint16_t>(0x1800u | (g.next() & 0x3FFu)));
        }
        if (type == 14) put16(o + 208, static_cast<uint16_t>(0x1800u | (g.next() & 0x3FFu)));
    }
    return t;
}

std::vector<SynTensor> syn_qwen35_tensors() {
    std::vector<SynTensor> v;
    auto add = [&](const std::string& n, uint32_t ty, std::vector<uint64_t> d) { v.push_back(syn_tensor(n, ty, d)); };
    add("token_embd.weight", 3, {256, 100});
    add("output.weight", 14, {256, 100});
    add("output_norm.weight", 0, {256});
    add("blk.0.attn_norm.weight", 0, {256});
    add("blk.0.post_attention_norm.weight", 0, {256});
    add("blk.0.attn_qkv.weight", 3, {256, 512});
    add("blk.0.attn_gate.weight", 2, {256, 256});
    add("blk.0.ssm_out.weight", 8, {256, 256});
    add("blk.0.ssm_alpha.weight", 3, {256, 4});
    add("blk.0.ssm_beta.weight", 1, {256, 4});
    add("blk.0.ssm_a", 0, {4});
    add("blk.0.ssm_dt.bias", 0, {4});
    add("blk.0.ssm_conv1d.weight", 0, {4, 512});
    add("blk.0.ssm_norm.weight", 0, {64});
    add("blk.0.ffn_up.weight", 14, {256, 256});
    add("blk.0.ffn_gate.weight", 12, {256, 256});
    add("blk.0.ffn_down.weight", 3, {256, 256});
    add("blk.3.attn_norm.weight", 0, {256});
    add("blk.3.post_attention_norm.weight", 0, {256});
    add("blk.3.attn_q.weight", 3, {256, 512});
    add("blk.3.attn_k.weight", 8, {256, 128});
    add("blk.3.attn_v.weight", 2, {256, 128});
    add("blk.3.attn_output.weight", 3, {256, 256});
    add("blk.3.attn_q_norm.weight", 0, {64});
    add("blk.3.attn_k_norm.weight", 0, {64});
    add("blk.3.ffn_up.weight", 3, {256, 256});
    add("blk.3.ffn_gate.weight", 3, {256, 256});
    add("blk.3.ffn_down.weight", 3, {256, 256});
    return v;
}

/// A GGUF v3 file of the tensors above under the qwen35 keys the view reads.
std::string write_syn_qwen35(const fs::path& dir, const std::vector<SynTensor>& ts) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    auto u32 = [&](uint32_t v) { f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 4); };
    auto u64 = [&](uint64_t v) { f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 8); };
    auto str = [&](const std::string& s) { u64(s.size()); f.insert(f.end(), s.begin(), s.end()); };
    const std::vector<std::pair<std::string, uint32_t>> kv = {
        {"qwen35.block_count", 4}, {"qwen35.embedding_length", 256}, {"qwen35.feed_forward_length", 256},
        {"qwen35.attention.head_count", 4}, {"qwen35.attention.head_count_kv", 2},
        {"qwen35.attention.key_length", 64}, {"qwen35.attention.value_length", 64},
        {"qwen35.ssm.conv_kernel", 4}, {"qwen35.ssm.state_size", 64}, {"qwen35.ssm.group_count", 2},
        {"qwen35.ssm.time_step_rank", 4}, {"qwen35.ssm.inner_size", 256},
        {"qwen35.full_attention_interval", 4}, {"qwen35.rope.dimension_count", 16}};
    u32(3);
    u64(ts.size());
    u64(kv.size() + 1);
    str("general.architecture");
    u32(8);
    str("qwen35");
    for (const auto& [k, v] : kv) {
        str(k);
        u32(4);
        u32(v);
    }
    uint64_t rel = 0;
    for (const auto& t : ts) {
        str(t.name);
        u32(static_cast<uint32_t>(t.dims.size()));
        for (uint64_t d : t.dims) u64(d);
        u32(t.type);
        u64(rel);
        rel = (rel + t.bytes.size() + 31) / 32 * 32;
    }
    while (f.size() % 32) f.push_back(0);
    const size_t base = f.size();
    rel = 0;
    for (const auto& t : ts) {
        while (f.size() < base + rel) f.push_back(0);
        f.insert(f.end(), t.bytes.begin(), t.bytes.end());
        rel = (rel + t.bytes.size() + 31) / 32 * 32;
    }
    const fs::path out = dir / "syn-qwen35.gguf";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

const char* const kViewNames[] = {
    "lm_head.weight", "model.norm.weight",
    "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.linear_attn.qkv_proj.weight", "model.layers.0.self_attn.gate_proj.weight",
    "model.layers.0.linear_attn.ssm_out_proj.weight", "model.layers.0.linear_attn.ssm_alpha_proj.bf16.weight",
    "model.layers.0.linear_attn.ssm_beta_proj.bf16.weight", "model.layers.0.linear_attn.ssm_a",
    "model.layers.0.linear_attn.ssm_dt.bias", "model.layers.0.linear_attn.ssm_conv1d.weight",
    "model.layers.0.linear_attn.ssm_norm.weight", "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight",
    "model.layers.3.self_attn.q_proj.weight", "model.layers.3.self_attn.k_proj.weight",
    "model.layers.3.self_attn.v_proj.weight", "model.layers.3.self_attn.o_proj.weight",
    "model.layers.3.self_attn.q_norm.weight", "model.layers.3.self_attn.k_norm.weight",
};

/// FNV-1a over every view tensor: dtype, row-major shape (u64 each), then its bytes.
uint64_t view_hash(const GgufQ4nxView& v, bool verbose) {
    uint64_t all = 0xcbf29ce484222325ull;
    for (const char* n : kViewNames) {
        const TensorMeta& m = v.meta(n);
        size_t nb = 0;
        const uint8_t* p = v.raw(n, &nb);
        uint64_t h = 0xcbf29ce484222325ull;
        h = fnv1a(h, m.dtype.data(), m.dtype.size());
        for (size_t s : m.shape) {
            const uint64_t s64 = s;
            h = fnv1a(h, &s64, 8);
        }
        h = fnv1a(h, p, nb);
        if (verbose) std::printf("  %-56s %s %016llx\n", n, m.dtype.c_str(), static_cast<unsigned long long>(h));
        all = fnv1a(all, &h, 8);
    }
    return all;
}

/// `gguf_pack_test --pack <manifest.json> <model.gguf> <out dir> <layer>...`: the engine's own
/// pool, consts and head bytes through the q4nx view, for open_kernels/model/gguf_q4nx_check.py.
int pack_mode(int argc, char** argv) {
    if (argc < 6) {
        std::printf("usage: gguf_pack_test --pack <manifest.json> <model.gguf> <out dir> <layer>...\n");
        return 2;
    }
    const Manifest m = Manifest::load(argv[2]);
    const GgufQ4nxView v(argv[3]);
    const fs::path out(argv[4]);
    fs::create_directories(out);
    auto dump = [&](const fs::path& p, const std::vector<uint8_t>& b) {
        std::ofstream of(p, std::ios::binary);
        of.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    };
    for (int i = 5; i < argc; ++i) {
        const int l = std::atoi(argv[i]);
        const LayerType& lt = m.layer_type(static_cast<size_t>(l));
        std::vector<uint8_t> pool(m.pool_bytes), consts(lt.consts_bytes);
        pools::pack_pool(m, lt, v, l, pool.data());
        pools::pack_consts(m, lt, v, l, consts.data());
        dump(out / ("pool_L" + std::to_string(l) + ".bin"), pool);
        dump(out / ("consts_L" + std::to_string(l) + ".bin"), consts);
        std::printf("layer %d (%s): pool %zu B, consts %zu B\n", l, m.layers[static_cast<size_t>(l)].c_str(),
                    pool.size(), consts.size());
    }
    std::vector<uint8_t> lm(m.lmhead_pool_bytes);
    pools::pack_lmhead(m, v, lm.data());
    dump(out / "pool_lmhead.bin", lm);
    std::vector<uint8_t> nw(m.norm_bytes);
    pools::pack_norm(v, m.norm_tensor, m.norm_bytes, nw.data());
    dump(out / "normw.bin", nw);
    std::printf("lm_head pool %zu B, final norm %zu B\n", lm.size(), nw.size());
    return 0;
}

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

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--pack") return pack_mode(argc, argv);
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
        GgufFile::gguf_name("model.layers.2.pre_feedforward_layernorm.weight") != "blk.2.post_attention_norm.weight" ||
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

    // ---- OPEN-GGUF-Q4NX-VIEW: the q4nx-form tensors of a synthetic qwen35 GGUF, hashed
    {
        const auto ts = syn_qwen35_tensors();
        const GgufQ4nxView v(write_syn_qwen35(dir, ts));
        const uint64_t h = view_hash(v, true);
        std::printf("%s q4nx view hash %016llx (want %016llx)\n", h == VIEW_FNV1A ? "ok  " : "FAIL",
                    static_cast<unsigned long long>(h), static_cast<unsigned long long>(VIEW_FNV1A));
        if (h != VIEW_FNV1A) return 1;
        // the conversions' meaning, independent of the Python twin
        const auto& G = v.geometry();
        bool ok = G.blocks == 4 && G.lin_v_dim == 64 && gguf_view::file_head(1, 2, 4) == 2 &&
                  gguf_view::file_head(2, 2, 4) == 1 && gguf_view::file_head(4, 2, 6) == 3 &&
                  gguf_view::file_head(1, 2, 6) == 2;
        // q_proj logical row 64 (head 1's q) is file row 128; row 256 (head 0's gate) is file row 64
        std::vector<float> want(256), got(256);
        v.gguf().embed_row("blk.3.attn_q.weight", 128, 256, want.data());
        size_t n = 0;
        const uint8_t* qc = v.raw("model.layers.3.self_attn.q_proj.weight", &n);
        auto chunk_row = [&](const uint8_t* chunks, uint64_t row, float* out) {   // a q4_1 chunk row, d / m in bf16
            for (unsigned kt = 0; kt < 1; ++kt)
                for (unsigned k = 0; k < 256; ++k) {
                    const uint8_t* c = chunks + (row / 32) * 5120;
                    const unsigned r = static_cast<unsigned>(row % 32), meta = (k / 32) * 32 + r;
                    uint16_t d, mn;
                    std::memcpy(&d, c + 2 * meta, 2);
                    std::memcpy(&mn, c + 512 + 2 * meta, 2);
                    const unsigned p = (r / 16) * 4096 + k * 16 + (r % 16);
                    const unsigned q = (c[1024 + (p >> 1)] >> (4 * (p & 1))) & 15u;
                    out[k] = bf16_to_f32(mn) + static_cast<float>(q) * bf16_to_f32(d);
                }
        };
        chunk_row(qc, 64, got.data());
        for (unsigned k = 0; k < 256 && ok; ++k) ok = std::fabs(got[k] - want[k]) <= 0.02f * (1.f + std::fabs(want[k]));
        v.gguf().embed_row("blk.3.attn_q.weight", 64, 256, want.data());
        chunk_row(qc, 256, got.data());
        for (unsigned k = 0; k < 256 && ok; ++k) ok = std::fabs(got[k] - want[k]) <= 0.02f * (1.f + std::fabs(want[k]));
        // ssm_a: grouped head 1 holds file head 2
        const auto a = v.f32("model.layers.0.linear_attn.ssm_a");
        std::vector<float> fa(4);
        v.gguf().embed_row("blk.0.ssm_a", 0, 4, fa.data());
        ok = ok && a.size() == 4 && a[0] == fa[0] && a[1] == fa[2] && a[2] == fa[1] && a[3] == fa[3];
        std::printf("%s q4nx view semantics (MTP-free block count, value-head untile, q | gate split)\n",
                    ok ? "ok  " : "FAIL");
        if (!ok) return 1;
        // OPEN-GGUF-QWEN35-SPEC: tests/test_qwen35_gguf_spec.py asserts the same values from the recipe side
        const nlohmann::json c = derive_config(v.gguf());
        const nlohmann::json want_cfg = {
            {"model_type", "qwen3_5_text"}, {"hidden_size", 256}, {"num_hidden_layers", 4}, {"vocab_size", 256},
            {"intermediate_size", 256}, {"head_dim", 64}, {"num_attention_heads", 4}, {"num_key_value_heads", 2},
            {"linear_num_value_heads", 4},
            {"layer_types", {"linear_attention", "linear_attention", "linear_attention", "full_attention"}}};
        bool cfg_ok = true;
        for (auto it = want_cfg.begin(); it != want_cfg.end(); ++it)
            cfg_ok = cfg_ok && c.contains(it.key()) && c[it.key()] == it.value();
        std::printf("%s derive_config: the qwen35 fields the kernel set checks %s\n", cfg_ok ? "ok  " : "FAIL",
                    c.dump().c_str());
        if (!cfg_ok) return 1;
    }

    std::printf("GGUF-PACK OK\n");
    return 0;
}
