/// \file gguf_view.cpp
/// \brief GGUF -> q4nx-form tensors for the qwen35 family (see gguf_view.hpp, OPEN-GGUF-Q4NX-VIEW).
#include "open_qwen36/gguf_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36 {

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("gguf view: " + what); }

constexpr size_t Q4 = 5120, Q8 = 8704;

inline unsigned code_at(unsigned r, unsigned k) { return (r / 16) * 4096 + k * 16 + (r % 16); }

uint64_t key(const GgufFile& g, const std::string& arch, const std::string& k) {
    if (!g.kv_has(arch + "." + k)) fail(g.kv_str("general.architecture") + " GGUF lacks " + arch + "." + k);
    return g.kv_u64(arch + "." + k);
}

void put_bf16(std::vector<uint8_t>& out, size_t i, float v) {
    const uint16_t b = f32_to_bf16(v);
    std::memcpy(out.data() + 2 * i, &b, 2);
}

/// Rows of a [rows, cols] GGUF tensor as f32, exactly as the file stores them.
std::vector<float> rows_f32(const GgufFile& g, const std::string& name, uint64_t rows, uint64_t cols) {
    const auto& t = g.tensor(name);
    const uint64_t have_rows = t.dims.size() >= 2 ? t.dims[1] : 1;
    if (t.dims[0] != cols || have_rows != rows)
        fail(name + " is [" + std::to_string(have_rows) + ", " + std::to_string(t.dims[0]) + "], the plan reads [" +
             std::to_string(rows) + ", " + std::to_string(cols) + "]");
    std::vector<float> v(rows * cols);
    for (uint64_t r = 0; r < rows; ++r) g.embed_row(name, r, cols, v.data() + r * cols);
    return v;
}

}  // namespace

uint64_t gguf_layer_count(const GgufFile& g) {
    const std::string a = g.kv_str("general.architecture");
    const uint64_t n = g.kv_u64(a + ".block_count");
    const uint64_t mtp = g.kv_has(a + ".nextn_predict_layers") ? g.kv_u64(a + ".nextn_predict_layers") : 0;
    if (mtp > n) fail(a + ".nextn_predict_layers exceeds block_count");
    return n - mtp;
}

nlohmann::json derive_config(const GgufFile& g) {
    const std::string arch = g.kv_str("general.architecture");
    auto u = [&](const std::string& k) { return g.kv_u64(arch + "." + k); };
    nlohmann::json c;
    c["model_type"] = arch;
    c["hidden_size"] = u("embedding_length");
    c["num_hidden_layers"] = u("block_count");
    if (g.kv_has(arch + ".vocab_size"))
        c["vocab_size"] = u("vocab_size");
    else
        c["vocab_size"] = g.tensor("token_embd.weight").dims.at(1);
    if (arch == "qwen35") {
        // recipes/spec.py _qwen35_gguf's derivation: the tower's model_type, MTP blocks out, the head padded to 256 rows
        const uint64_t blocks = gguf_layer_count(g);
        const uint64_t rows = g.tensor("token_embd.weight").dims.at(1);
        c["model_type"] = "qwen3_5_text";
        c["num_hidden_layers"] = blocks;
        c["vocab_size"] = (rows + 255) / 256 * 256;
        c["linear_num_value_heads"] = u("ssm.time_step_rank");
        const uint64_t every = u("full_attention_interval");
        if (!every) fail(arch + ".full_attention_interval is 0");
        nlohmann::json lt = nlohmann::json::array();
        for (uint64_t l = 0; l < blocks; ++l) lt.push_back((l + 1) % every == 0 ? "full_attention" : "linear_attention");
        c["layer_types"] = lt;
    }
    c["num_attention_heads"] = u("attention.head_count");
    c["num_key_value_heads"] = u("attention.head_count_kv");
    c["intermediate_size"] = u("feed_forward_length");
    if (g.kv_has(arch + ".attention.key_length")) c["head_dim"] = g.kv_u64(arch + ".attention.key_length");
    else c["head_dim"] = u("embedding_length") / u("attention.head_count");
    if (g.kv_has(arch + ".attention.sliding_window"))
        c["sliding_window"] = g.kv_u64(arch + ".attention.sliding_window");
    return c;
}

uint64_t gguf_view::file_head(uint64_t h, uint64_t num_k, uint64_t num_v) {
    const uint64_t grp = num_v / num_k;
    return (h % grp) * num_k + h / grp;
}

void gguf_view::requant_q8_block(const float* v, uint16_t* scale, int8_t* codes) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
    *scale = pools::bf16_ceil_bits(amax / 127.0f);
    const float s = bf16_to_f32(*scale);
    const float inv = s > 0.f ? 1.0f / s : 0.f;
    for (int i = 0; i < 32; ++i) {
        const float t = v[i] * inv;
        int c = static_cast<int>(std::floor(t + 0.5f));
        codes[i] = static_cast<int8_t>(std::min(127, std::max(-127, c)));
    }
}

size_t gguf_view::to_q4nx_chunks(const GgufFile& g, const std::string& name, uint64_t rows, uint64_t cols,
                                 const std::vector<uint64_t>* row_of, const std::vector<uint64_t>* blk_of,
                                 bool head, std::vector<uint8_t>& out) {
    const auto& t = g.tensor(name);
    if (t.dims.size() != 2 || t.dims[0] != cols)
        fail(name + " is not a [rows, " + std::to_string(cols) + "] matrix");
    if (rows % 32 || cols % 256) fail(name + ": [" + std::to_string(rows) + ", " + std::to_string(cols) + "] is not whole 32 x 256 chunks");
    const uint64_t src_rows = t.dims[1], nb = cols / 32, ncol = cols / 256;
    using T = GgufFile::Type;
    const bool q4 = !head && (t.type == T::Q4_1 || t.type == T::Q4_0);
    const bool q8 = t.type == T::Q8_0;
    const size_t ch = (head || q8) ? Q8 : Q4;
    out.assign((rows / 32) * ncol * ch, 0);
    size_t n = 0;
    const uint8_t* base = g.raw(name, &n);
    const size_t blk = t.type == T::Q4_1 ? 20 : t.type == T::Q4_0 ? 18 : 34;
    std::vector<float> rowv(cols);
    for (uint64_t r = 0; r < rows; ++r) {
        const uint64_t sr = row_of ? (*row_of)[r] : r;
        if (sr >= src_rows) continue;
        if (!q4 && !q8) g.embed_row(name, sr, cols, rowv.data());
        const uint8_t* srow = base + sr * nb * blk;
        const unsigned rl = static_cast<unsigned>(r % 32);
        for (uint64_t b = 0; b < nb; ++b) {
            const uint64_t sb = blk_of ? (*blk_of)[b] : b;
            uint8_t* c = out.data() + ((r / 32) * ncol + b / 8) * ch;
            const unsigned kb = static_cast<unsigned>(b % 8), meta = kb * 32 + rl;
            if (q8) {
                const uint8_t* p = srow + sb * blk;
                uint16_t d16;
                std::memcpy(&d16, p, 2);
                const uint16_t s = f32_to_bf16(fp16_to_f32(d16));
                std::memcpy(c + 2 * meta, &s, 2);
                for (unsigned i = 0; i < 32; ++i) c[512 + code_at(rl, kb * 32 + i)] = p[2 + i];
            } else if (head) {
                uint16_t s;
                int8_t codes[32];
                requant_q8_block(rowv.data() + sb * 32, &s, codes);
                std::memcpy(c + 2 * meta, &s, 2);
                for (unsigned i = 0; i < 32; ++i) c[512 + code_at(rl, kb * 32 + i)] = static_cast<uint8_t>(codes[i]);
            } else {
                uint16_t du, mu;
                uint8_t q[32];
                if (q4) {
                    const uint8_t* p = srow + sb * blk;
                    uint16_t d16, m16 = 0;
                    std::memcpy(&d16, p, 2);
                    const float df = fp16_to_f32(d16);
                    float mf = -8.f * df;
                    if (t.type == T::Q4_1) {
                        std::memcpy(&m16, p + 2, 2);
                        mf = fp16_to_f32(m16);
                    }
                    du = f32_to_bf16(df);
                    mu = f32_to_bf16(mf);
                    const uint8_t* qs = p + (t.type == T::Q4_1 ? 4 : 2);
                    for (unsigned i = 0; i < 16; ++i) {
                        q[i] = qs[i] & 0xF;
                        q[i + 16] = qs[i] >> 4;
                    }
                } else {
                    pools::requant_block_q4_1(rowv.data() + sb * 32, &du, &mu, q);
                }
                std::memcpy(c + 2 * meta, &du, 2);
                std::memcpy(c + 512 + 2 * meta, &mu, 2);
                for (unsigned i = 0; i < 32; ++i) {
                    const unsigned p = code_at(rl, kb * 32 + i);
                    c[1024 + (p >> 1)] |= static_cast<uint8_t>((p & 1) ? (q[i] << 4) : q[i]);
                }
            }
        }
    }
    return ch;
}

bool GgufQ4nxView::supports(const std::string& arch) { return arch == "qwen35"; }

GgufQ4nxView::GgufQ4nxView(const std::string& path) : GgufQ4nxView(std::make_unique<GgufFile>(path)) {}

GgufQ4nxView::GgufQ4nxView(std::unique_ptr<GgufFile> g) : g_(std::move(g)) {
    const std::string a = g_->kv_str("general.architecture");
    if (!supports(a)) fail("architecture " + a + " has no q4nx view (have qwen35)");
    geo_.hidden = key(*g_, a, "embedding_length");
    geo_.heads = key(*g_, a, "attention.head_count");
    geo_.kv_heads = key(*g_, a, "attention.head_count_kv");
    geo_.head_dim = key(*g_, a, "attention.key_length");
    geo_.lin_k_heads = key(*g_, a, "ssm.group_count");
    geo_.lin_v_heads = key(*g_, a, "ssm.time_step_rank");
    geo_.lin_k_dim = key(*g_, a, "ssm.state_size");
    geo_.lin_v_dim = key(*g_, a, "ssm.inner_size") / geo_.lin_v_heads;
    geo_.conv = key(*g_, a, "ssm.conv_kernel");
    if (geo_.lin_v_heads % geo_.lin_k_heads) fail("value heads are not a whole multiple of key heads");
    geo_.blocks = gguf_layer_count(*g_);
}

namespace {

struct Rule {
    const char* suffix;    // q4nx name after "model.layers.N."
    const char* gguf;      // llama.cpp's, after "blk.N."
};

const Rule kRules[] = {
    {"input_layernorm.weight", "attn_norm.weight"},
    {"post_attention_layernorm.weight", "post_attention_norm.weight"},
    {"self_attn.q_proj.weight", "attn_q.weight"},
    {"self_attn.k_proj.weight", "attn_k.weight"},
    {"self_attn.v_proj.weight", "attn_v.weight"},
    {"self_attn.o_proj.weight", "attn_output.weight"},
    {"self_attn.q_norm.weight", "attn_q_norm.weight"},
    {"self_attn.k_norm.weight", "attn_k_norm.weight"},
    {"self_attn.gate_proj.weight", "attn_gate.weight"},
    {"linear_attn.qkv_proj.weight", "attn_qkv.weight"},
    {"linear_attn.ssm_out_proj.weight", "ssm_out.weight"},
    {"linear_attn.ssm_alpha_proj.bf16.weight", "ssm_alpha.weight"},
    {"linear_attn.ssm_beta_proj.bf16.weight", "ssm_beta.weight"},
    {"linear_attn.ssm_a", "ssm_a"},
    {"linear_attn.ssm_dt.bias", "ssm_dt.bias"},
    {"linear_attn.ssm_conv1d.weight", "ssm_conv1d.weight"},
    {"linear_attn.ssm_norm.weight", "ssm_norm.weight"},
    {"mlp.up_proj.weight", "ffn_up.weight"},
    {"mlp.gate_proj.weight", "ffn_gate.weight"},
    {"mlp.down_proj.weight", "ffn_down.weight"},
};

bool split_layer(const std::string& name, uint64_t* layer, std::string* suffix) {
    const std::string pre = "model.layers.";
    if (name.rfind(pre, 0) != 0) return false;
    const size_t dot = name.find('.', pre.size());
    if (dot == std::string::npos) return false;
    try {
        *layer = std::stoull(name.substr(pre.size(), dot - pre.size()));
    } catch (const std::logic_error&) {
        return false;
    }
    *suffix = name.substr(dot + 1);
    return true;
}

}  // namespace

std::string GgufQ4nxView::source_name(const std::string& name) const {
    if (name == "model.embed_tokens.weight") return "token_embd.weight";
    if (name == "model.norm.weight") return "output_norm.weight";
    if (name == "lm_head.weight") return g_->has("output.weight") ? "output.weight" : "token_embd.weight";
    uint64_t l = 0;
    std::string sfx;
    if (!split_layer(name, &l, &sfx) || l >= geo_.blocks) return "";
    for (const Rule& r : kRules)
        if (sfx == r.suffix) return "blk." + std::to_string(l) + "." + r.gguf;
    return "";
}

bool GgufQ4nxView::has(const std::string& name) const {
    const std::string s = source_name(name);
    return !s.empty() && g_->has(s);
}

GgufQ4nxView::Tensor GgufQ4nxView::build(const std::string& name) const {
    const std::string src = source_name(name);
    if (src.empty() || !g_->has(src)) fail("no tensor " + name + " (GGUF " + (src.empty() ? "?" : src) + ")");
    const GgufQwen35Geometry& G = geo_;
    Tensor t;
    auto chunks = [&](uint64_t rows, uint64_t cols, const std::vector<uint64_t>* row_of,
                      const std::vector<uint64_t>* blk_of, bool head) {
        const size_t ch = gguf_view::to_q4nx_chunks(*g_, src, rows, cols, row_of, blk_of, head, t.bytes);
        t.meta.dtype = "I8";
        t.meta.shape = {t.bytes.size() / ch, ch};
    };
    auto small = [&](const std::vector<float>& v, std::vector<size_t> shape, bool as_f32) {
        t.meta.shape = std::move(shape);
        t.meta.dtype = as_f32 ? "F32" : "BF16";
        t.bytes.assign(v.size() * (as_f32 ? 4 : 2), 0);
        for (size_t i = 0; i < v.size(); ++i) {
            if (as_f32) std::memcpy(t.bytes.data() + 4 * i, &v[i], 4);
            else put_bf16(t.bytes, i, v[i]);
        }
    };
    const uint64_t vh = G.lin_v_heads, kh = G.lin_k_heads, dv = G.lin_v_dim, qk = 2 * kh * G.lin_k_dim;
    auto value_rows = [&](uint64_t first, uint64_t unit) {
        std::vector<uint64_t> m(first + vh * unit);
        for (uint64_t r = 0; r < first; ++r) m[r] = r;
        for (uint64_t h = 0; h < vh; ++h)
            for (uint64_t p = 0; p < unit; ++p) m[first + h * unit + p] = first + gguf_view::file_head(h, kh, vh) * unit + p;
        return m;
    };
    if (name == "lm_head.weight") {
        // the vocabulary the spec derivation pads to (recipes/spec.py _qwen35_gguf), whole 128-row bands
        const uint64_t rows = g_->tensor(src).dims.at(1);
        chunks((rows + 255) / 256 * 256, G.hidden, nullptr, nullptr, true);
        return t;
    }
    if (name == "model.norm.weight") {
        small(rows_f32(*g_, src, 1, G.hidden), {G.hidden}, false);
        return t;
    }
    if (name == "model.embed_tokens.weight") {
        t.meta.dtype = "BF16";
        t.meta.shape = {static_cast<size_t>(g_->tensor(src).dims.at(1)), static_cast<size_t>(G.hidden)};
        return t;
    }
    uint64_t l = 0;
    std::string sfx;
    split_layer(name, &l, &sfx);
    const uint64_t qw = G.heads * G.head_dim, kvw = G.kv_heads * G.head_dim, vw = vh * dv, nch = qk + vw;
    const uint64_t ff = g_->has("blk." + std::to_string(l) + ".ffn_up.weight")
                            ? g_->tensor("blk." + std::to_string(l) + ".ffn_up.weight").dims.at(1) : 0;
    if (sfx == "self_attn.q_proj.weight") {
        std::vector<uint64_t> m(2 * qw);
        for (uint64_t r = 0; r < 2 * qw; ++r) {
            const uint64_t part = r / qw, rr = r % qw, h = rr / G.head_dim, i = rr % G.head_dim;
            m[r] = h * 2 * G.head_dim + part * G.head_dim + i;   // llama.cpp keeps [q_h | gate_h] per head
        }
        chunks(2 * qw, G.hidden, &m, nullptr, false);
    } else if (sfx == "self_attn.k_proj.weight" || sfx == "self_attn.v_proj.weight") {
        chunks(kvw, G.hidden, nullptr, nullptr, false);
    } else if (sfx == "self_attn.o_proj.weight") {
        chunks(G.hidden, qw, nullptr, nullptr, false);
    } else if (sfx == "self_attn.gate_proj.weight") {
        const auto m = value_rows(0, dv);
        chunks(vw, G.hidden, &m, nullptr, false);
    } else if (sfx == "linear_attn.qkv_proj.weight") {
        const auto m = value_rows(qk, dv);
        chunks(nch, G.hidden, &m, nullptr, false);
    } else if (sfx == "linear_attn.ssm_out_proj.weight") {
        if (dv % 32) fail(name + ": the value head dim is not whole 32-column blocks");
        const auto m = value_rows(0, dv / 32);
        chunks(G.hidden, vw, nullptr, &m, false);
    } else if (sfx == "mlp.up_proj.weight" || sfx == "mlp.gate_proj.weight") {
        chunks(ff, G.hidden, nullptr, nullptr, false);
    } else if (sfx == "mlp.down_proj.weight") {
        chunks(G.hidden, ff, nullptr, nullptr, false);
    } else if (sfx == "linear_attn.ssm_alpha_proj.bf16.weight" || sfx == "linear_attn.ssm_beta_proj.bf16.weight") {
        const auto v = rows_f32(*g_, src, vh, G.hidden);
        std::vector<float> o(v.size());
        for (uint64_t h = 0; h < vh; ++h)
            std::memcpy(o.data() + h * G.hidden, v.data() + gguf_view::file_head(h, kh, vh) * G.hidden, G.hidden * 4);
        small(o, {vh, G.hidden}, false);
    } else if (sfx == "linear_attn.ssm_a" || sfx == "linear_attn.ssm_dt.bias") {
        const auto v = rows_f32(*g_, src, 1, vh);
        std::vector<float> o(vh);
        for (uint64_t h = 0; h < vh; ++h) o[h] = v[gguf_view::file_head(h, kh, vh)];
        small(o, {vh}, true);
    } else if (sfx == "linear_attn.ssm_conv1d.weight") {
        const auto v = rows_f32(*g_, src, nch, G.conv);
        const auto m = value_rows(qk, dv);
        std::vector<float> o(G.conv * nch);
        for (uint64_t c = 0; c < nch; ++c)
            for (uint64_t k = 0; k < G.conv; ++k) o[k * nch + c] = v[m[c] * G.conv + k];   // the container is [taps, channels]
        small(o, {G.conv, nch}, false);
    } else if (sfx == "linear_attn.ssm_norm.weight") {
        small(rows_f32(*g_, src, 1, dv), {dv}, false);
    } else if (sfx == "self_attn.q_norm.weight" || sfx == "self_attn.k_norm.weight") {
        small(rows_f32(*g_, src, 1, G.head_dim), {G.head_dim}, false);
    } else {
        small(rows_f32(*g_, src, 1, G.hidden), {G.hidden}, false);
    }
    return t;
}

const GgufQ4nxView::Tensor& GgufQ4nxView::get(const std::string& name) const {
    auto it = cache_.find(name);
    if (it == cache_.end()) {
        // a pointer from raw() must outlive the next few reads (an op holds at most two tensors)
        while (cache_.size() >= 8) {
            auto old = std::min_element(cache_.begin(), cache_.end(),
                                        [](const auto& a, const auto& b) { return a.second.used < b.second.used; });
            cache_.erase(old);
        }
        it = cache_.emplace(name, build(name)).first;
    }
    it->second.used = ++clock_;
    return it->second;
}

const uint8_t* GgufQ4nxView::raw(const std::string& name, size_t* nbytes) const {
    if (name == "model.embed_tokens.weight") fail("the embedding is read by row (embed_row), not whole");
    const Tensor& t = get(name);
    if (nbytes) *nbytes = t.bytes.size();
    return t.bytes.data();
}

size_t GgufQ4nxView::chunk_bytes(const std::string& name) const {
    const TensorMeta& m = meta(name);
    return m.dtype == "I8" && !m.shape.empty() ? m.shape.back() : 0;
}

const TensorMeta& GgufQ4nxView::meta(const std::string& name) const { return get(name).meta; }

std::vector<float> GgufQ4nxView::bf16(const std::string& name) const {
    const Tensor& t = get(name);
    if (t.meta.dtype != "BF16") fail(name + " is " + t.meta.dtype + ", not BF16");
    std::vector<float> out(t.bytes.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        uint16_t u;
        std::memcpy(&u, t.bytes.data() + 2 * i, 2);
        out[i] = bf16_to_f32(u);
    }
    return out;
}

std::vector<float> GgufQ4nxView::f32(const std::string& name) const {
    const Tensor& t = get(name);
    if (t.meta.dtype != "F32") fail(name + " is " + t.meta.dtype + ", not F32");
    std::vector<float> out(t.bytes.size() / 4);
    std::memcpy(out.data(), t.bytes.data(), t.bytes.size());
    return out;
}

void GgufQ4nxView::embed_row(const std::string& name, size_t row, size_t dim, float* out) const {
    const std::string src = source_name(name);
    if (src.empty()) fail("no tensor " + name);
    const uint64_t cols = g_->tensor(src).dims.at(0);
    if (dim < cols) fail(name + " rows are " + std::to_string(cols) + " wide, the caller asked for " + std::to_string(dim));
    g_->embed_row(src, row, cols, out);
    for (size_t i = cols; i < dim; ++i) out[i] = 0.f;
}

void GgufQ4nxView::bf16_row(const std::string& name, size_t row, size_t dim, float* out) const {
    embed_row(name, row, dim, out);
}

void GgufQ4nxView::drop_pages() {
    cache_.clear();
    g_->drop_pages();
}

}  // namespace open_qwen36
