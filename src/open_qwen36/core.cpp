/// \file core.cpp
/// \brief The resident open-kernel decode engine: a manifest interpreter (see core.hpp).
#include "open_qwen36/core.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_xclbin.h"

#include "open_qwen36/block_host.hpp"

namespace open_qwen36 {

namespace fs = std::filesystem;

namespace {

// 0167/#32: the GEMM route's host math reuses this file's own
// open_qwen36::bf16_to_f32 / open_qwen36::f32_to_bf16 (q4nx_file.hpp) --
// both already round-to-nearest-even, matching open_npue/npue_pack.cpp's
// bf16_rne and ml_dtypes.bfloat16's cast exactly (checked: same bit
// arithmetic, `u + 0x7FFF + ((u>>16)&1)`). NOT open_npue's own tile_b,
// which has internal (anonymous-namespace) linkage and is not declared in
// its header, so it is not callable from here; its algorithm is reproduced
// in tile_gemm_x() below instead, using these two conversions.

constexpr int kOpcode = 3;
constexpr size_t kBoAlign = 1u << 20;  // XDNA wants 1 MB-aligned buffer sizes
size_t padup(size_t n) { return (n + kBoAlign - 1) / kBoAlign * kBoAlign; }

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open_qwen36: cannot read " + p.string());
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> v(static_cast<size_t>(n));
    if (n > 0 && !f.read(reinterpret_cast<char*>(v.data()), n)) throw std::runtime_error("open_qwen36: short read " + p.string());
    return v;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void Core::log(const std::string& s) const {
    if (cfg_.verbose) std::fprintf(stderr, "open_qwen36: %s\n", s.c_str());
}

Core::Core(const CoreConfig& cfg, xrt::device* dev) : cfg_(cfg) {
    // ---- the kernel set's manifest, and the model it must agree with
    man_ = Manifest::load((fs::path(cfg_.kernel_dir) / "manifest.json").string());
    fs::path md(cfg_.model_dir);
    std::ifstream cf(md / "config.json");
    if (!cf) throw std::runtime_error("open_qwen36: no config.json in " + cfg_.model_dir);
    auto j = nlohmann::json::parse(cf, nullptr, false);
    if (!j.is_object()) throw std::runtime_error("open_qwen36: bad config.json in " + cfg_.model_dir);
    man_.check_model(j, md.filename().string());
    // The VLM bits: the image token the app expands per merged patch, and how the
    // rotary pairs split over (t, h, w). Absent on text-only models.
    image_token_id_ = j.value("image_token_id", -1);
    if (j.contains("rope_parameters") && j["rope_parameters"].is_object()) {
        const auto& rp = j["rope_parameters"];
        if (rp.contains("mrope_section") && rp["mrope_section"].is_array() && rp["mrope_section"].size() == 3) {
            size_t sum = 0;
            for (const auto& v : rp["mrope_section"]) { mrope_section_.push_back(v.get<int>()); sum += v.get<int>(); }
            mrope_interleaved_ = rp.value("mrope_interleaved", false);
            if (sum != man_.rotary_dim / 2)
                throw std::runtime_error("open_qwen36: mrope_section sums to " + std::to_string(sum) + ", not the " +
                                         std::to_string(man_.rotary_dim / 2) + " rotary pairs");
        }
    }
    int total = static_cast<int>(man_.layers.size());
    nl_ = cfg_.num_layers > 0 && cfg_.num_layers < total ? cfg_.num_layers : total;
    types_.resize(nl_);
    for (int l = 0; l < nl_; ++l) types_[l] = &man_.layer_type(l);
    file_ = std::make_unique<Q4nxFile>((md / "model.q4nx").string());
    int nattn = 0;
    for (int l = 0; l < nl_; ++l) nattn += is_attention_layer(l);
    log("model " + md.filename().string() + " (" + man_.family + ", " + man_.spec_hash.substr(0, 19) + "): " +
        std::to_string(nl_) + " of " + std::to_string(total) + " layers, " + std::to_string(nattn) +
        " attention, context capacity " + std::to_string(cfg_.max_ctx));

    // ---- device, contexts, kernels (only the kernels the running layers' programs and the tail name)
    if (dev) {
        dev_ = dev;
    } else {
        owned_dev_ = std::make_unique<xrt::device>(0u);
        dev_ = owned_dev_.get();
    }
    std::map<std::string, bool> wanted;
    for (int l = 0; l < nl_; ++l) {
        for (const auto& s : types_[l]->program) wanted[s.kernel] = true;
        // 0167/#32: the GEMM-route block's 5 GEMM kernels, plus "dxB"
        // (the attention half, driven directly by Core rather than via a
        // Step -- see manifest.hpp's GemmBlockProgram) which the manifest
        // parser already required to exist whenever gemm_block is present.
        for (const auto& s : types_[l]->gemm_block.program) wanted[s.kernel] = true;
        for (const auto& s : types_[l]->gemm_block.shared_program) wanted[s.kernel] = true;
        if (types_[l]->gemm_block.t && types_[l]->gemm_block.kind == "dense") wanted["dxB"] = true;
        if (!types_[l]->gemm_block.moe_kernel.empty()) wanted[types_[l]->gemm_block.moe_kernel] = true;
    }
    for (const auto& s : man_.tail) wanted[s.kernel] = true;
    for (const auto& [name, d] : man_.kernels)
        if (wanted.count(name)) load_kernel(name, d);
    logits_host_.assign(man_.vocab, 0.f);

    // 0167/#32: the GEMM-route block size every loaded layer type agrees on.
    // Disagreement (a mixed dense_local/dense manifest where only one carries
    // a gemm_block program) or no gemm_block program anywhere both read as
    // "unsupported" (0), never a guess at which layer type's T applies --
    // step_gemm_block() refuses. gemm_block_t_ == 0 is not an error (most
    // kernel sets have no gemm_block program), but it silently means every
    // prefill runs one token at a time even on a model that DOES have one, if
    // the kernel_dir actually loaded is a stale copy without it
    // (Engine::find_kernels() prefers <model>/open_kernels over
    // FLM_OPEN_KERNELS_DIR/FLM_XCLBIN_PATH -- this project lost real time to
    // exactly that before this log line existed).
    gemm_block_t_ = nl_ > 0 ? types_[0]->gemm_block.t : 0;
    for (int l = 1; l < nl_; ++l)
        if (types_[l]->gemm_block.t != gemm_block_t_) { gemm_block_t_ = 0; break; }
    log("block prefill route: T = " + std::to_string(gemm_block_t_) +
        (gemm_block_t_ ? "" : " (no gemm_block program in this kernel set, or its layer types disagree)"));
}

Core::~Core() = default;

xrt::hw_context& Core::context(const std::string& name) {
    auto it = ctxs_.find(name);
    if (it != ctxs_.end()) return *it->second;
    fs::path p = fs::path(cfg_.kernel_dir) / man_.contexts.at(name);
    if (!fs::exists(p)) throw std::runtime_error("open_qwen36: missing kernel " + p.string());
    xrt::xclbin xcl(p.string());
    auto uuid = dev_->register_xclbin(xcl);
    auto ctx = std::make_unique<xrt::hw_context>(*dev_, uuid);
    return *(ctxs_[name] = std::move(ctx));
}

void Core::load_kernel(const std::string& name, const KernelDesc& d) {
    fs::path p = fs::path(cfg_.kernel_dir) / d.insts;
    if (!fs::exists(p)) throw std::runtime_error("open_qwen36: missing instruction stream " + p.string());
    Kern& k = kerns_[name];
    k.name = name;
    k.patch = d.patch;
    k.k = std::make_unique<xrt::kernel>(context(d.context), "MLIR_AIE");
    auto insts = read_file(p);
    if (insts.empty() || insts.size() % 4) throw std::runtime_error("open_qwen36: " + p.string() + " is not word-sized");
    k.words.resize(insts.size() / 4);
    std::memcpy(k.words.data(), insts.data(), insts.size());
    k.instr = std::make_unique<xrt::bo>(*dev_, insts.size(), xrt::bo::flags::cacheable, k.k->group_id(1));
    std::memcpy(k.instr->map<void*>(), insts.data(), insts.size());
    k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (d.patch == "moeroute2") k.moe2 = stream_patch::moe2_table(k.words, name, man_.moe);
    else if (d.patch == "attnpos") {
        k.attn = stream_patch::attn_table(k.words, name, man_.attn);
        k.geom = man_.attn;
        k.geom.window = d.window;
    }
}

xrt::bo Core::alloc(size_t bytes, const uint8_t* init, size_t init_bytes) {
    xrt::bo bo = xrt::ext::bo(*dev_, padup(bytes));
    auto* m = bo.map<uint8_t*>();
    std::memset(m, 0, padup(bytes));
    if (init) std::memcpy(m, init, init_bytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
}

void Core::load_weights(const std::function<void(int, int)>& progress) {
    auto t0 = std::chrono::steady_clock::now();
    pools_.clear(); consts_.clear(); act_.clear(); state_.clear(); globals_.clear();
    gemm_w_.clear(); hc_.clear();
    ln_w_bf16_.clear(); post_ln_w_bf16_.clear();
    pools_.reserve(nl_); consts_.reserve(nl_); act_.reserve(nl_); state_.reserve(nl_);
    if (gemm_block_t_) {
        ln_w_bf16_.resize(nl_); post_ln_w_bf16_.resize(nl_);
        hc_.resize(nl_);
    }
    // the block route's per-layer weight buffers: each a contiguous run of pack ops of
    // the freshly packed host bytes (pool or consts), copied before that buffer's upload
    auto build_weights = [&](const LayerType& lt, int l, const std::string& from, const uint8_t* host) {
        if (!(gemm_block_t_ && lt.gemm_block.t)) return;
        std::map<std::string, GemmWeight> all = lt.gemm_block.weights;
        all.insert(lt.gemm_block.shared_weights.begin(), lt.gemm_block.shared_weights.end());
        for (const auto& [name, gw] : all) {
            if (gw.from != from) continue;
            size_t off0 = 0, total = 0;
            for (size_t i = 0; i < gw.ops.size(); ++i) {
                auto [off, bytes] = op_region(lt, from, gw.ops[i]);
                if (i == 0) off0 = off;
                else if (off != off0 + total)
                    throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + ": the " + from + " ops of " + name +
                                             " are not contiguous -- the route needs one memcpy per weight buffer");
                total += bytes;
            }
            xrt::bo w = xrt::ext::bo(*dev_, padup(total));
            std::memcpy(w.map<uint8_t*>(), host + off0, total);
            w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto& v = gemm_w_[name];
            if (v.size() != static_cast<size_t>(nl_)) v.resize(nl_);
            v[l] = std::move(w);
        }
    };
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        xrt::bo pool = xrt::ext::bo(*dev_, man_.pool_bytes);
        uint8_t* pool_host = pool.map<uint8_t*>();
        pools::pack_pool(man_, lt, *file_, l, pool_host);
        build_weights(lt, l, "pool", pool_host);
        pool.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        pools_.push_back(std::move(pool));
        xrt::bo c = xrt::ext::bo(*dev_, padup(lt.consts_bytes));
        std::memset(c.map<uint8_t*>(), 0, padup(lt.consts_bytes));
        uint8_t* c_host = c.map<uint8_t*>();
        pools::pack_consts(man_, lt, *file_, l, c_host);
        build_weights(lt, l, "consts", c_host);
        if (gemm_block_t_ && lt.gemm_block.t) {
            if (lt.gemm_block.kind == "dense") {
                // the dense route's host RMSNorm reads the two norm weights as bf16;
                // dense.py's consts plan puts input_layernorm at byte 0 and
                // post_attention_layernorm right after it (ELN = hidden * 2 each)
                ln_w_bf16_[l].resize(man_.hidden);
                post_ln_w_bf16_[l].resize(man_.hidden);
                std::memcpy(ln_w_bf16_[l].data(), c_host, man_.hidden * 2);
                std::memcpy(post_ln_w_bf16_[l].data(), c_host + man_.hidden * 2, man_.hidden * 2);
            } else {
                // the MoE kinds' host stages read their small tensors straight from the
                // file, by the names the consts plan carries (no consts layout knowledge here)
                const GemmBlockProgram& gb = lt.gemm_block;
                HostConsts& h = hc_[l];
                auto bf = [&](const char* suffix) { return file_->bf16(const_tensor(lt, suffix, l)); };
                auto want = [&](const std::vector<float>& v, size_t n, const char* what) {
                    if (v.size() != n)
                        throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + ": " + what + " has " +
                                                 std::to_string(v.size()) + " values, the route wants " + std::to_string(n));
                };
                h.ln = bf("input_layernorm.weight");
                h.postln = bf("post_attention_layernorm.weight");
                h.router = bf("moe_router.weight");
                h.sgw = bf("shared_expert_gate.weight");
                want(h.ln, man_.hidden, "input_layernorm");
                want(h.postln, man_.hidden, "post_attention_layernorm");
                want(h.router, man_.hidden * man_.moe.experts, "moe_router");
                want(h.sgw, man_.hidden, "shared_expert_gate");
                if (gb.kind == "linear") {
                    const std::string wa = const_tensor(lt, "ssm_alpha_proj.weight", l);
                    const auto& shape = file_->meta(wa).shape;
                    if (shape.size() != 2 || shape[0] != man_.hidden)
                        throw std::runtime_error("open_qwen36: " + wa + " is not [hidden, lanes]");
                    h.lanes = shape[1];
                    h.Wa = file_->bf16(wa);
                    h.Wb = bf("ssm_beta_proj.weight");
                    h.A = file_->f32(const_tensor(lt, "ssm_a", l));
                    h.dtb = file_->f32(const_tensor(lt, "ssm_dt.bias", l));
                    h.convw = bf("ssm_conv1d.weight");
                    h.nw = bf("ssm_norm.weight");
                    want(h.Wb, man_.hidden * h.lanes, "ssm_beta_proj");
                    want(h.A, gb.value_heads, "ssm_a");
                    want(h.dtb, gb.value_heads, "ssm_dt.bias");
                    want(h.convw, gb.conv_kernel * gb.qkv_dim, "ssm_conv1d");
                    want(h.nw, gb.head_dim, "ssm_norm");
                } else {
                    h.qn = bf("q_norm.weight");
                    h.kn = bf("k_norm.weight");
                    want(h.qn, gb.hd, "q_norm");
                    want(h.kn, gb.hd, "k_norm");
                }
            }
        }
        c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        consts_.push_back(std::move(c));
        act_.push_back(alloc(lt.act_bytes));
        state_.push_back(alloc(lt.state_kind == "kv" ? cfg_.max_ctx * lt.state_row : lt.state_bytes));
        if (progress) progress(l + 1, nl_ + 1);
        if ((l + 1) % 10 == 0 || l + 1 == nl_)
            log(std::to_string(l + 1) + "/" + std::to_string(nl_) + " layers resident (" +
                std::to_string(static_cast<int>(ms_since(t0) / 1000)) + " s)");
    }
    // ---- the globals: the lm_head pool and the final norm's weight from the file, the ptab
    // computed, everything else zero (xres, zero, xresf, hn, logits)
    for (const auto& [name, bytes] : man_.globals) {
        if (name == "lmpool") {
            xrt::bo lm = xrt::ext::bo(*dev_, bytes);
            pools::pack_lmhead(man_, *file_, lm.map<uint8_t*>());
            lm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            globals_[name] = std::move(lm);
        } else if (name == "normw") {
            size_t n = 0;
            const uint8_t* nw = file_->raw(man_.norm_tensor, &n);
            if (n != man_.norm_bytes) throw std::runtime_error("open_qwen36: " + man_.norm_tensor + " is not " + std::to_string(man_.norm_bytes) + " B");
            globals_[name] = alloc(bytes, nw, n);
        } else {
            globals_[name] = alloc(bytes);
        }
    }
    for (const auto& [name, rg] : man_.per_row_globals) {
        std::vector<uint8_t> pt(cfg_.max_ctx * rg.per_row);
        pools::build_ptab(man_, rg, cfg_.max_ctx, pt.data());
        globals_[name] = alloc(pt.size(), pt.data(), pt.size());
    }
    file_->drop_pages();  // the packers are done with the container; keep only what the steps touch
    if (progress) progress(nl_ + 1, nl_ + 1);
    weights_loaded_ = true;
    pos_ = 0;
    log("weights resident: " + std::to_string(nl_) + " pools + lm_head, " +
        std::to_string(static_cast<int>(ms_since(t0) / 1000)) + " s");
}

void Core::reset() {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: reset before load_weights");
    // The linear layers' state must start at zero. The KV rows need not: the
    // window read is [0, max(pos, 1)) and row 0 at position 0 is a dummy the
    // kernel masks.
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        if (lt.state_kind != "linear") continue;
        std::memset(state_[l].map<uint8_t*>(), 0, lt.state_bytes);
        state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
    }
    // A request with an image rewrote the position records of the rows it used
    // (write_record); the next request expects row p to say position p again.
    if (ptab_dirty_) {
        for (const auto& [name, rg] : man_.per_row_globals) {
            xrt::bo& bo = globals_.at(name);
            pools::build_ptab(man_, rg, ptab_dirty_, bo.map<uint8_t*>());
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, ptab_dirty_ * rg.per_row, 0);
        }
        ptab_dirty_ = 0;
    }
    mrope_on_ = false;
    mrope_pos_ = 0;
    pos_ = 0;
}

xrt::bo& Core::buffer(const std::string& name, int layer) {
    if (name == "pool") return pools_[layer];
    if (name == "consts") return consts_[layer];
    if (name == "act") return act_[layer];
    if (name == "state") return state_[layer];
    // the block route's per-layer weight buffers (load_weights)
    if (auto g = gemm_w_.find(name); g != gemm_w_.end()) {
        if (layer < 0 || static_cast<size_t>(layer) >= g->second.size() || !g->second[layer])
            throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " has no '" + name + "' (gemm_block) buffer");
        return g->second[layer];
    }
    auto it = globals_.find(name);
    if (it == globals_.end()) throw std::runtime_error("open_qwen36: the program names an unknown buffer '" + name + "'");
    return it->second;
}

double Core::run(Kern& k, const std::vector<std::string>& args, int layer) {
    auto t0 = std::chrono::steady_clock::now();
    xrt::run r(*k.k);
    r.set_arg(0, kOpcode);
    r.set_arg(1, *k.instr);
    r.set_arg(2, static_cast<int>(k.words.size()));
    int i = 3;
    for (const auto& a : args) r.set_arg(i++, buffer(a, layer));
    r.start();
    auto st = cfg_.timeout_ms ? r.wait(std::chrono::milliseconds(cfg_.timeout_ms)) : r.wait();
    if (st != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("open_qwen36: kernel " + k.name + " at position " + std::to_string(pos_) +
                                 " ended in ERT state " + std::to_string(static_cast<int>(st)) +
                                 (st == ERT_CMD_STATE_TIMEOUT ? " (timeout)" : ""));
    return ms_since(t0);
}

void Core::route(Kern& k, int layer, uint64_t act_off) {
    auto t0 = std::chrono::steady_clock::now();
    if (k.moe2.empty()) throw std::runtime_error("open_qwen36: moeroute2 on " + k.name + ", which has no routed-expert table");
    xrt::bo& act = act_[layer];
    const size_t off = act_off + man_.rout_idx_off;
    act.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 32, off);
    uint32_t idx[8];
    std::memcpy(idx, act.map<uint8_t*>() + off, 32);
    for (unsigned s = 0; s < man_.moe.topk; ++s)
        if (idx[s] >= man_.moe.experts) throw std::runtime_error("open_qwen36: router produced expert index " + std::to_string(idx[s]));
    stream_patch::moe2_apply(k.iw(), k.moe2, idx, man_.moe);
    k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    timing_.route_ms += ms_since(t0);
}

void Core::step(int token, bool want_logits) { step_impl(token, nullptr, want_logits, nullptr); }

void Core::step_embed(const float* x, bool want_logits, const int64_t mpos[3]) {
    if (!has_mrope()) throw std::runtime_error("open_qwen36: step_embed on a model without M-RoPE");
    step_impl(-1, x, want_logits, mpos);
}

void Core::mrope_begin() {
    if (mrope_on_) return;
    mrope_on_ = true;
    mrope_pos_ = pos_;
}

void Core::write_record(size_t row, const double pos[3]) {
    for (const auto& [name, rg] : man_.per_row_globals) {
        xrt::bo& bo = globals_.at(name);
        uint8_t* r = bo.map<uint8_t*>() + row * rg.per_row;
        pools::build_ptab_record(man_, rg, row, pos, mrope_section_, mrope_interleaved_, r);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, rg.per_row, row * rg.per_row);
    }
    if (row + 1 > ptab_dirty_) ptab_dirty_ = row + 1;
}

void Core::step_impl(int token, const float* x, bool want_logits, const int64_t* mpos) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: step before load_weights");
    if (static_cast<size_t>(pos_) >= cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: position " + std::to_string(pos_) + " reached the context capacity " +
                                 std::to_string(cfg_.max_ctx));
    if (!x && (token < 0 || static_cast<size_t>(token) >= man_.vocab)) throw std::runtime_error("open_qwen36: token id out of range");
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    xrt::bo& xres = buffer("xres", 0);
    if (x)
        std::memcpy(xres.map<float*>(), x, man_.hidden * 4);
    else
        file_->bf16_row(man_.embed_tensor, static_cast<size_t>(token), man_.hidden, xres.map<float*>());
    xres.sync(XCL_BO_SYNC_BO_TO_DEVICE, man_.hidden * 4, 0);
    if (mpos) {
        const double p3[3] = {static_cast<double>(mpos[0]), static_cast<double>(mpos[1]), static_cast<double>(mpos[2])};
        write_record(static_cast<size_t>(pos_), p3);
    } else if (mrope_on_) {
        const double cpos = static_cast<double>(mrope_pos_);
        const double p3[3] = {cpos, cpos, cpos};
        write_record(static_cast<size_t>(pos_), p3);
    }
    for (auto& [name, k] : kerns_) {
        if (k.patch != "attnpos") continue;
        stream_patch::attn_apply(k.iw(), k.attn, static_cast<uint64_t>(pos_), k.geom);
        k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    for (int l = 0; l < nl_; ++l) {
        int nrun = 0;
        for (const Step& s : types_[l]->program) {
            Kern& k = kerns_.at(s.kernel);
            if (s.op == "run") {
                double ms = run(k, s.args, l);
                (nrun++ == 0 ? timing_.part0_ms : timing_.part1_ms) += ms;
            } else {
                route(k, l, s.act_off);
            }
        }
    }
    if (want_logits) {
        auto t1 = std::chrono::steady_clock::now();
        for (const Step& s : man_.tail) run(kerns_.at(s.kernel), s.args, 0);
        xrt::bo& lg = buffer("logits", 0);
        lg.sync(XCL_BO_SYNC_BO_FROM_DEVICE, man_.vocab * 4, 0);
        std::memcpy(logits_host_.data(), lg.map<uint8_t*>(), man_.vocab * 4);
        timing_.lmhead_ms = ms_since(t1);
    }
    ++pos_;
    if (!mpos && mrope_on_) ++mrope_pos_;      // a text token after an image: (c, c, c), then c + 1
    timing_.total_ms = ms_since(t0);
}

// ============================================================================
// 0167/#32: the GEMM prefill route. See manifest.hpp's GemmBlockProgram
// docstring for the shape of the chain; core.hpp's field comments explain
// each buffer's lifetime and why it is per-layer vs global.
// ============================================================================

std::pair<size_t, size_t> Core::op_region(const LayerType& lt, const std::string& from, size_t idx) const {
    const auto& ops = from == "pool" ? lt.pool : lt.consts;
    if (idx >= ops.size())
        throw std::runtime_error("open_qwen36: op_region: " + from + " op " + std::to_string(idx) + " out of range (" +
                                 std::to_string(ops.size()) + " ops)");
    const PackOp& op = ops[idx];
    // the GEMM dequantises the q4_1 band law, which is what std_perm writes; a q8_perm
    // projection has no route (the recipe does not emit one) and is refused here too
    if (op.op != "std_perm")
        throw std::runtime_error("open_qwen36: op_region: " + from + " op " + std::to_string(idx) + " is a " + op.op +
                                 ", not a band-law projection the GEMM reads");
    return {static_cast<size_t>(op.dst), static_cast<size_t>(op.nch) * man_.chunk_bytes};
}

std::string Core::const_tensor(const LayerType& lt, const std::string& suffix, int layer) const {
    for (const PackOp& op : lt.consts) {
        const std::string& t = op.tensor;
        if (t.size() >= suffix.size() && t.compare(t.size() - suffix.size(), suffix.size(), suffix) == 0) {
            std::string name = t;
            const size_t at = name.find("{l}");
            if (at != std::string::npos) name.replace(at, 3, std::to_string(layer));
            return name;
        }
    }
    throw std::runtime_error("open_qwen36: layer type " + lt.name + " has no consts tensor ending in " + suffix);
}

std::vector<float> Core::gemm(const Step& s, const std::vector<float>& x, size_t T, size_t K, size_t N, int layer) {
    xrt::bo& xb = buffer(s.args[1], 0);
    xrt::bo& yb = buffer(s.args[2], 0);
    if (xb.size() < K * T * 2 || yb.size() < N * T * 4)
        throw std::runtime_error("open_qwen36: gemm " + s.kernel + ": the x / y globals are smaller than [" +
                                 std::to_string(K) + "] x " + std::to_string(T) + " -> [" + std::to_string(N) + "]");
    auto t0 = std::chrono::steady_clock::now();
    host::tile_x(x.data(), T, K, xb.map<uint16_t*>());   // straight into the mapped buffer
    timing_.part1_ms += ms_since(t0);
    xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, K * T * 2, 0);
    timing_.part0_ms += run(kerns_.at(s.kernel), s.args, layer);
    yb.sync(XCL_BO_SYNC_BO_FROM_DEVICE, N * T * 4, 0);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> out(T * N);
    host::transpose(yb.map<float*>(), N, T, out.data());   // [N, T] on the device -> [T, N]
    timing_.part1_ms += ms_since(t1);
    return out;
}

void Core::tail_logits(const float* row) {
    auto t1 = std::chrono::steady_clock::now();
    xrt::bo& xres1 = buffer("xres", 0);
    std::memcpy(xres1.map<uint8_t*>(), row, man_.hidden * 4);
    xres1.sync(XCL_BO_SYNC_BO_TO_DEVICE, man_.hidden * 4, 0);
    for (const Step& s : man_.tail) run(kerns_.at(s.kernel), s.args, 0);
    xrt::bo& lg = buffer("logits", 0);
    lg.sync(XCL_BO_SYNC_BO_FROM_DEVICE, man_.vocab * 4, 0);
    std::memcpy(logits_host_.data(), lg.map<uint8_t*>(), man_.vocab * 4);
    timing_.lmhead_ms = ms_since(t1);
}

void Core::shuttle_buf(xrt::bo& wide, xrt::bo& scratch1, size_t token, size_t act_bytes, bool wide_to_scratch) {
    // `wide` is an explicit argument rather than a fixed per-layer buffer
    // because the GEMM route's T-wide attention scratch ("gact") is a
    // GLOBAL, not per-layer (act_bytes is uniform across every Granite dense
    // layer, so a per-layer copy would only cost memory).
    const size_t off = token * act_bytes;
    if (wide_to_scratch) {
        wide.sync(XCL_BO_SYNC_BO_FROM_DEVICE, act_bytes, off);
        std::memcpy(scratch1.map<uint8_t*>(), wide.map<uint8_t*>() + off, act_bytes);
        scratch1.sync(XCL_BO_SYNC_BO_TO_DEVICE, act_bytes, 0);
    } else {
        scratch1.sync(XCL_BO_SYNC_BO_FROM_DEVICE, act_bytes, 0);
        std::memcpy(wide.map<uint8_t*>() + off, scratch1.map<uint8_t*>(), act_bytes);
        wide.sync(XCL_BO_SYNC_BO_TO_DEVICE, act_bytes, off);
    }
}

void Core::rmsnorm_host(const std::vector<double>& x, size_t T, size_t hid, const std::vector<uint16_t>& w_bf16,
                        double eps, std::vector<float>& out) {
    // Reduction AND the final multiply both in fp64 (trap 11: a fp32
    // reduction over 2560+ terms is not a safe correctness metric at this
    // width). out[t,k] = x[t,k]/sqrt(mean_k(x^2)+eps)*w[k].
    out.assign(T * hid, 0.f);
    for (size_t t = 0; t < T; ++t) {
        const double* row = &x[t * hid];
        double ss = 0;
        for (size_t k = 0; k < hid; ++k) ss += row[k] * row[k];
        const double rms = std::sqrt(ss / static_cast<double>(hid) + eps);
        float* orow = &out[t * hid];
        for (size_t k = 0; k < hid; ++k) {
            const double w = static_cast<double>(bf16_to_f32(w_bf16[k]));
            orow[k] = static_cast<float>((row[k] / rms) * w);
        }
    }
}

void Core::tile_gemm_x(const std::vector<float>& x_tk, size_t T, size_t K, std::vector<uint16_t>& out) {
    out.assign(K * T, 0);
    host::tile_x(x_tk.data(), T, K, out.data());
}

void Core::tile_gemm_x_reference(const std::vector<float>& x_tk, size_t T, size_t K, std::vector<uint16_t>& out) {
    // [T,K] fp32 -> bf16, pre-tiled [K,T] "k,n" order (K_TILE=64, MAC 8x8,
    // tile_n=32 -- gemm_q4_prefill.py's own GQP_TILE_N default), matching
    // open_npue/npue_pack.cpp's tile_b algorithm exactly (copied, not
    // called -- that function has internal linkage in its own translation
    // unit). `x_tk` is [T,K] row-major (T rows of K elements, this route's
    // own natural RMSNorm-output layout); the transpose to logical [K,T] is
    // done by indexing, not a separate pass.
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (K % TK || T % TN)
        throw std::runtime_error("open_qwen36: gemm-route tile_gemm_x: K=" + std::to_string(K) + " or T=" +
                                 std::to_string(T) + " does not tile by (" + std::to_string(TK) + "," + std::to_string(TN) + ")");
    out.assign(K * T, 0);
    size_t w = 0;
    for (size_t kb = 0; kb < K / TK; ++kb)
        for (size_t nb = 0; nb < T / TN; ++nb)
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s)
                        for (size_t t = 0; t < MAC; ++t) {
                            const size_t r = kb * TK + si * MAC + s;  // K index
                            const size_t c = nb * TN + ti * MAC + t;  // T index
                            out[w++] = f32_to_bf16(x_tk[c * K + r]);
                        }
}

void Core::step_gemm_block_layer(int l, std::vector<double>& xres, size_t T) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const size_t hid = man_.hidden, qw = gb.qw, kvw = gb.kvw, ff = gb.ff;
    const size_t n_qkv3 = qw + 2 * kvw;

    // One GEMM dispatch: tile `x` [T,K] -> upload -> run -> download `y` [N,T].
    // Timing: part0_ms sums ALL 5 GEMM dispatches
    // (qkv3, o, gate, up, down); route_ms (otherwise unused by a dense/Granite
    // layer type -- no MoE routing here) is repurposed for the T attention
    // (dxB) dispatches below, so the two are cleanly separable instead of
    // both landing in part1_ms.
    auto run_gemm = [&](size_t idx, const std::vector<float>& x, size_t K, size_t N, std::vector<float>& y_out) {
        const Step& s = gb.program[idx];
        std::vector<uint16_t> xt;
        tile_gemm_x(x, T, K, xt);
        xrt::bo& xb = buffer(s.args[1], 0);
        std::memcpy(xb.map<uint8_t*>(), xt.data(), xt.size() * 2);
        xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, xt.size() * 2, 0);
        Kern& k = kerns_.at(s.kernel);
        timing_.part0_ms += run(k, s.args, l);
        xrt::bo& yb = buffer(s.args[2], 0);
        yb.sync(XCL_BO_SYNC_BO_FROM_DEVICE, N * T * 4, 0);
        y_out.assign(N * T, 0.f);
        std::memcpy(y_out.data(), yb.map<uint8_t*>(), N * T * 4);
    };

    // ---- entry RMSNorm, GEMM A' (qkv3, real q|k|v pool weight, ONE dispatch) ----
    std::vector<float> xnorm;
    rmsnorm_host(xres, T, hid, ln_w_bf16_[l], gb.eps, xnorm);
    std::vector<float> y_qkv3;  // [n_qkv3, T] row-major f32
    run_gemm(0, xnorm, hid, n_qkv3, y_qkv3);

    // ---- T single-token dxB dispatches, position-patched, through a GLOBAL
    // T-wide "gact" scratch buffer, shuttled one token at a time via
    // shuttle_buf() -- proven on hardware before this was wired in. ----
    xrt::bo& gact = buffer("gact", 0);
    xrt::bo& act1 = buffer("act", l);
    const size_t AD = lt.act_bytes;
    {
        // Fill gact's Q/K/V region for every token from y_qkv3's columns
        // (f32 bytes, matching the T=1 "act" buffer's own AD_Q/AD_KVN format
        // -- the SAME format Core::step() writes there today).
        std::vector<uint8_t> host_gact(T * AD, 0);
        for (size_t tk = 0; tk < T; ++tk) {
            uint8_t* base = host_gact.data() + tk * AD;
            float* qd = reinterpret_cast<float*>(base + gb.ad_q);
            float* kd = reinterpret_cast<float*>(base + gb.ad_kvn);
            float* vd = reinterpret_cast<float*>(base + gb.ad_kvn + kvw * 4);
            for (size_t c = 0; c < qw; ++c) qd[c] = y_qkv3[c * T + tk];
            for (size_t c = 0; c < kvw; ++c) kd[c] = y_qkv3[(qw + c) * T + tk];
            for (size_t c = 0; c < kvw; ++c) vd[c] = y_qkv3[(qw + kvw + c) * T + tk];
        }
        std::memcpy(gact.map<uint8_t*>(), host_gact.data(), T * AD);
        gact.sync(XCL_BO_SYNC_BO_TO_DEVICE, T * AD, 0);
    }
    {
        Kern& dxb = kerns_.at("dxB");
        const std::vector<std::string> attn_args = {"pool", "xres", "consts", "state", "act", "ptab"};
        for (size_t tk = 0; tk < T; ++tk) {
            const uint64_t pos = static_cast<uint64_t>(pos_) + tk;
            stream_patch::attn_apply(dxb.iw(), dxb.attn, pos, dxb.geom);
            dxb.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            shuttle_buf(gact, act1, tk, AD, /*wide_to_scratch=*/true);
            timing_.route_ms += run(dxb, attn_args, l);
            shuttle_buf(gact, act1, tk, AD, /*wide_to_scratch=*/false);
        }
    }
    // ---- read back AD_OG (bf16, qw elements/token) as [T,qw] f32 -----------
    std::vector<float> og(T * qw, 0.f);
    {
        gact.sync(XCL_BO_SYNC_BO_FROM_DEVICE, T * AD, 0);
        const uint8_t* base = gact.map<uint8_t*>();
        for (size_t tk = 0; tk < T; ++tk) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(base + tk * AD + gb.ad_og);
            for (size_t c = 0; c < qw; ++c) og[tk * qw + c] = bf16_to_f32(src[c]);
        }
    }

    // ---- GEMM O (o_proj, real weight, reusing the "qkv"-shaped context) ---
    std::vector<float> y_o;  // [hid, T]
    run_gemm(1, og, qw, hid, y_o);

    // ---- host: residual add, post-attention RMSNorm ------------------------
    std::vector<double> res1(T * hid);
    for (size_t tk = 0; tk < T; ++tk)
        for (size_t c = 0; c < hid; ++c) res1[tk * hid + c] = xres[tk * hid + c] + static_cast<double>(y_o[c * T + tk]);
    std::vector<float> xm;
    rmsnorm_host(res1, T, hid, post_ln_w_bf16_[l], gb.eps, xm);

    // ---- GEMM gate_proj + up_proj (SAME context, zero switch between them) -
    std::vector<float> y_gate, y_up;  // both [ff, T]
    run_gemm(2, xm, hid, ff, y_gate);
    run_gemm(3, xm, hid, ff, y_up);

    // ---- host SwiGLU: silu(gate) * up ---------------------------------------
    std::vector<float> h(T * ff);
    for (size_t tk = 0; tk < T; ++tk)
        for (size_t c = 0; c < ff; ++c) {
            const double g = static_cast<double>(y_gate[c * T + tk]);
            const double u = static_cast<double>(y_up[c * T + tk]);
            h[tk * ff + c] = static_cast<float>((g / (1.0 + std::exp(-g))) * u);
        }

    // ---- GEMM down_proj, then residual -> next layer's xres -----------------
    std::vector<float> y_down;  // [hid, T]
    run_gemm(4, h, ff, hid, y_down);
    for (size_t tk = 0; tk < T; ++tk)
        for (size_t c = 0; c < hid; ++c) xres[tk * hid + c] = res1[tk * hid + c] + static_cast<double>(y_down[c * T + tk]);
}

void Core::step_gemm_block(const std::vector<int>& ids, size_t t_real, bool want_logits) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: step_gemm_block before load_weights");
    const size_t T = ids.size();
    if (T == 0) return;
    if (gemm_block_t_ == 0 || T != gemm_block_t_)
        throw std::runtime_error("open_qwen36: step_gemm_block called with " + std::to_string(T) +
                                 " tokens, but this kernel set's gemm-route block size is " +
                                 std::to_string(gemm_block_t_) + " (0 = no gemm_block program loaded)");
    if (t_real == 0 || t_real > T) throw std::runtime_error("open_qwen36: step_gemm_block: t_real must be in (0, T]");
    // Positions [pos_, pos_+T) are all touched (padding columns included --
    // hardware-proven exact: the real columns' output does not depend on
    // what the padding columns carry), so the CAPACITY check must cover T,
    // not just the real tokens -- even though only t_real of them advance
    // pos_ afterward.
    if (static_cast<size_t>(pos_) + T > cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: gemm-block [" + std::to_string(pos_) + ", " +
                                 std::to_string(pos_ + T) + ") would exceed the context capacity " +
                                 std::to_string(cfg_.max_ctx));
    for (int tok : ids)
        if (tok < 0 || static_cast<size_t>(tok) >= man_.vocab) throw std::runtime_error("open_qwen36: token id out of range");
    if (types_[0]->gemm_block.kind != "dense") {
        step_block_moe(ids, t_real, want_logits);
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    // ---- embed all T tokens (padding included) into a HOST-resident fp64
    // xres[T,hidden] -- this route's running residual stream lives on the
    // HOST between GEMM dispatches (RMSNorm/residual/SwiGLU are host-side,
    // not fused on-core), so there is no device-resident T-wide buffer for
    // it, unlike the per-layer weight/activation buffers below. ------------
    std::vector<double> xres(T * man_.hidden);
    {
        std::vector<float> row(man_.hidden);
        for (size_t tk = 0; tk < T; ++tk) {
            file_->bf16_row(man_.embed_tensor, static_cast<size_t>(ids[tk]), man_.hidden, row.data());
            for (size_t c = 0; c < man_.hidden; ++c) xres[tk * man_.hidden + c] = static_cast<double>(row[c]);
        }
    }

    for (int l = 0; l < nl_; ++l) {
        if (types_[l]->gemm_block.t == 0)
            throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + " (" + types_[l]->name + ") has no gemm_block program");
        step_gemm_block_layer(l, xres, T);
    }
    pos_ += static_cast<int>(t_real);

    block_logits_.clear();
    if (block_logits_all_) {
        std::vector<float> row(man_.hidden);
        for (size_t t = 0; t < t_real; ++t) {
            for (size_t c = 0; c < man_.hidden; ++c) row[c] = static_cast<float>(xres[t * man_.hidden + c]);
            tail_logits(row.data());
            block_logits_.push_back(logits_host_);
        }
    }
    if (want_logits) {
        // only the last REAL token's logits, as step() does for a prefill
        std::vector<float> last_row(man_.hidden);
        for (size_t c = 0; c < man_.hidden; ++c) last_row[c] = static_cast<float>(xres[(t_real - 1) * man_.hidden + c]);
        tail_logits(last_row.data());
    }
    timing_.total_ms = ms_since(t0);
}

// ---- the MoE families' block: kinds linear and full. Timing: part0 = the GEMM
// dispatches, part1 = the host stages, route = the per-token MoE (routing + kernel).

void Core::step_block_moe(const std::vector<int>& ids, size_t t_real, bool want_logits) {
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};
    const size_t T = ids.size(), hid = man_.hidden;
    std::vector<float> xres(T * hid);
    for (size_t t = 0; t < T; ++t) file_->bf16_row(man_.embed_tensor, static_cast<size_t>(ids[t]), hid, xres.data() + t * hid);
    for (int l = 0; l < nl_; ++l) {
        const std::string& kind = types_[l]->gemm_block.kind;
        if (kind == "linear") block_layer_linear(l, xres, T, t_real);
        else if (kind == "full") block_layer_full(l, xres, T, t_real);
        else throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + " (" + types_[l]->name + ") has no block route");
    }
    pos_ += static_cast<int>(t_real);
    block_logits_.clear();
    if (block_logits_all_)
        for (size_t t = 0; t < t_real; ++t) {
            tail_logits(xres.data() + t * hid);
            block_logits_.push_back(logits_host_);
        }
    if (want_logits) tail_logits(xres.data() + (t_real - 1) * hid);
    timing_.total_ms = ms_since(t0);
}

void Core::moe_token(int l, const float* xm, const float* res, const float* probs, const int32_t* idx, const float* w,
                     float* out) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const size_t hid = man_.hidden, E = man_.moe.experts, topk = man_.moe.topk;
    xrt::bo& act = act_[l];
    uint8_t* a = act.map<uint8_t*>();
    auto tp = std::chrono::steady_clock::now();
    // what the sequential layer's first dispatch would have left in act: xm (bf16), the
    // router record [probs f32[E] | idx i32[8] @rout_idx_off | w f32[8]], the residual (f32)
    uint16_t* xmb = reinterpret_cast<uint16_t*>(a + gb.a_xm);
    for (size_t i = 0; i < hid; ++i) xmb[i] = f32_to_bf16(xm[i]);
    std::memcpy(a + gb.a_rout, probs, E * 4);
    int32_t* ri = reinterpret_cast<int32_t*>(a + gb.a_rout + man_.rout_idx_off);
    float* rw = reinterpret_cast<float*>(a + gb.a_rout + man_.rout_idx_off + 8 * 4);
    for (size_t s = 0; s < 8; ++s) {
        ri[s] = s < topk ? idx[s] : 0;
        rw[s] = s < topk ? w[s] : 0.f;
    }
    std::memcpy(a + gb.a_res, res, hid * 4);
    act.sync(XCL_BO_SYNC_BO_TO_DEVICE, hid * 2, gb.a_xm);
    act.sync(XCL_BO_SYNC_BO_TO_DEVICE, E * 4 + 16 * 4, gb.a_rout);
    act.sync(XCL_BO_SYNC_BO_TO_DEVICE, hid * 4, gb.a_res);
    timing_.moe_prep_ms += ms_since(tp);
    // the MoE-only dispatch (mx.py): the routed slots patched from the ids we just wrote
    // (no readback of the record), then run
    auto t0 = std::chrono::steady_clock::now();
    Kern& mk = kerns_.at(gb.moe_kernel);
    if (mk.moe2.empty()) throw std::runtime_error("open_qwen36: " + gb.moe_kernel + " has no routed-expert table");
    uint32_t slots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t s = 0; s < topk; ++s) {
        if (idx[s] < 0 || static_cast<unsigned>(idx[s]) >= E) throw std::runtime_error("open_qwen36: router produced expert index " + std::to_string(idx[s]));
        slots[s] = static_cast<uint32_t>(idx[s]);
    }
    stream_patch::moe2_apply(mk.iw(), mk.moe2, slots, man_.moe);
    mk.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    timing_.moe_patch_ms += ms_since(t0);
    timing_.moe_run_ms += run(mk, gb.moe_args, l);
    auto t1 = std::chrono::steady_clock::now();
    xrt::bo& xr = buffer("xres", 0);
    xr.sync(XCL_BO_SYNC_BO_FROM_DEVICE, hid * 4, 0);
    std::memcpy(out, xr.map<uint8_t*>(), hid * 4);
    timing_.moe_read_ms += ms_since(t1);
    timing_.route_ms += ms_since(tp);
}

namespace {
// residual + post-attention norm, the router, then the MoE per real token: the tail both
// MoE kinds share once their attention half has produced `out` [T, hid]
struct MoeTail {
    std::vector<float> res, xm, probs, w;
    std::vector<int32_t> idx;
};
}  // namespace

// The shared expert, once over the block instead of once per token: up|gate (one GEMM --
// the two are contiguous in the pool and both band-law) then down, with silu, the sigmoid
// gate and the add on the host. The kernels' own formula, from moe_silu32 / moe_hdr2 /
// moe_accfin: h = silu(g) * u, out += sigmoid(xm . sgw) * down(h). xm is rounded to bf16
// first because that is what the dispatch would have seen.
void Core::shared_expert_block(int l, const float* xm, float* res, size_t T, size_t t_real) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const size_t hid = man_.hidden, ff = gb.shared_ff;
    std::vector<float> xv(xm, xm + T * hid);
    const std::vector<float> ug = gemm(gb.shared_program[0], xv, T, hid, 2 * ff, l);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> h(T * ff);
    for (size_t t = 0; t < T; ++t) {
        const float* u = ug.data() + t * 2 * ff;
        const float* g = u + ff;
        float* ho = h.data() + t * ff;
        for (size_t j = 0; j < ff; ++j) ho[j] = g[j] / (1.f + std::exp(-g[j])) * u[j];
    }
    timing_.shared_ms += ms_since(t0);
    const std::vector<float> y = gemm(gb.shared_program[1], h, T, ff, hid, l);
    auto t1 = std::chrono::steady_clock::now();
    const std::vector<float>& sgw = hc_[l].sgw;
    for (size_t t = 0; t < t_real; ++t) {
        const float* x = xm + t * hid;
        double d = 0;
        for (size_t i = 0; i < hid; ++i) d += static_cast<double>(bf16_to_f32(f32_to_bf16(x[i]))) * sgw[i];
        const float gate = 1.f / (1.f + std::exp(-static_cast<float>(d)));
        const float* yr = y.data() + t * hid;
        float* r = res + t * hid;
        for (size_t i = 0; i < hid; ++i) r[i] += gate * yr[i];
    }
    timing_.shared_ms += ms_since(t1);
}

void Core::block_layer_linear(int l, std::vector<float>& xres, size_t T, size_t t_real) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const HostConsts& hc = hc_[l];
    const size_t hid = man_.hidden, nch = gb.qkv_dim, vw = gb.vw, E = man_.moe.experts, topk = man_.moe.topk;

    std::vector<float> xn(T * hid);
    host::rmsnorm_rows(xres.data(), T, hid, hc.ln.data(), gb.eps, xn.data());
    const std::vector<float> y = gemm(gb.program[0], xn, T, hid, nch + vw, l);
    std::vector<float> qkv(T * nch), z(T * vw);
    for (size_t t = 0; t < T; ++t) {
        std::memcpy(qkv.data() + t * nch, y.data() + t * (nch + vw), nch * 4);
        std::memcpy(z.data() + t * vw, y.data() + t * (nch + vw) + nch, vw * 4);
    }
    // the conv rows and S live in the state BO; the recurrence runs on the host in place
    xrt::bo& st = state_[l];
    auto ts = std::chrono::steady_clock::now();
    st.sync(XCL_BO_SYNC_BO_FROM_DEVICE, lt.state_bytes, 0);
    timing_.state_ms += ms_since(ts);
    uint8_t* sp = st.map<uint8_t*>();
    host::DeltaGeom g;
    g.T = T; g.t_real = t_real; g.hid = hid;
    g.key_heads = gb.key_heads; g.value_heads = gb.value_heads; g.head_dim = gb.head_dim; g.taps = gb.conv_kernel;
    g.lanes = hc.lanes; g.s_rows = gb.s_rows; g.eps = gb.eps;
    std::vector<float> og(T * vw);
    auto t0 = std::chrono::steady_clock::now();
    host::deltanet_block(g, qkv.data(), z.data(), xn.data(), hc.convw.data(), hc.Wa.data(), hc.Wb.data(), hc.A.data(),
                         hc.dtb.data(), hc.nw.data(), reinterpret_cast<uint16_t*>(sp),
                         reinterpret_cast<float*>(sp + gb.state_s_off), og.data());
    timing_.part1_ms += ms_since(t0);
    timing_.mid_ms += ms_since(t0);
    ts = std::chrono::steady_clock::now();
    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
    timing_.state_ms += ms_since(ts);
    const std::vector<float> out = gemm(gb.program[1], og, T, vw, hid, l);

    auto t1 = std::chrono::steady_clock::now();
    MoeTail m;
    m.res.resize(T * hid);
    for (size_t i = 0; i < T * hid; ++i) m.res[i] = xres[i] + out[i];
    m.xm.resize(T * hid);
    host::rmsnorm_rows(m.res.data(), T, hid, hc.postln.data(), gb.eps, m.xm.data());
    m.probs.resize(T * E); m.idx.resize(T * topk); m.w.resize(T * topk);
    host::router_block(t_real, hid, E, topk, m.xm.data(), hc.router.data(), m.probs.data(), m.idx.data(), m.w.data());
    timing_.part1_ms += ms_since(t1);
    timing_.tail_ms += ms_since(t1);
    shared_expert_block(l, m.xm.data(), m.res.data(), T, t_real);
    for (size_t t = 0; t < T; ++t) {
        if (t < t_real)
            moe_token(l, m.xm.data() + t * hid, m.res.data() + t * hid, m.probs.data() + t * E, m.idx.data() + t * topk,
                      m.w.data() + t * topk, xres.data() + t * hid);
        else
            std::memcpy(xres.data() + t * hid, m.res.data() + t * hid, hid * 4);   // padding: carried, never read
    }
}

void Core::block_layer_full(int l, std::vector<float>& xres, size_t T, size_t t_real) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const HostConsts& hc = hc_[l];
    const size_t hid = man_.hidden, qw = gb.qw, kvw = gb.kvw, nf = 2 * qw + 2 * kvw;
    const size_t E = man_.moe.experts, topk = man_.moe.topk;

    std::vector<float> xn(T * hid);
    host::rmsnorm_rows(xres.data(), T, hid, hc.ln.data(), gb.eps, xn.data());
    const std::vector<float> y = gemm(gb.program[0], xn, T, hid, nf, l);
    std::vector<float> q(T * qw), k(T * kvw), v(T * kvw), gate(T * qw);
    for (size_t t = 0; t < T; ++t) {
        const float* row = y.data() + t * nf;
        std::memcpy(q.data() + t * qw, row, qw * 4);
        std::memcpy(k.data() + t * kvw, row + qw, kvw * 4);
        std::memcpy(v.data() + t * kvw, row + qw + kvw, kvw * 4);
        std::memcpy(gate.data() + t * qw, row + qw + 2 * kvw, qw * 4);
    }
    // the KV rows: [0, pos_) read, [pos_, pos_ + t_real) written by the host attention
    xrt::bo& st = state_[l];
    const size_t row = lt.state_row;
    auto ts = std::chrono::steady_clock::now();
    if (pos_ > 0) st.sync(XCL_BO_SYNC_BO_FROM_DEVICE, static_cast<size_t>(pos_) * row, 0);
    timing_.state_ms += ms_since(ts);
    host::AttnGeom g;
    g.T = T; g.t_real = t_real; g.nh = gb.nh; g.kvh = gb.kvh; g.hd = gb.hd; g.rot = gb.rot;
    g.pos0 = static_cast<size_t>(pos_); g.eps = gb.eps;
    std::vector<float> og(T * qw);
    auto t0 = std::chrono::steady_clock::now();
    host::attention_block(g, q.data(), k.data(), v.data(), gate.data(), hc.qn.data(), hc.kn.data(), man_.rope_inv_freq.data(),
                          st.map<uint16_t*>(), row / 2, og.data());
    timing_.part1_ms += ms_since(t0);
    timing_.mid_ms += ms_since(t0);
    ts = std::chrono::steady_clock::now();
    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, t_real * row, static_cast<size_t>(pos_) * row);
    timing_.state_ms += ms_since(ts);
    const std::vector<float> out = gemm(gb.program[1], og, T, qw, hid, l);

    auto t1 = std::chrono::steady_clock::now();
    MoeTail m;
    m.res.resize(T * hid);
    for (size_t i = 0; i < T * hid; ++i) m.res[i] = xres[i] + out[i];
    m.xm.resize(T * hid);
    host::rmsnorm_rows(m.res.data(), T, hid, hc.postln.data(), gb.eps, m.xm.data());
    m.probs.resize(T * E); m.idx.resize(T * topk); m.w.resize(T * topk);
    host::router_block(t_real, hid, E, topk, m.xm.data(), hc.router.data(), m.probs.data(), m.idx.data(), m.w.data());
    timing_.part1_ms += ms_since(t1);
    timing_.tail_ms += ms_since(t1);
    shared_expert_block(l, m.xm.data(), m.res.data(), T, t_real);
    for (size_t t = 0; t < T; ++t) {
        if (t < t_real)
            moe_token(l, m.xm.data() + t * hid, m.res.data() + t * hid, m.probs.data() + t * E, m.idx.data() + t * topk,
                      m.w.data() + t * topk, xres.data() + t * hid);
        else
            std::memcpy(xres.data() + t * hid, m.res.data() + t * hid, hid * 4);
    }
}

void Core::seek(int pos) {
    if (pos < 0 || static_cast<size_t>(pos) >= cfg_.max_ctx) throw std::runtime_error("open_qwen36: seek out of range");
    pos_ = pos;
}

Snapshot Core::checkpoint() const {
    Snapshot s;
    s.pos = pos_;
    s.mrope_pos = mrope_pos_;
    s.mrope_on = mrope_on_;
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        xrt::bo& bo = const_cast<xrt::bo&>(state_[l]);
        if (lt.state_kind == "kv") {
            size_t n = static_cast<size_t>(pos_) * lt.state_row;
            std::vector<uint8_t> rows(n);
            if (n) {
                bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, n, 0);
                std::memcpy(rows.data(), bo.map<uint8_t*>(), n);
            }
            s.kv.push_back(std::move(rows));
        } else {
            std::vector<uint8_t> st(lt.state_bytes);
            bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, lt.state_bytes, 0);
            std::memcpy(st.data(), bo.map<uint8_t*>(), lt.state_bytes);
            s.states.push_back(std::move(st));
        }
    }
    return s;
}

void Core::restore(const Snapshot& s) {
    size_t il = 0, ia = 0;
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        if (lt.state_kind == "kv") {
            const auto& rows = s.kv.at(ia++);
            if (!rows.empty()) {
                std::memcpy(state_[l].map<uint8_t*>(), rows.data(), rows.size());
                state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, rows.size(), 0);
            }
        } else {
            const auto& st = s.states.at(il++);
            if (st.size() != lt.state_bytes) throw std::runtime_error("open_qwen36: snapshot state size mismatch");
            std::memcpy(state_[l].map<uint8_t*>(), st.data(), lt.state_bytes);
            state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
        }
    }
    pos_ = s.pos;
    mrope_pos_ = s.mrope_pos;
    mrope_on_ = s.mrope_on;
}

void Core::kv_row(int layer, int row, bool value, uint16_t* out) {
    if (layer < 0 || layer >= nl_ || !is_attention_layer(layer)) throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " has no KV cache");
    if (row < 0 || static_cast<size_t>(row) >= cfg_.max_ctx) throw std::runtime_error("open_qwen36: KV row out of range");
    const size_t kv_row = types_[layer]->state_row;
    size_t off = static_cast<size_t>(row) * kv_row + (value ? kv_row / 2 : 0);
    state_[layer].sync(XCL_BO_SYNC_BO_FROM_DEVICE, kv_row / 2, off);
    std::memcpy(out, state_[layer].map<uint8_t*>() + off, kv_row / 2);
}

}  // namespace open_qwen36
