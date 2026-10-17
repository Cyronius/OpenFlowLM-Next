// Traces: OPEN-VISION-GEMMA3 -- the image block's dispatch schedule and the attnpos words it patches.
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "open_qwen36/bidir_block.hpp"
#include "stream_patch.hpp"

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

// attn_apply as it was before attn_apply_rows existed (feat/gemma3-1b a412c9e7), the oracle for the refactor.
void attn_apply_before(uint32_t* iw, const std::vector<stream_patch::AttnPatch>& table, uint64_t pos,
                       const stream_patch::AttnGeometry& g) {
    uint64_t start, nf;
    stream_patch::attn_window(pos, g.window, &start, &nf);
    if (g.rb > 1) {
        const uint64_t valid = pos - start;
        nf = g.rb * (valid / g.rb + 1) - 1;
    }
    for (const auto& p : table) {
        uint64_t v = p.kind == 0 ? nf * g.kv_row / 4 : p.kind == 1 ? pos * g.kv_row : p.kind == 2 ? pos * g.ptab_row : start * g.kv_row;
        iw[p.word] = static_cast<uint32_t>(v) | p.flags;
    }
}

const std::vector<stream_patch::AttnPatch> kTable = {{0, 0, 0}, {1, 1, 0}, {2, 2, 0x80000000u}, {3, 3, 0}};

void patch_tests() {
    bool same = true;
    for (uint64_t window : {0ull, 512ull, 1024ull})
        for (uint64_t rb : {0ull, 1ull, 2ull, 4ull})
            for (uint64_t pos = 0; pos < 3000; ++pos) {
                stream_patch::AttnGeometry g;
                g.kv_row = 4096;
                g.ptab_row = 2048;
                g.window = window;
                g.rb = rb;
                uint32_t a[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};
                attn_apply_before(a, kTable, pos, g);
                stream_patch::attn_apply(b, kTable, pos, g);
                for (int i = 0; i < 4; ++i) same = same && a[i] == b[i];
            }
    check(same, "attn_apply writes the same four words as before, every position below 3000, windows 0/512/1024, rb 0/1/2/4");

    stream_patch::AttnGeometry g;
    g.kv_row = 4096;
    g.ptab_row = 2048;
    g.rb = 2;
    uint32_t w[4] = {0, 0, 0, 0};
    stream_patch::attn_apply_rows(w, kTable, 7, 300, 520, 266, g);
    check(w[0] == (2 * (300 / 2 + 1) - 1) * 4096 / 4 && w[1] == 520 * 4096 && w[2] == (266 * 2048 | 0x80000000u) &&
              w[3] == 7 * 4096,
          "attn_apply_rows: fill length, drain row, record row (flag kept) and fill start are each the one asked for");
    g.rb = 1;
    stream_patch::attn_apply_rows(w, kTable, 0, 0, 3, 3, g);
    check(w[0] == 4096 / 4, "attn_apply_rows: no real rows still streams the one row the kernel masks");
}

/// Replays a layer's schedule on a model of the block's cache rows; returns false on the first fill that is wrong.
bool replay(uint64_t p0, uint32_t n, uint64_t window, bool last, std::string* why) {
    const std::vector<open_qwen36::bidir::Step> s = open_qwen36::bidir::layer_schedule(p0, n, last, true);
    if (s.size() != (last ? n : 2 * n - 1)) { *why = "dispatch count"; return false; }
    std::vector<int> row(p0 + n + 4, -1);          // which token's K/V each cache row holds; -1 = stale
    for (const auto& d : s) {
        if (d.copy_to >= 0) row[d.copy_to] = row[d.copy_from];
        uint64_t start, valid;
        open_qwen36::bidir::rows(p0, n, d.t, window, &start, &valid);
        const uint64_t pos = p0 + d.t;
        const uint64_t want_start = (window && pos + 1 > window) ? pos + 1 - window : 0;
        if (start != want_start || start + valid != p0 + n - 1) { *why = "window of token " + std::to_string(d.t); return false; }
        if (d.keep) {
            std::multiset<int> seen;
            for (uint64_t r = p0; r < start + valid; ++r) seen.insert(row[r]);
            std::multiset<int> want;
            for (uint32_t u = 0; u < n; ++u)
                if (u != d.t) want.insert(static_cast<int>(u));
            if (seen != want) { *why = "token " + std::to_string(d.t) + " does not read every other image row once"; return false; }
        }
        row[d.drain] = static_cast<int>(d.t);
    }
    for (uint32_t u = 0; u < n; ++u)
        if (row[p0 + u] != static_cast<int>(u)) { *why = "rows not back in natural order"; return false; }
    return true;
}

void schedule_tests() {
    bool ok = true;
    std::string why;
    for (uint64_t p0 : {0ull, 5ull, 900ull, 1100ull})
        for (uint32_t n : {1u, 2u, 3u, 7u, 256u})
            for (uint64_t window : {0ull, 1024ull})
                for (bool last : {false, true})
                    if (ok && !replay(p0, n, window, last, &why)) {
                        ok = false;
                        why += " (p0 " + std::to_string(p0) + ", n " + std::to_string(n) + ", window " +
                               std::to_string(window) + (last ? ", last layer)" : ")");
                    }
    check(ok, "every kept dispatch reads each other image row exactly once and the rows end in natural order" +
                  (ok ? std::string() : ": " + why));

    const auto a = open_qwen36::bidir::layer_schedule(10, 256, false);
    size_t copies = 0, kept = 0;
    for (const auto& d : a) {
        copies += d.copy_to >= 0;
        kept += d.keep;
    }
    check(a.size() == 511 && copies == 255 && kept == 256, "a 256-token layer: 255 K/V dispatches, 256 output dispatches, 255 row copies");
    const auto l = open_qwen36::bidir::layer_schedule(10, 256, true, false);
    bool none = true;
    for (const auto& d : l) none = none && !d.keep && d.copy_to < 0 && d.drain == 10 + d.t;
    check(l.size() == 256 && none, "the last layer: one dispatch per token, natural rows, nothing kept unless asked");
    uint64_t start, valid;
    open_qwen36::bidir::rows(1100, 256, 0, 1024, &start, &valid);
    check(start == 77 && valid == 1278, "a local layer past its window: the query's own window start, every image row after it");
}

}  // namespace

int main() {
    patch_tests();
    schedule_tests();
    std::printf("%s\n", g_failed ? "FAILED" : "PASS");
    return g_failed ? 1 : 0;
}
