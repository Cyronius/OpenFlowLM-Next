/// \file bidir_block.hpp
/// \brief OPEN-VISION-GEMMA3: an image block through the decode kernel, attending both ways inside
///        the block. specs/open-engine/plans/gemma3-images.md has the derivation; this header is the
///        schedule alone, so it is tested without a device (bidir_block_test.cpp).
#pragma once

#include <cstdint>
#include <vector>

namespace open_qwen36::bidir {

/// One single-layer dispatch for block token `t`.
struct Step {
    uint32_t t = 0;          ///< the token, 0-based in the block
    uint64_t drain = 0;      ///< the cache row its own K/V lands in
    int64_t copy_from = -1;  ///< before it runs, the host copies this cache row ...
    int64_t copy_to = -1;    ///< ... over this one (-1: no copy)
    bool keep = false;       ///< its output is the token's input to the next layer
};

/// Token t's cached rows: [start, start + valid) is the prefix inside its window, then every
/// image row but its own, which the dispatch computes itself. `window` 0 is a global layer.
inline void rows(uint64_t p0, uint32_t n, uint32_t t, uint64_t window, uint64_t* start, uint64_t* valid) {
    const uint64_t pos = p0 + t;
    const uint64_t s = (window && pos + 1 > window) ? pos + 1 - window : 0;
    *start = s;
    *valid = p0 + n - 1 - s;
}

/// One layer's dispatches in order. Pass A writes every token's K/V, pass B reads them all; the
/// copy before each pass-B token keeps every image row but its own in the rows the fill reads,
/// and the rows end in natural order. The last layer feeds no later layer, so it runs once with
/// natural drains, in order, which makes its final token's output exact (`keep_last`).
inline std::vector<Step> layer_schedule(uint64_t p0, uint32_t n, bool last, bool keep_last = false) {
    std::vector<Step> s;
    if (n == 0) return s;
    const int64_t tail = static_cast<int64_t>(p0 + n - 1);
    if (last) {
        for (uint32_t t = 0; t < n; ++t) s.push_back({t, p0 + t, -1, -1, keep_last && t + 1 == n});
        return s;
    }
    for (uint32_t t = 1; t < n; ++t) s.push_back({t, p0 + t - 1, -1, -1, false});
    for (uint32_t t = 0; t < n; ++t)
        s.push_back({t, static_cast<uint64_t>(tail), t ? tail : -1, t ? static_cast<int64_t>(p0 + t - 1) : -1, true});
    return s;
}

}  // namespace open_qwen36::bidir
