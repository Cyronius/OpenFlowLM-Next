// MXFP4 GEMV tile for a compile-only Peano look at its cost (2026-10-16; plans/gptoss-expert-layout.md).
//   python open_kernels/kernel_remarks.py utilities/aie-probes/gemv_mx4_probe.cc -- -Os [-DMX4_DECODE=2]
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../../open_kernels/designs/gemv_q4/gemv_tab.h"

// The tile of open_kernels/model/gptoss_layout.py: nibble planes at 0 and 2304, E8M0 at 4608, bias at 4928.
#ifndef MX4_NB
#define MX4_NB 9
#endif
static constexpr unsigned kRows = 32;
static constexpr unsigned kNB = MX4_NB;
static constexpr unsigned kTileK = 32 * kNB;
static constexpr unsigned kPlane = 8 * kTileK;
static constexpr unsigned kBiasOff = 4928;

#ifndef MX4_DECODE
#define MX4_DECODE 1
#endif
#if MX4_DECODE == 1
// codes 0..15 (bit 3 the sign) -> 2 x e2m1 as int8: mag = max(t, 2t-4, 4t-16), t = code & 7
static inline aie::vector<int8_t, 64> e2m1x2(const aie::vector<uint8_t, 64> &c) {
  const aie::vector<int8_t, 64> t = aie::bit_and((uint8_t)7, c).template cast_to<int8_t>();
  const aie::vector<int8_t, 64> t2 = aie::add(t, t);
  const aie::vector<int8_t, 64> t4 = aie::add(t2, t2);
  const aie::vector<int8_t, 64> mag =
      aie::max(t, aie::max(aie::sub(t2, (int8_t)4), aie::sub(t4, (int8_t)16)));
  const auto neg = aie::ge(c, (uint8_t)8);
  return aie::select(mag, aie::neg(mag), neg);
}
#else
// codes sorted by value at pack time (0..14) -> 2 x e2m1 + 12; the 12 leaves through the block sums like q4_1's min
static inline aie::vector<int8_t, 64> e2m1x2(const aie::vector<uint8_t, 64> &c) {
  const aie::vector<int8_t, 64> t = c.template cast_to<int8_t>();
  const aie::vector<int8_t, 64> t2 = aie::add(t, t);
  const aie::vector<int8_t, 64> t4 = aie::add(t2, t2);
  const aie::vector<int8_t, 64> lo = aie::min(aie::min(aie::add(t, (int8_t)5), aie::add(t2, (int8_t)2)), t4);
  return aie::max(lo, aie::max(aie::sub(t2, (int8_t)6), aie::sub(t4, (int8_t)32)));
}
#endif

__attribute__((noinline)) inline void gemv_mx4_tile(const uint8_t *__restrict tile,
                                                    const uint8_t *__restrict tab, unsigned K,
                                                    unsigned kt, bool first, bool last,
                                                    float *__restrict y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const uint8_t *__restrict nib0 = tile;
  const uint8_t *__restrict nib1 = tile + kPlane;
  const uint8_t *__restrict ep = tile + 2 * kPlane;
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kTileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * K) + kt * kNB;

  aie::accum<accfloat, kRows> acc;
  if (first) {
    // the bias seeds the accumulator, in its [evens | odds] lane order
    const aie::vector<bfloat16, kRows> b = aie::load_v<kRows>((const bfloat16 *)(tile + kBiasOff));
    auto [be, bo] = aie::interleave_unzip(b, b, 1);
    acc.from_vector(aie::concat(be.template extract<16>(0), bo.template extract<16>(0)));
  } else {
    acc.from_vector(aie::load_v<kRows>(y));
  }

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kNB; ++kb) {
    aie::vector<int32_t, 8> ve[2], vo[2];
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict src = (rb == 0 ? nib0 : nib1) + kb * 256;
      aie::mmul<4, 8, 8, int16_t, int8_t> Ce, Co;
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const aie::vector<int16_t, 32> A =
            aie::load_v<8>(xi + kb * 32 + oc * 8).template grow_replicate<32>();
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(src + oc * 64);
        const aie::vector<uint8_t, 64> lo = aie::bit_and((uint8_t)0x0F, q);
#ifdef MX4_NO_SHIFT
        const aie::vector<uint8_t, 64> hi = aie::bit_and((uint8_t)0x0E, q);   // cost probe only: wrong values
#else
        // the high nibbles move down through a 16-bit view: a uint8 shift does not legalize
        const aie::vector<uint8_t, 64> hi = aie::bit_and(
            (uint8_t)0x0F, aie::downshift(q.template cast_to<uint16_t>(), 4).template cast_to<uint8_t>());
#endif
        if (oc == 0) {
          Ce.mul(A, e2m1x2(lo));
          Co.mul(A, e2m1x2(hi));
        } else {
          Ce.mac(A, e2m1x2(lo));
          Co.mac(A, e2m1x2(hi));
        }
      }
      ve[rb] = Ce.template to_vector<int32_t>().template extract<8>(0);
      vo[rb] = Co.template to_vector<int32_t>().template extract<8>(0);
    }
    const aie::vector<int32_t, kRows> vi = aie::concat(ve[0], ve[1], vo[0], vo[1]);
    aie::accum<accfloat, kRows> part;
    part.from_vector(aie::to_float<float>(vi, sh[kb]));
    const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
    const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
    // 2^(e - 128) as bf16 bits (e - 1) << 7: exact for every E8M0 byte >= 2
    const aie::vector<int16_t, kRows> e16 = aie::unpack(aie::load_v<kRows>(ep + kb * kRows)).template cast_to<int16_t>();
    const aie::vector<bfloat16, kRows> sc =
        aie::upshift(aie::sub(e16, (int16_t)1), 7).template cast_to<bfloat16>();
    auto [se, so] = aie::interleave_unzip(sc, sc, 1);
    const aie::vector<bfloat16, kRows> sperm = aie::concat(se.template extract<16>(0), so.template extract<16>(0));
    acc = aie::mac(acc, hi, sperm);
    acc = aie::mac(acc, lo, sperm);
#if MX4_DECODE != 1
    const bfloat16 *__restrict xsh = (const bfloat16 *)(tab + 2 * K + 4 * (K / 32)) + kt * kNB;
    const aie::vector<bfloat16, kRows> m12 = aie::mul(sperm, (bfloat16)(-12.0f)).template to_vector<bfloat16>();
    acc = aie::mac(acc, m12, xsh[kb]);
    acc = aie::mac(acc, m12, xsh[kb + K / 32]);
#endif
  }

  const aie::vector<float, kRows> yv = acc.template to_vector<float>();
  if (last) {
    auto [r0, r1] = aie::interleave_zip(yv.template extract<16>(0), yv.template extract<16>(1), 1);
    aie::store_v(y, r0);
    aie::store_v(y + 16, r1);
  } else {
    aie::store_v(y, yv);
  }
}

extern "C" {
void gemv_mx4_probe(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                    int32_t kt) {
  gemv_mx4_tile(t, tab, kTileK * 10, (unsigned)kt, kt == 0, kt == 9, y);
}
}
