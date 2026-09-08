//===- dequant_q4.cc ---------------------------------------*- C++ -*-===//
//
// One q4_1 pool chunk (32 output rows x 256 k, gemv_q4.h's layout) -> one k-half (DQ_KH k)
// of the matmul's B operand: bf16, tiled the way mm.cc's row-major B wants it,
//   bt[k/8][row/8][8 k][8 rows]     (r, s, t = 4, 8, 8)
// with value = nib * d[kb][r] + m[kb][r].
//
// The nibbles are row-fastest: 16 rows per 8 bytes at one k, even nibble = even row. A
// 64-byte load is therefore eight k's of one 16-row half. Masking the two nibbles gives
// the even rows (low nibble) and the odd rows (high nibble, still in place) as
// two 64-lane vectors, and ONE byte-level zip puts them back in row order:
// [k0: r0 r1 .. r15][k1: ..] .. -- four k's per 64-lane result.
//
// No int -> float conversion anywhere: 0x4300 | nib is the bf16 of 128 + nib exactly (nib
// < 16 sits in the low mantissa bits), so
//   value = (128 + nib) * d + (m - 128 d)
// with m - 128 d an fp32 accumulator built once per (k-block, row half). The odd rows'
// high nibble is shifted down AFTER widening to 16 bits (a uint8 shift is not legal here);
// 16 nib would not fit the seven mantissa bits, which is how the first build got every odd
// row wrong. One bf16 rounding per value, at the end (conv_even).
//
// Every vector op is 512 bits wide (64 x u8, 32 x u16 / bf16) -- the backend could not
// legalize a 128-bit AND -- except the 8-lane stores into the 8 x 8 blocks.
// TRAPS (LLMNpuTest CLAUDE.md, gemv_q4.h): no scalar float; no fp32 vector multiply;
// aie::downshift on uint8 is an error (hence the 16x trick); default rounding is floor.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DQ_KH
#define DQ_KH 128            // k per half; 256 / DQ_KH halves per chunk
#endif
static constexpr unsigned kRows = 32, kTileK = 256, kKB = 32, kKH = DQ_KH;
static constexpr unsigned kDOff = 0, kMOff = 512, kNibOff = 1024;
static_assert(kKH % kKB == 0 && kTileK % kKH == 0, "dequant_q4: a half is whole k-blocks");

extern "C" void dequant_q4_half(const uint8_t *__restrict chunk, bfloat16 *__restrict bt, int half) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *__restrict d = (const bfloat16 *)(chunk + kDOff);
  const bfloat16 *__restrict m = (const bfloat16 *)(chunk + kMOff);
  const unsigned k0 = (unsigned)half * kKH;
  const aie::vector<uint16_t, 32> magic = aie::broadcast<uint16_t, 32>((uint16_t)0x4300);
  const aie::vector<uint16_t, 32> nibmask = aie::broadcast<uint16_t, 32>((uint16_t)0x000F);
  const aie::vector<uint8_t, 64> lomask = aie::broadcast<uint8_t, 64>((uint8_t)0x0F);
  const aie::vector<uint8_t, 64> himask = aie::broadcast<uint8_t, 64>((uint8_t)0xF0);
  const bfloat16 m128 = (bfloat16)-128.0f;

#pragma clang loop unroll(disable)
  for (unsigned kb = k0 / kKB; kb < (k0 + kKH) / kKB; ++kb) {
#pragma clang loop unroll(full)
    for (unsigned rh = 0; rh < 2; ++rh) {                    // rows 0..15 / 16..31
      const aie::vector<bfloat16, 16> d16 = aie::load_v<16>(d + kb * kRows + rh * 16);
      const aie::vector<bfloat16, 16> m16 = aie::load_v<16>(m + kb * kRows + rh * 16);
      const aie::vector<bfloat16, 32> d32 = aie::concat(d16, d16);                                  // [rows][rows]
      aie::accum<accfloat, 32> off;
      off.from_vector(aie::concat(m16, m16));
      off = aie::mac(off, d32, m128);                                                               // m - 128 d
      const uint8_t *__restrict nib = chunk + kNibOff + rh * 2048 + kb * 256;                       // 32 k x 8 B
      bfloat16 *__restrict out = bt + ((kb * kKB - k0) / 8) * (kRows / 8) * 64 + rh * 2 * 64;
#pragma clang loop unroll(disable)
      for (unsigned g = 0; g < kKB / 8; ++g) {                // eight k's per 64-byte load
        const aie::vector<uint8_t, 64> q = aie::load_v<64>(nib + g * 64);
        auto z = aie::interleave_zip(aie::bit_and(lomask, q), aie::bit_and(himask, q), 1);   // rows in order
        const aie::vector<uint8_t, 64> zz[2] = {z.first, z.second};                            // k 0..3, k 4..7
#pragma clang loop unroll(full)
        for (unsigned zi = 0; zi < 2; ++zi) {
#pragma clang loop unroll(full)
          for (unsigned c = 0; c < 2; ++c) {                  // two k's of 16 rows
            // even lanes hold nib, odd lanes 16 nib: (w >> 4) | (w & 0xF) is nib on every lane
            const aie::vector<uint16_t, 32> w = aie::unpack(zz[zi].template extract<32>(c));
            const aie::vector<bfloat16, 32> v =
                aie::bit_or(aie::bit_or(aie::downshift(w, 4), aie::bit_and(w, nibmask)), magic).template cast_to<bfloat16>();
            aie::accum<accfloat, 32> a = off;
            a = aie::mac(a, v, d32);
            const aie::vector<bfloat16, 32> o = a.template to_vector<bfloat16>();
            const unsigned k = g * 8 + zi * 4 + c * 2;        // k within the block, then k + 1
            // block (k/8, rh*2 + {0,1}), row k%8: 8 rows each
            bfloat16 *__restrict o0 = out + ((k / 8) * (kRows / 8)) * 64 + (k % 8) * 8;
            aie::store_v(o0, o.template extract<8>(0));
            aie::store_v(o0 + 64, o.template extract<8>(1));
            aie::store_v(o0 + 8, o.template extract<8>(2));
            aie::store_v(o0 + 64 + 8, o.template extract<8>(3));
          }
        }
      }
    }
  }
}
