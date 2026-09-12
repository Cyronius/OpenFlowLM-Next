# Traces: OPEN-MOE-BATCH (canonical spec: specs/open-engine/spec.md)
"""The routed experts are already in the band law the prefill GEMM reads.

The expert pools are packed by `expert_stripes` (up|gate, 128-row stripes) and
`expert_down`, neither of which is the 64-row band law `gemm_q4_prefill.py`'s
weight tap walks -- which is why the plan called for a stripe reader and a
repack. They are the same law: a 128-row stripe is two bands interleaved at
k-tile granularity, so a band is a STRIDED read of the pool as packed.

These tests derive the band's byte pattern from the packer's own permutations,
so they fail if either law is ever repermuted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "open_kernels"))

from recipes.pack import down_perm, std_perm, stripe_transpose  # noqa: E402

CH = 5120          # one q4_1 chunk: 32 rows x 256 K
BAND_BYTES = 10240  # gemm_q4_prefill.BAND_BYTES: 64 rows x 256 K = two chunks


def chunk_map(perm, in_dim: int) -> list[tuple[int, int]]:
    """pool chunk index -> (rowblock32, ktile), from a packer permutation."""
    ncol = in_dim // 256
    return [(int(f) // ncol, int(f) % ncol) for f in perm]


def test_a_stripe_is_two_bands_interleaved_at_k_tile_granularity():
    # one 128-row stripe of a [512, 2048] up / gate projection
    stripe = chunk_map(stripe_transpose(2048), 2048)
    law = chunk_map(std_perm(128, 2048), 2048)
    # a band is 16 chunks: rowblocks 2b and 2b+1 across the 8 k-tiles. Inside a 128-row
    # stripe that is band 0 (rowblocks 0, 1) and band 1 (rowblocks 2, 3).
    for band, first in ((0, 0), (1, 2)):
        want = [(rb + 2 * band, kt) for rb, kt in law[:16]]
        got = [stripe.index(w) for w in want]
        assert got == [first + 4 * j + h for j in range(8) for h in range(2)], (band, got)


def test_the_up_gate_band_tap_is_sizes_8_10240_strides_20480_1():
    stripe = chunk_map(stripe_transpose(2048), 2048)
    for band, offset in ((0, 0), (1, BAND_BYTES)):
        want = [(rb + 2 * band, kt) for rb, kt in chunk_map(std_perm(128, 2048), 2048)[:16]]
        byte_offsets = [stripe.index(w) * CH for w in want]
        # every consecutive PAIR of chunks is one 10240 B band element, and the elements
        # step by 20480 B -- exactly TensorAccessPattern(off, [8, 10240], [20480, 1])
        assert byte_offsets[0] == offset
        elems = byte_offsets[0::2]
        assert elems == [offset + 20480 * j for j in range(8)]
        assert byte_offsets[1::2] == [e + CH for e in elems], "a band element is two contiguous chunks"


def test_the_down_band_tap_has_the_same_shape_with_two_elements():
    # one expert's down [2048, 512]: 128 chunks, K = 512 -> 2 band elements per band
    dn = chunk_map(down_perm(128), 512)
    law = chunk_map(std_perm(128, 512), 512)
    for band in range(4):
        want = [(rb + 2 * band, kt) for rb, kt in law[:4]]
        byte_offsets = [dn.index(w) * CH for w in want]
        base = (band // 2) * 40960 + (band % 2) * BAND_BYTES
        assert byte_offsets[0::2] == [base, base + 20480], (band, byte_offsets)
        assert byte_offsets[1::2] == [base + CH, base + 20480 + CH]


@pytest.mark.parametrize("e", [0, 1, 255])
def test_an_experts_bands_land_inside_its_own_pool_region(e):
    """The offsets the kernel will emit, against the pack plan's own arithmetic."""
    stripe_bytes, stripes = 163840, 4
    up_gate_region = 2 * stripes * stripe_bytes          # per expert
    for band in range(8):                                 # 512 rows of up
        off = e * up_gate_region + 2 * (band // 2) * stripe_bytes + (band % 2) * BAND_BYTES
        last = off + 20480 * 7 + BAND_BYTES
        assert e * up_gate_region <= off < (e + 1) * up_gate_region
        assert last <= (e + 1) * up_gate_region
    down_bytes = 655360
    for band in range(32):                                # 2048 rows of down
        off = e * down_bytes + (band // 2) * 40960 + (band % 2) * BAND_BYTES
        assert e * down_bytes <= off < (e + 1) * down_bytes
        assert off + 20480 + BAND_BYTES <= (e + 1) * down_bytes
