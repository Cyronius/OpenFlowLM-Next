# Traces: OPEN-HIDDEN-STATE (canonical spec: specs/open-engine/spec.md) -- the CPU reference must hold the weights the q4_1 pool holds.
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen3_ref as R  # noqa: E402


def test_bf16_rounds_to_nearest_even():
    ulp = 2.0 ** -7
    x = np.array([1.0, 1 + ulp / 2, 1 + 1.5 * ulp, 1 + 0.75 * ulp], np.float32)
    assert R.bf16_rne(x).tolist() == [1.0, 1.0, 1 + 2 * ulp, 1 + ulp]


def test_fp16_flushes_subnormals_and_rounds_to_even():
    x = np.array([1.0, 1e-6, -3e-5, 1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, 65504.0], np.float32)
    assert R.fp16_rne_flush(x).tolist() == [1.0, 0.0, 0.0, 1.0, 1 + 2.0 ** -9, 65504.0]


def _q4_block(d, m, q, with_min=True):
    lo, hi = q[:16], q[16:]
    nib = (lo | (hi << 4)).astype(np.uint8).tobytes()
    head = np.float16(d).tobytes() + (np.float16(m).tobytes() if with_min else b"")
    return np.frombuffer(head + nib, np.uint8)


def test_a_q4_1_block_packs_exactly_with_bf16_scales():
    q = np.arange(32) % 16
    d, m = 0.1, -0.7                                  # neither is a bf16 value, so the narrowing shows
    raw = _q4_block(d, m, q)
    w = R.q4_1_pool("Q4_1", raw, np.zeros((1, 32), np.float32))
    dd, mm = R.bf16_rne(np.float16(d).astype(np.float32)), R.bf16_rne(np.float16(m).astype(np.float32))
    assert np.array_equal(w[0], (dd * q + mm).astype(np.float32))
    assert dd != np.float16(d).astype(np.float32)


def test_a_q4_0_block_is_q4_1_with_min_minus_8d():
    q = (np.arange(32) * 7) % 16
    raw = _q4_block(0.25, 0.0, q, with_min=False)
    w = R.q4_1_pool("Q4_0", raw, np.zeros((1, 32), np.float32))
    assert np.array_equal(w[0], (0.25 * q - 2.0).astype(np.float32))


def test_a_q8_0_row_is_requantized_by_min_and_max():
    v = np.linspace(-1.0, 2.0, 32, dtype=np.float32)[None]
    w = R.q4_1_pool("Q8_0", None, v)
    d = R.fp16_rne_flush(np.float32(3.0) / np.float32(15.0))
    assert w[0, 0] == R.bf16_rne(np.float32(-1.0))
    assert len(np.unique(w)) == 16
    assert np.abs(w - v).max() <= d / 2 + 0.01         # half a step plus the bf16 narrowing of d and m


def _blob(b):
    return struct.pack("<Q", len(b)) + b


def test_gguf_float32_epsilon_reads_as_the_config_decimal(tmp_path):
    p = tmp_path / "x.gguf"
    kv = (_blob(b"general.architecture") + struct.pack("<I", 8) + _blob(b"qwen3")
          + _blob(b"qwen3.attention.layer_norm_rms_epsilon") + struct.pack("<If", 6, 1e-6)
          + _blob(b"qwen3.block_count") + struct.pack("<II", 4, 28))
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 3) + kv)
    md = R.gguf_kv(p)
    assert md == {"general.architecture": "qwen3", "qwen3.attention.layer_norm_rms_epsilon": 1e-06, "qwen3.block_count": 28}
