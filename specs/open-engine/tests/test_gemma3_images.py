# Traces: OPEN-VISION-GEMMA3 (canonical spec: specs/open-engine/spec.md)
"""The image mask the fp64 replica scores the NPU against, checked against transformers' own construction."""
from __future__ import annotations

import numpy as np
import pytest

from replica_gemma3_image import allowed

# text 0..2, image A 3..5, text 6, image B 7..8, text 9
TYPES = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
GROUPS = np.array([-1, -1, -1, 0, 0, 0, -1, 1, 1, -1])


def test_an_image_reads_itself_both_ways_and_the_past_causally():
    a = allowed(10, GROUPS, 0, bidir=True)
    assert a[3, 5] and a[5, 3]                  # inside image A, both ways
    assert a[3, 0] and not a[2, 3]              # A reads the text before it; that text does not read A
    assert not a[5, 7] and a[7, 5]              # A does not read image B; B reads A, which came first
    assert a[6, 5] and not a[6, 7]              # text after A reads it, not the image after it
    assert a[9].all()                           # the last text token reads everything


def test_the_causal_mask_is_the_one_gemma_does_not_use():
    a = allowed(10, GROUPS, 0, bidir=False)
    assert np.array_equal(a, np.tril(np.ones((10, 10), bool)))


def test_the_sliding_window_still_bounds_the_past():
    a = allowed(10, GROUPS, 4, bidir=True)
    assert not a[8, 3] and a[8, 5]              # i - j < 4 holds for the past ...
    assert a[3, 5]                              # ... and a future row of the same image stays readable


def test_the_mask_is_transformers_own():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import Gemma3TextConfig
    from transformers.models.gemma3 import modeling_gemma3 as M
    if not hasattr(M, "create_masks_for_vision_model"):
        pytest.skip(f"transformers {transformers.__version__} builds Gemma 3's image mask another way")
    for window in (1024, 4):
        cfg = Gemma3TextConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=1,
                               num_key_value_heads=1, head_dim=8, vocab_size=16, sliding_window=window)
        cfg._attn_implementation = "eager"
        tt = torch.from_numpy(TYPES[None].copy())
        masks = M.create_masks_for_vision_model(
            config=cfg, inputs_embeds=torch.zeros(1, 10, 8), attention_mask=torch.ones(1, 10, dtype=torch.long),
            past_key_values=None, position_ids=torch.arange(10)[None],
            block_sequence_ids=M.get_block_sequence_ids_for_mask(tt))
        for kind, w in (("full_attention", 0), ("sliding_attention", window)):
            got = masks[kind][0, 0].numpy()
            got = got > -1 if got.dtype != bool else got
            assert np.array_equal(got, allowed(10, GROUPS, w, bidir=True)), (kind, window)
