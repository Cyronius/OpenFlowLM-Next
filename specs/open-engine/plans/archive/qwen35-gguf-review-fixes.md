# qwen35 GGUF route: review fixes (2026-10-17)

Follows `qwen35-gguf-view.md`. Modified requirements, no new or removed ones.

- **OPEN-GGUF-QWEN35-SPEC.** MiniCPM-V-4.7-1B's spec moves to `recipes/specs/pending/`, outside the
  default export, so no build or release ships its set until the maintainer decides. The "differs
  only in `real_vocab`" criterion is restated as what it tests (a spec differing only in
  `real_vocab`), and the spec records that the official 0.8B container derives `linear_out=q8`.
- **OPEN-GGUF-Q4NX-VIEW.** The view refuses a body tensor whose rows disagree with the metadata.
  `oflm-add`'s qwen3.5 GGUF route needs `--open-kernels` (opt-in), so a default install is
  unchanged. A tied model's `token_embd` gets the same head exemption as `output.weight`.

Tests: `test_gguf_q4nx_view.py`, `test_qwen35_gguf_spec.py`, `test_gguf_installer.py`, and
`gguf_pack_test.cpp` (ctest GGUF-PACK). Merged into `spec.md`; archived.
