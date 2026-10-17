# Specs not in the default export

Every `*.json` one level up is exported by `utilities/export-kernels.py`, the Linux CMake build
and release staging, so it ships. A spec here does not: it builds kernels for a model that
works, but whether its set ships is still the maintainer's call.

Export one by path:

```
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/pending/<name>.json
```

- `minicpm-v-4.7-1b.json`: MiniCPM-V-4.7-1B (text) read from its GGUF (OPEN-GGUF-QWEN35-SPEC).
  Its kernels are the 0.8B shape at q4_1 everywhere; it has its own set because `real_vocab`
  is in the spec hash and its quant map differs from the official 0.8B container's.
