# Plan: the Whisper kernel set in release builds

2026-10-16, revised after review on 2026-10-17.

**Spec impact:** new requirement OPEN-WHISPER-RELEASE. Removed from "Not settled": "Releases
do not carry the Whisper kernel set".

## What changes

- `utilities/export-kernels.py` runs `open_kernels/export_whisper_kernels.py` as a third
  family, after the open_kernels specs and the BERT sets, so their commands, their order and
  their environment stay as they are. `--force` is passed on. `--skip-whisper` and
  `--bert-only` leave it out; `--specs`, `--skip-specs` and `--skip-bert` name the other two
  families and do not.
- The Whisper exporter gains `--build-dir`. A hand run keeps its default (`<out>/build`); the
  release export puts the build cache in `open_kernels/designs/whisper_gemm/build-export/`,
  which git ignores. Everything under `src/xclbins` ships, and the default cache is 34 MB of
  compiler intermediates, including zero-byte `.lock` files that `verify-prebuilts.py` refuses.
- The exporter keeps a stream's build only when the stream's sources and the toolchain match
  too, as it already did for `fa`. The CMake target runs it without `--force`, and a
  shape-only stamp kept old streams after a source edit or a toolchain change.
- `stage-prebuilts.sh`'s shape check reads a Whisper set by its own layout. Its generic rule
  wants `<kernel>/final.xclbin` and `insts.bin` in every directory, so it would refuse `fa/`
  (which holds `air.xclbin` and `air.insts.bin`). It also refuses kernel files sitting
  directly in an `open_kernels/` with no `whisper_kernels.json`: the exporter deletes the
  marker first, so that is what an export cut short leaves.
- `verify-package.sh` requires the set's `whisper_kernels.json`, as it requires the other two
  families, since the same run builds all three.
- `nix/open-kernels.nix` gains `skipWhisper`, which passes `--skip-whisper`, beside its
  `skipBert` and `skipSpecs`. It is true by default: the set has only been built on
  `ironvenv-requirements.txt`'s toolchain, not on the derivation's pins, and the package
  build fails on any failed set.

## Verification

- test: `test_release_export.py` (the commands `export-kernels.py` runs),
  `test_release_staging.py` (the shape check, run from the script on built trees) and
  `test_release_package.py` (`verify-package.sh` on a built TGZ), all in
  `specs/open-whisper/tests/`.
- manual: the stream cache (procedure and results in the spec, under OPEN-WHISPER-RELEASE).
