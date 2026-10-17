# Traces: OPEN-WHISPER-RELEASE (canonical spec: specs/open-whisper/spec.md)
# Staging's shape check is the last stop before a half-built kernel set is committed to the release branch.
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
STAGE = REPO / "utilities" / "release" / "stage-prebuilts.sh"
WHISPER = "src/xclbins/Whisper-V3-Turbo-NPU2/open_kernels"
OTHER = "src/xclbins/Qwen3-4B-NPU2/open_kernels"
STREAMS = ("insts_conv1.bin", "insts_qkv.bin")
COMPLETE = {"complete": True, "fa": True}


def shape_check() -> str:
    # The check is an inline block of stage-prebuilts.sh; this runs that block as the script has it.
    text = STAGE.read_text(encoding="utf-8")
    start = text.index("\n", text.index("<<'PY' || die \"the built kernels are not complete")) + 1
    return text[start:text.index("\nPY\n", start)]


def put(root: Path, rel: str, data: object = b"x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())


def other_family(root: Path) -> None:
    for f, data in (("manifest.json", {}), ("toolchain.json", {}), ("spec.json", {"name": "qwen3-4b"})):
        put(root, f"{OTHER}/{f}", data)
    for kernel in ("dx", "lm_head_q4"):
        put(root, f"{OTHER}/{kernel}/final.xclbin")
        put(root, f"{OTHER}/{kernel}/insts.bin")


def whisper_set(root: Path, marker: dict | None) -> None:
    for f in ("final.xclbin", "insts.bin", *STREAMS, "toolchain.json"):
        put(root, f"{WHISPER}/{f}")
    put(root, f"{WHISPER}/design.json", {"streams": [{"file": s} for s in STREAMS]})
    for f in ("air.xclbin", "air.insts.bin", "fa.json", "toolchain.json"):
        put(root, f"{WHISPER}/fa/{f}")
    if marker is not None:
        put(root, f"{WHISPER}/whisper_kernels.json", marker)


def check(root: Path) -> subprocess.CompletedProcess:
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    return subprocess.run([sys.executable, "-c", shape_check(), *files], cwd=root,
                          capture_output=True, text=True)


def test_a_complete_whisper_set_passes_beside_another_family(tmp_path):
    other_family(tmp_path)
    whisper_set(tmp_path, COMPLETE)
    r = check(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "2 kernel directories" in r.stdout and "1 Whisper set(s)" in r.stdout


def test_an_export_cut_short_is_refused(tmp_path):
    # What a killed export leaves: it deletes the marker and fa/ before it replaces the streams.
    other_family(tmp_path)
    whisper_set(tmp_path, None)
    shutil.rmtree(tmp_path / WHISPER / "fa")
    r = check(tmp_path)
    assert r.returncode == 1
    assert "no whisper_kernels.json" in r.stderr


@pytest.mark.parametrize("change, message", [
    (lambda root: put(root, f"{WHISPER}/whisper_kernels.json", {"complete": False, "fa": True}), "not complete"),
    (lambda root: (root / WHISPER / "insts_qkv.bin").unlink(), "no insts_qkv.bin"),
    (lambda root: shutil.rmtree(root / WHISPER / "fa"), "no fa/air.xclbin"),
    (lambda root: put(root, f"{WHISPER}/build/qkv/final.xclbin"), "carries build/"),
], ids=["an --only export", "a stream missing", "fa missing", "the build cache inside"])
def test_a_whisper_set_with_a_gap_is_refused(tmp_path, change, message):
    other_family(tmp_path)
    whisper_set(tmp_path, COMPLETE)
    change(tmp_path)
    r = check(tmp_path)
    assert r.returncode == 1
    assert message in r.stderr


def test_the_other_families_keep_their_own_rule(tmp_path):
    other_family(tmp_path)
    whisper_set(tmp_path, COMPLETE)
    (tmp_path / OTHER / "dx" / "insts.bin").unlink()
    r = check(tmp_path)
    assert r.returncode == 1
    assert "dx: no insts.bin" in r.stderr
