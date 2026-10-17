# Traces: OPEN-WHISPER-RELEASE (canonical spec: specs/open-whisper/spec.md)
# verify-package.sh is the release workflow's last look at a package before it is published.
from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
VERIFY = REPO / "utilities" / "release" / "verify-package.sh"
SHARE = "opt/openflowlm/share/oflm"
DIRS = [f"{SHARE}/xclbins", f"{SHARE}/utilities/oflm-test"]
FILES = ["usr/bin/oflm", "etc/profile.d/openflowlm.sh", "opt/openflowlm/bin/oflm",
         "opt/openflowlm/lib64/libq4_npu_eXpress.so", f"{SHARE}/model_list.json",
         f"{SHARE}/oflm-add/oflm-add.py", f"{SHARE}/xclbins/Qwen3-4B-NPU2/open_kernels/dx/final.xclbin",
         f"{SHARE}/xclbins/BERT-h384-bf16/gemm_rtp/final.xclbin",
         f"{SHARE}/xclbins/Whisper-V3-Turbo-NPU2/open_kernels/final.xclbin"]
MARKER = f"{SHARE}/xclbins/Whisper-V3-Turbo-NPU2/open_kernels/whisper_kernels.json"
NO_MARKER = ("  FAIL: ships no Whisper kernel set. utilities/export-kernels.py builds it in the same run "
             "as the open kernel sets, so without it the open Whisper engine refuses every install and "
             "the bundle is incomplete.")
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="verify-package.sh needs bash and tar")


def tgz(path: Path, files: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as t:
        for d in DIRS:
            info = tarfile.TarInfo(f"openflowlm-0.0.0-Linux/{d}")
            info.type = tarfile.DIRTYPE
            t.addfile(info)
        for f in files:
            t.addfile(tarfile.TarInfo(f"openflowlm-0.0.0-Linux/{f}"), io.BytesIO(b""))


def verify(root: Path, *args: str) -> tuple[int, str]:
    # On stdin, as bytes: a Windows checkout holds the script with CRLF line ends, which bash will not run.
    r = subprocess.run([BASH, "-s", "--", "pkgs", *args], cwd=root, capture_output=True,
                       input=VERIFY.read_text(encoding="utf-8").encode())
    return r.returncode, r.stdout.decode()


def test_a_package_needs_the_whisper_marker(tmp_path):
    # One run checks both: the set's xclbin alone does not count.
    tgz(tmp_path / "pkgs" / "a-with.tar.gz", [*FILES, MARKER])
    tgz(tmp_path / "pkgs" / "b-without.tar.gz", FILES)
    rc, out = verify(tmp_path)
    with_marker, without = out.split("== b-without.tar.gz")
    assert rc == 1
    assert "ok:   ships the Whisper kernel set" in with_marker and "FAIL" not in with_marker
    assert [line for line in without.splitlines() if "FAIL:" in line] == [NO_MARKER]


def test_allow_missing_kernels_lets_it_through(tmp_path):
    tgz(tmp_path / "pkgs" / "b-without.tar.gz", FILES)
    rc, out = verify(tmp_path, "--allow-missing-kernels")
    assert rc == 0, out
    assert "ships no Whisper kernel set (--allow-missing-kernels)" in out
