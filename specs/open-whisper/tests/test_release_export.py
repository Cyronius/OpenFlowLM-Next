# Traces: OPEN-WHISPER-RELEASE (canonical spec: specs/open-whisper/spec.md)
# A release without the set starts fine and only refuses Whisper installs at load, so the export must not miss it.
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "utilities" / "export-kernels.py"
WHISPER = str(REPO / "open_kernels" / "export_whisper_kernels.py")
BUILD_DIR = REPO / "open_kernels" / "designs" / "whisper_gemm" / "build-export"
OTHER_FAMILIES = ("export_qwen36_kernels.py", "export_gemm_rtp.py", "check_design_sets.py")


def run_export(monkeypatch, tmp_path, *args, failing=None):
    """Run main() with these arguments; returns (exit code, every command it ran, venv python)."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python3").write_text("")
    monkeypatch.setenv("OFLM_VENV_DIR", str(venv))
    monkeypatch.setenv("OFLM_SKIP_VENV_SETUP", "1")
    # main() rewrites these; setenv makes the teardown put them back.
    for var in ("PATH", "PYTHONPATH", "MLIR_AIE_ROOT"):
        monkeypatch.setenv(var, os.environ.get(var, ""))
    spec = importlib.util.spec_from_file_location("export_kernels_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calls: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        calls.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 1 if failing in calls[-1] else 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.shutil, "which", lambda tool: "/stub/bin/" + tool)
    monkeypatch.setattr(sys, "argv", ["export-kernels.py", *args])
    return mod.main(), calls, str(venv / "bin" / "python3")


def whisper_calls(calls):
    return [c for c in calls if WHISPER in c]


def test_release_staging_runs_the_whisper_export_last_with_force(monkeypatch, tmp_path):
    # utilities/release/stage-prebuilts.sh runs exactly this.
    rc, calls, py = run_export(monkeypatch, tmp_path, "--force")
    assert rc == 0
    assert calls[-1] == [py, WHISPER, "--build-dir", str(BUILD_DIR), "--force"]
    assert len(whisper_calls(calls)) == 1
    others = [i for i, c in enumerate(calls) if any(f in " ".join(c) for f in OTHER_FAMILIES)]
    assert others and max(others) < len(calls) - 1


def test_the_cmake_target_builds_it_without_force(monkeypatch, tmp_path):
    # src/CMakeLists.txt's export_kernels target passes --specs=${OFLM_KERNEL_SPECS}, empty by default.
    rc, calls, py = run_export(monkeypatch, tmp_path, "--specs=")
    assert rc == 0
    assert calls[-1] == [py, WHISPER, "--build-dir", str(BUILD_DIR)]
    assert len(whisper_calls(calls)) == 1


def test_the_build_cache_stays_out_of_the_shipped_tree():
    assert REPO / "src" / "xclbins" not in BUILD_DIR.parents
    probe = (BUILD_DIR / "qkv" / "final.xclbin").relative_to(REPO).as_posix()
    ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", probe], cwd=REPO)
    assert ignored.returncode == 0, "an unignored build cache makes stage-prebuilts.sh see a dirty tree"


@pytest.mark.parametrize("args", [
    ["--force"],
    ["--specs="],
    ["--specs", "qwen3-4b,gemma3-4b"],
    ["--skip-specs", "qwen25-3b"],
    ["--skip-bert"],
    ["--force", "--jobs", "4", "--skip-bert"],  # nix/open-kernels.nix with skipWhisper = false
])
def test_the_whisper_step_changes_no_other_command(monkeypatch, tmp_path, args):
    rc, with_whisper, _ = run_export(monkeypatch, tmp_path, *args)
    rc_without, without, _ = run_export(monkeypatch, tmp_path, *args, "--skip-whisper")
    assert rc == rc_without == 0
    assert len(whisper_calls(with_whisper)) == 1 and with_whisper[-1] == whisper_calls(with_whisper)[0]
    assert with_whisper[:-1] == without
    assert whisper_calls(without) == []


@pytest.mark.parametrize("args, runs", [
    (["--bert-only"], False),
    (["--skip-whisper"], False),
    (["--force", "--jobs", "4", "--skip-bert", "--skip-whisper"], False),  # nix/open-kernels.nix's default
    (["--skip-bert"], True),
    (["--specs", "qwen3-4b"], True),
    (["--skip-specs", "qwen3-4b"], True),
])
def test_which_flags_leave_it_out(monkeypatch, tmp_path, args, runs):
    rc, calls, _ = run_export(monkeypatch, tmp_path, *args)
    assert rc == 0
    assert len(whisper_calls(calls)) == (1 if runs else 0)


def test_a_failed_whisper_export_fails_the_run(monkeypatch, tmp_path):
    rc, calls, _ = run_export(monkeypatch, tmp_path, "--force", failing=WHISPER)
    assert rc == 1
    assert WHISPER in calls[-1]
