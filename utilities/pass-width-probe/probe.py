"""Pass-width probe: what a q4 pass over M rows costs against a pass over one.

Phase 0 of the diffusion-LM plan (specs/open-engine/plans/diffusion-lm.md). It builds open_kernels/designs/gemm_q4 at several
row counts for two shapes and times them through harness/run_kernel. Rounds are
interleaved across configs, so drift on the box lands on every config alike.

Shapes:
- a: N 8192 x K 2048 (PR #90's shape).
- b: N 24576 x K 4096 (Qwen3-8B's gate|up, the largest per-layer stream).
Both are built from gemv_q4's verified block (make_test.py --n / --k), so every
config is checked for correctness before it is timed.

    python probe.py build [names...]            # WSL builds, skipping built ones
    python probe.py run [--rounds 10] [names...]  # fixtures + check, then timed rounds

`run` needs a quiet box: no other NPU client, no CPU load.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OK = ROOT / "open_kernels"
DESIGN = OK / "designs" / "gemm_q4"
# name -> shape and build knobs. bpt / acc follow PR #90's L1 law,
# 20736 + M * (576 + 256 * bpt * acc_depth) <= 65536, with n_tiles = bpc / bpt <= 4.
CONFIGS: dict[str, dict] = {
    "a1": dict(n=8192, k=2048, m=1, bpt=4),
    "a8": dict(n=8192, k=2048, m=8, bpt=4),
    "a12": dict(n=8192, k=2048, m=12, bpt=4),
    "a16": dict(n=8192, k=2048, m=16, bpt=4),
    "a32": dict(n=8192, k=2048, m=32, bpt=2, acc=1),
    "a16i8": dict(n=8192, k=2048, m=16, bpt=4, int8=1),
    "a16q4k": dict(n=8192, k=2048, m=16, bpt=4, q4k=1),
    "b1": dict(n=24576, k=4096, m=1, bpt=12),
    "b8": dict(n=24576, k=4096, m=8, bpt=12),
    "b12": dict(n=24576, k=4096, m=12, bpt=6, acc=1),
    "b16": dict(n=24576, k=4096, m=16, bpt=6, acc=1),
    "b32": dict(n=24576, k=4096, m=32, bpt=3, acc=1),
    "b16i8": dict(n=24576, k=4096, m=16, bpt=6, acc=1, int8=1),
}
CORES = 32
WSL_ENV = ("source ~/ironenv142/bin/activate && source ~/mlir-aie/utils/env_setup.sh >/dev/null && "
           "export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib")
# harness/build.cmd's output; RUN_KERNEL points at another checkout's build instead
RUN_KERNEL = Path(os.environ.get("RUN_KERNEL", OK / "harness" / "out" / "run_kernel.exe"))
RUN_LINE = re.compile(r"^run (\S+) \[\d+ bufs\] -> state (\d+) \((\d+\.\d+) ms\)")


def build_dir(name: str) -> Path:
    return DESIGN / f"build_probe_{name}"


def wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


def build(name: str) -> None:
    c = CONFIGS[name]
    if (build_dir(name) / "final.xclbin").exists():
        print(f"{name}: built")
        return
    env = (f"GEMM_N={c['n']} GEMM_K={c['k']} GEMM_RS=2 GEMM_CORES={CORES} GEMM_M={c['m']} "
           f"GEMM_PER_CALL=1 GEMM_BLOCKED=1 GEMM_BPT={c['bpt']}")
    if c.get("acc"):
        env += f" GEMM_ACC_DEPTH={c['acc']}"
    if c.get("int8"):
        env += " GEMM_INT8=1"
    if c.get("q4k"):
        env += " GEMM_Q4K=1 GEMM_PREQ=1"
    cmd = (f"{WSL_ENV} && cd {wsl_path(OK)} && {env} python build_design.py "
           f"designs/gemm_q4/gemm_q4.py designs/gemm_q4/{build_dir(name).name}")
    print(f"{name}: building ({env})", flush=True)
    r = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--exec", "bash", "-lc", cmd],
                       capture_output=True, text=True)
    if r.returncode != 0 or not (build_dir(name) / "final.xclbin").exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"{name}: build failed")
    print(f"{name}: ok")


def run_cfg(cfg: Path) -> list[tuple[int, float]]:
    """(state, ms) per `run` line, from run_kernel with the cfg's directory as cwd."""
    p = subprocess.run([str(RUN_KERNEL), cfg.name], cwd=cfg.parent, capture_output=True, text=True)
    out = [(int(m.group(2)), float(m.group(3))) for m in map(RUN_LINE.match, p.stdout.splitlines()) if m]
    if p.returncode != 0 or not out:
        raise SystemExit(f"{cfg.name}: run_kernel exit {p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-800:]}")
    return out


def prepare(name: str, runs: int) -> Path:
    """Fixtures, one correctness check, then a timing cfg of its own (make_test's
    outputs are keyed by M only, so configs sharing an M would collide)."""
    c = CONFIGS[name]
    m = c["m"]
    args = [sys.executable, "make_test.py", "--m", str(m), "--n", str(c["n"]), "--k", str(c["k"]),
            "--cores", str(CORES), "--blocked", "--bpt", str(c["bpt"]), "--build", build_dir(name).name,
            "--runs", "1"]
    if c.get("q4k"):
        args.append("--preq")
    subprocess.run(args, cwd=DESIGN, check=True, capture_output=True)
    cfg = DESIGN / f"run_m{m}.cfg"
    states = run_cfg(cfg)
    if any(s != 4 for s, _ in states):
        raise SystemExit(f"{name}: kernel state {states}")
    bar = ["i8"] if c.get("int8") else []
    chk = subprocess.run([sys.executable, "check.py", str(m), *bar], cwd=DESIGN,
                         capture_output=True, text=True)
    last = chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else chk.stderr[-300:]
    print(f"{name}: {last}")
    if chk.returncode != 0:
        raise SystemExit(f"{name}: wrong output, not timing it")
    # timing cfg: same buffers, its own x file, no dump
    x_src = DESIGN / f"x_m{m}.bin"
    x_own = DESIGN / f"x_probe_{name}.bin"
    shutil.copyfile(x_src, x_own)
    text = cfg.read_text(encoding="utf-8").replace(x_src.name, x_own.name)
    lines = [l for l in text.splitlines() if not l.startswith(("run ", "dump "))]
    lines += ["run k w x y"] * runs
    own = DESIGN / f"run_probe_{name}.cfg"
    own.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return own


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "run"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--runs", type=int, default=30, help="submits per config per round")
    ap.add_argument("--warm", type=int, default=3, help="submits dropped per round")
    a = ap.parse_args()
    names = a.names or list(CONFIGS)
    if a.action == "build":
        for n in names:
            build(n)
        return 0

    cfgs = {n: prepare(n, a.runs) for n in names}
    times: dict[str, list[float]] = {n: [] for n in names}
    for r in range(a.rounds):
        for n in names:
            ms = [t for s, t in run_cfg(cfgs[n])[a.warm:] if s == 4]
            times[n].append(min(ms))
        print(f"round {r + 1}/{a.rounds} done", flush=True)

    print(f"\n{'config':8s} {'N':>6s} {'K':>5s} {'M':>3s} {'min ms':>8s} {'med ms':>8s}  per-round minima")
    for n in names:
        c = CONFIGS[n]
        print(f"{n:8s} {c['n']:6d} {c['k']:5d} {c['m']:3d} {min(times[n]):8.3f} "
              f"{statistics.median(times[n]):8.3f}  {' '.join(f'{t:.3f}' for t in times[n])}")
    for shape in ("a", "b"):
        one = f"{shape}1"
        if one in times:
            base = statistics.median(times[one])
            print(f"\nshape {shape}: median time relative to M = 1 ({base:.3f} ms)")
            for n in names:
                if n.startswith(shape) and n != one:
                    print(f"  {n:8s} M={CONFIGS[n]['m']:2d}  x{statistics.median(times[n]) / base:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
