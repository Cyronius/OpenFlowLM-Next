"""MLIR of every recipe build that runs a given design, for proving a refactor moved no byte (2026-10-16).

    python utilities/mlir_equiv.py dump OUT [--designs dense/dx.py,dense/dx_attn.py] [--specs a.json,b.json] [--root TREE]
    python utilities/mlir_equiv.py diff OLD NEW
    python utilities/mlir_equiv.py ppcheck NEW_DUMP OLD_ROOT [NEW_ROOT]

Run in the mlir-aie environment. Each build is generated in its own process, with the build's own
env and OPEN_KERNELS_SPEC, exactly as build_design.py would (CallableDesign.as_mlir(), no Peano).
The MLIR does not carry the kernels' compile flags, so `dump` also records every ExternalFunction's
source and flags, and `ppcheck` preprocesses each (source, flags) pair against both trees: equal
MLIR plus equal preprocessed kernels is equal binaries.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OK = REPO / "open_kernels"

GEN = r'''
import importlib.util, json, sys
from pathlib import Path
import aie.iron as iron
import aie.iron.kernel as _k
_efs = []
_init = _k.ExternalFunction.__init__
def _rec(self, *a, **kw):
    _init(self, *a, **kw)
    _efs.append({"name": a[0] if a else kw.get("name"), "source": kw.get("source_file"),
                 "flags": list(kw.get("compile_flags") or []), "include": list(kw.get("include_dirs") or [])})
_k.ExternalFunction.__init__ = _rec
from aie.iron.device import from_name
iron.set_current_device(from_name("npu2", n_cols=None))
src = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(src.stem, src)
mod = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(src.parent))
spec.loader.exec_module(mod)
Path(sys.argv[2]).write_text(str(mod.DESIGN.specialize(**mod.SPECIALIZE).as_mlir()))
Path(sys.argv[2] + ".efs.json").write_text(json.dumps(_efs, indent=1))
'''


def dump(out: Path, designs: set[str], specs: list[Path], ok: Path = OK) -> int:
    sys.path.insert(0, str(ok))
    from recipes.families import for_spec
    from recipes.load import load_spec
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for sf in specs:
        spec = load_spec(sf)
        for name, b in sorted(for_spec(spec).builds(spec).items()):
            if b["design"] not in designs:
                continue
            env = dict(os.environ, OPEN_KERNELS_SPEC=str(sf), **b["env"])
            dst = out / f"{sf.stem}__{name}.mlir"
            r = subprocess.run([sys.executable, "-c", GEN, str(ok / "designs" / b["design"]), str(dst)],
                               env=env, cwd=str(ok), capture_output=True, text=True)
            if r.returncode:
                bad += 1
                (out / f"{sf.stem}__{name}.err").write_text(r.stdout + r.stderr)
                print(f"FAILED {sf.stem} {name}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
            else:
                print(f"ok     {sf.stem} {name}")
    return 1 if bad else 0


def _norm(p: Path) -> bytes:
    """A dump file's bytes, with an efs record's tree root cut off its paths (open_kernels/... stays)."""
    if not p.name.endswith(".efs.json"):
        return p.read_bytes()
    import re
    return re.sub(r'"[^"]*?/open_kernels/', '"', p.read_text()).encode()


def diff(a: Path, b: Path) -> int:
    names = sorted({p.name for p in a.glob("*.mlir*")} | {p.name for p in b.glob("*.mlir*")})
    bad = 0
    for n in names:
        pa, pb = a / n, b / n
        if not pa.is_file() or not pb.is_file():
            print(f"MISSING  {n}")
            bad += 1
        elif _norm(pa) != _norm(pb):
            print(f"DIFFERENT {n}")
            bad += 1
        else:
            print(f"same     {n}")
    print(f"{len(names) - bad} of {len(names)} identical")
    return 1 if bad else 0


def ppcheck(dump_dir: Path, old_root: Path, new_root: Path) -> int:
    """Preprocess every recorded kernel source at its recorded flags in both trees and compare tokens."""
    from aie.utils import config
    clang = Path(config.peano_install_dir()) / "bin" / "clang++"
    hdr = Path(config.cxx_header_path())
    pairs = {}
    for f in sorted(dump_dir.glob("*.efs.json")):
        for ef in json.loads(f.read_text()):
            if not ef["source"]:
                continue
            rel = Path(ef["source"]).resolve().relative_to(new_root)
            pairs.setdefault((str(rel), tuple(ef["flags"])), f.name.split("__")[0])
    bad = gen = 0
    for (rel, flags), tag in sorted(pairs.items()):
        toks = []
        for root in (old_root, new_root):
            src = root / rel
            if not src.is_file():
                toks.append(None)
                continue
            inc = [f"-I{root / 'include'}", f"-I{src.parent}", f"-I{root / 'designs' / 'gemv_q4'}",
                   f"-I{root / 'designs' / 'attn'}", f"-I{root / 'designs' / 'ln'}", f"-I{root / 'designs' / 'lin_layer'}"]
            cmd = [str(clang), "-E", "-P", f"-I{hdr}", f"-I{hdr / 'aie_kernels'}", f"-I{hdr / 'aie_kernels' / 'aie2p'}",
                   "-D__AIE_API_AIE_ADF_HPP__", "--target=aie2p-none-unknown-elf", "-std=c++20", "-O2", "-DNDEBUG",
                   "-Wno-macro-redefined", *inc, *flags, str(src), "-o", "-"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            toks.append(r.stdout.split() if r.returncode == 0 else ("ERR", r.stderr[-300:]))
        if toks[0] is None:
            # an export-time generated TU (designs/*/gen_kernels.py): not in a git tree, so not this diff's
            print(f"generated  {tag:18s} {rel}")
            gen += 1
            continue
        same = toks[0] == toks[1]
        bad += not same
        print(f"{'identical' if same else 'DIFFERENT'}  {tag:18s} {rel} {' '.join(f for f in flags if f.startswith('-D'))[:120]}")
    print(f"{len(pairs) - bad - gen} of {len(pairs) - gen} (source, flags) pairs preprocess identically "
          f"({gen} generated at export time, not compared)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("out")
    d.add_argument("--designs", default="dense/dx.py,dense/dx_attn.py")
    d.add_argument("--specs", default="", help="spec stems or paths (default: every recipes/specs/*.json)")
    d.add_argument("--root", default=str(OK), help="the open_kernels tree to generate from (a pristine copy for a baseline)")
    c = sub.add_parser("diff")
    c.add_argument("old")
    c.add_argument("new")
    q = sub.add_parser("ppcheck")
    q.add_argument("dump")
    q.add_argument("old_root")
    q.add_argument("new_root", nargs="?", default=str(OK))
    a = ap.parse_args()
    if a.cmd == "diff":
        return diff(Path(a.old), Path(a.new))
    if a.cmd == "ppcheck":
        return ppcheck(Path(a.dump), Path(a.old_root).resolve(), Path(a.new_root).resolve())
    ok = Path(a.root).resolve()
    sd = ok / "recipes" / "specs"
    specs = [Path(s) if s.endswith(".json") else sd / f"{s}.json" for s in a.specs.split(",") if s] or sorted(sd.glob("*.json"))
    return dump(Path(a.out), set(a.designs.split(",")), specs, ok)


if __name__ == "__main__":
    sys.exit(main())
