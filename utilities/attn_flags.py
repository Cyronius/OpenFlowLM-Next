"""Print `<spec stem>|<ATTN_* flags>` for every recipes/specs/*.json, the exact flags its
attention cores compile attn.h with (2026-10-16). Feeds utilities/attn_pp_identical.sh --all.

Dense and LFM2 specs import designs/dense/dx.py and read its ATTN_FLAGS; the MoE and
Qwen3.5 specs rebuild designs/layer_x/xlayer.py's list from the recipe's attention geometry.
A spec whose dx.py builds differ in attention (Gemma 4's four G4_PART kinds) prints one
`<spec stem>@<build>|<flags>` line per distinct flag set.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "open_kernels"
sys.path.insert(0, str(ROOT))

from recipes.families import for_spec  # noqa: E402
from recipes.load import load_spec  # noqa: E402


def layer_x_flags(spec) -> list[str] | None:
    A = for_spec(spec).recipe(spec).attn
    if A is None:
        return None
    f = [f"-DATTN_NH={A.NH}", f"-DATTN_KVH={A.KVH}", f"-DATTN_HD={A.HD}", f"-DATTN_ROT={A.ROT}", "-DATTN_GATE=1",
         f"-DATTN_VEXP={A.VEXP}", f"-DATTN_NHL={A.NHL}"]
    if A.RB > 1:
        f.append(f"-DATTN_RB={A.RB}")
    if A.BLOCK:
        f.append("-DATTN_BLOCK_ONLY=1")
    return f


def dense_flags(spec_file: Path, env: dict[str, str] | None = None) -> list[str]:
    os.environ["OPEN_KERNELS_SPEC"] = str(spec_file)
    saved = {k: os.environ.get(k) for k in env or {}}
    os.environ.update(env or {})
    try:
        src = ROOT / "designs" / "dense" / "dx.py"
        tag = "_".join([spec_file.stem, *(env or {}).values()]).replace("-", "_").replace(".", "_")
        ms = importlib.util.spec_from_file_location(f"dx_{tag}", src)
        mod = importlib.util.module_from_spec(ms)
        ms.loader.exec_module(mod)
        return list(mod.ATTN_FLAGS)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def dense_sets(spec_file: Path, spec) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for name, b in sorted(for_spec(spec).builds(spec).items()):
        if b["design"] != "dense/dx.py":
            continue
        flags = dense_flags(spec_file, b["env"])
        if all(flags != f for _, f in out):
            out.append((name, flags))
    if len(out) == 1:
        return [(spec_file.stem, out[0][1])]
    return [(f"{spec_file.stem}@{name}", flags) for name, flags in out]


def main() -> int:
    for sf in sorted((ROOT / "recipes" / "specs").glob("*.json")):
        spec = load_spec(sf)
        try:
            if spec.family in ("qwen36moe", "qwen35"):
                sets = [(sf.stem, layer_x_flags(spec))]
            else:
                sets = dense_sets(sf, spec)
        except Exception as e:
            print(f"# {sf.stem}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for tag, flags in sets:
            if flags:
                print(f"{tag}|{' '.join(flags)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
