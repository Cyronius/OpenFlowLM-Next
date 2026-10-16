"""Print `<spec stem>|<ATTN_* flags>` for every recipes/specs/*.json, the exact flags its
attention cores compile attn.h with (2026-10-16). Feeds utilities/attn_pp_identical.sh --all.

Dense and LFM2 specs import designs/dense/dx.py and read its ATTN_FLAGS; the MoE and
Qwen3.5 specs rebuild designs/layer_x/xlayer.py's list from the recipe's attention geometry.
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


def dense_flags(spec_file: Path) -> list[str]:
    os.environ["OPEN_KERNELS_SPEC"] = str(spec_file)
    src = ROOT / "designs" / "dense" / "dx.py"
    ms = importlib.util.spec_from_file_location(f"dx_{spec_file.stem.replace('-', '_').replace('.', '_')}", src)
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    return list(mod.ATTN_FLAGS)


def main() -> int:
    for sf in sorted((ROOT / "recipes" / "specs").glob("*.json")):
        spec = load_spec(sf)
        try:
            flags = layer_x_flags(spec) if spec.family in ("qwen36moe", "qwen35") else dense_flags(sf)
        except Exception as e:
            print(f"# {sf.stem}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if flags:
            print(f"{sf.stem}|{' '.join(flags)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
