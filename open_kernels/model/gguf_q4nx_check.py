r"""Hold the engine's GGUF packing to the NumPy packer on a real model (OPEN-GGUF-Q4NX-VIEW).

    src/open_qwen36/out/gguf_pack_test.exe --pack <manifest.json> <model.gguf> <dump dir> 0 3
    python open_kernels/model/gguf_q4nx_check.py <manifest.json> <model.gguf> <dump dir> 0 3

The first writes the pools, consts, lm_head pool and final norm the C++ view packs; this
packs the same through gguf_q4nx.py and recipes/pack.py and compares them byte for byte.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from gguf_q4nx import GgufAsQ4nx  # noqa: E402
from recipes import pack as PK  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("gguf")
    ap.add_argument("dump")
    ap.add_argument("layers", type=int, nargs="+")
    a = ap.parse_args(argv)
    m = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    plan = {"pool_bytes": m["pack"]["pool_bytes"], "chunk_bytes": m["pack"]["chunk_bytes"],
            "layer_types": {lt: d["pack"] for lt, d in m["layer_types"].items()},
            "lm_head": m["pack"]["lm_head"], "embed": m["pack"]["embed"], "norm": m["pack"]["norm"]}
    v = GgufAsQ4nx(a.gguf)
    d = Path(a.dump)
    bad = 0

    def same(what, py, cpp_file):
        nonlocal bad
        cpp = np.fromfile(d / cpp_file, np.uint8)
        py = np.asarray(py, np.uint8).reshape(-1)
        ok = py.size == cpp.size and np.array_equal(py, cpp)
        diff = int((py != cpp).sum()) if py.size == cpp.size else -1
        print(f"{'ok  ' if ok else 'FAIL'} {what}: {py.size} B" + ("" if ok else f", {diff} bytes differ"), flush=True)
        bad += not ok

    for l in a.layers:
        lt = m["layers"][l]
        same(f"layer {l} ({lt}) pool", PK.build_layer_pool(plan, lt, v, l), f"pool_L{l}.bin")
        same(f"layer {l} ({lt}) consts",
             PK.build_consts(plan, lt, v, l, m["layer_types"][lt]["buffers"]["consts"]), f"consts_L{l}.bin")
    same("lm_head pool", PK.build_lmhead_pool(plan, v), "pool_lmhead.bin")
    same("final norm", v.raw(plan["norm"]["tensor"]), "normw.bin")
    print("GGUF-Q4NX-VIEW real model:", "PASS" if not bad else f"{bad} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
