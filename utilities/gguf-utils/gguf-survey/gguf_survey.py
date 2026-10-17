# Survey every GGUF repo for a model on Hugging Face by HTTP range request: tensor types per role and key metadata (2026-10-16).
#   python -I gguf_survey.py search "Qwen3-Embedding-0.6B" [--limit 40]
#   python -I gguf_survey.py headers <repo>::<file.gguf> [...]
#   python -I gguf_survey.py files <repo>
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import urllib.request

API = "https://huggingface.co/api/models"
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
              23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
              30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
KEEP_KV = ("general.architecture", "general.name", "general.basename", "general.file_type",
           "general.quantization_version", "general.base_model.0.repo_url", "general.base_model.0.name")
KEEP_SUFFIX = (".pooling_type", ".block_count", ".embedding_length", ".feed_forward_length",
               ".context_length", ".vocab_size", ".attention.head_count", ".attention.head_count_kv",
               ".attention.key_length", ".rope.freq_base", ".attention.layer_norm_rms_epsilon")
KEEP_TOK = ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.bos_token_id", "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_bos_token")


def _token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    p = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
    try:
        return open(p, encoding="utf-8").read().strip()
    except OSError:
        return None


def get_json(url):
    req = urllib.request.Request(url)
    tok = _token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_head(repo, fname, nbytes):
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    cmd = ["curl", "-sL", "-r", f"0-{nbytes - 1}", url]
    tok = _token()
    if tok:
        cmd[1:1] = ["-H", f"Authorization: Bearer {tok}"]
    return subprocess.run(cmd, capture_output=True, timeout=300).stdout


class Short(Exception):
    pass


def parse(buf):
    pos = 0

    def take(n):
        nonlocal pos
        if pos + n > len(buf):
            raise Short()
        b = buf[pos:pos + n]
        pos += n
        return b

    def u32():
        return struct.unpack("<I", take(4))[0]

    def u64():
        return struct.unpack("<Q", take(8))[0]

    def string():
        return take(u64()).decode("utf-8", "replace")

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = u32(), u64()
            if et == 8:
                items = [string() for _ in range(n)]
                return items if n <= 16 else (f"<array str x{n}>", n)
            if et in SCALAR:
                sz = struct.calcsize(SCALAR[et])
                take(sz * n)
                return (f"<array {et} x{n}>", n)
            raise ValueError(f"nested array type {et}")
        return struct.unpack(SCALAR[t], take(struct.calcsize(SCALAR[t])))[0]

    if take(4) != b"GGUF":
        raise ValueError("not a GGUF")
    version = u32()
    n_t, n_kv = u64(), u64()
    kv = {}
    for _ in range(n_kv):
        k = string()
        kv[k] = value(u32())
    tensors = {}
    for _ in range(n_t):
        name = string()
        nd = u32()
        dims = [u64() for _ in range(nd)]
        t = u32()
        u64()
        tensors[name] = (GGML_TYPES.get(t, str(t)), dims)
    return version, kv, tensors


def role_of(name):
    if name.startswith("blk."):
        return name.split(".", 2)[2]
    return name


def report(repo, fname):
    size = 8 << 20
    for _ in range(5):
        try:
            version, kv, tensors = parse(fetch_head(repo, fname, size))
            break
        except Short:
            size *= 2
    else:
        print(f"\n{repo}/{fname}: header longer than {size} bytes")
        return None
    roles = {}
    for name, (t, dims) in tensors.items():
        r = roles.setdefault(role_of(name), [set(), dims])
        r[0].add(t)
    print(f"\n{repo}/{fname}  (GGUF v{version}, {len(tensors)} tensors)")
    for k, v in kv.items():
        if k in KEEP_KV or k in KEEP_TOK or k.endswith(KEEP_SUFFIX):
            print(f"    kv  {k:44s} {v}")
        if k == "tokenizer.ggml.tokens":
            print(f"    kv  {k:44s} {v[1] if isinstance(v, tuple) else len(v)} tokens")
    for role in sorted(roles):
        types, dims = roles[role]
        print(f"    {role:32s} {', '.join(sorted(types)):12s} dims {dims}")
    print(f"    output.weight present: {'output.weight' in tensors}")
    return kv, tensors


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=40)
    h = sub.add_parser("headers")
    h.add_argument("specs", nargs="+")
    f = sub.add_parser("files")
    f.add_argument("repo")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "search":
        repos = get_json(f"{API}?search={urllib.request.quote(a.query)}&filter=gguf&sort=downloads&direction=-1&limit={a.limit}")
        print(f"{len(repos)} GGUF repos match {a.query!r} (by downloads):")
        for r in repos:
            print(f"  {r.get('downloads', 0):>9}  {r['id']}")
    elif a.cmd == "files":
        info = get_json(f"{API}/{a.repo}?blobs=true")
        for sib in info.get("siblings", []):
            if sib["rfilename"].endswith(".gguf"):
                print(f"  {sib.get('size', 0) / 1e6:9.1f} MB  {sib['rfilename']}")
    else:
        for spec in a.specs:
            repo, fname = spec.split("::")
            report(repo, fname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
