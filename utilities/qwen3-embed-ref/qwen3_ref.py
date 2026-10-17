# CPU references and NPU checks for Qwen3-Embedding / Qwen3-Reranker (OPEN-HIDDEN-STATE, SERVER-EMBED-QWEN3, SERVER-RERANK) (2026-10-17).
import argparse
import json
import random
import struct
import sys
from pathlib import Path

import numpy as np

EOS = 151643
CARD_TASK = "Given a web search query, retrieve relevant passages that answer the query"
SCI_TASK = "Given a scientific claim, retrieve documents that support or refute the claim"
PROMPT_LENS = [5, 9, 17, 33, 65, 129, 200, 300, 400, 500]
N_PASSAGES, N_QUERIES, N_DOCS, DOC_TOKENS = 200, 50, 200, 128
LEAN_ROWS = 4096
RR_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
             "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CARD_Q = ["What is the capital of China?", "Explain gravity"]
CARD_D = ["The capital of China is Beijing.",
          "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and "
          "is responsible for the movement of planets around the sun."]
GG2HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
         "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
         "ffn_down": "mlp.down_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
         "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
MATMUL = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}


def instruct(task, q):
    return f"Instruct: {task}\nQuery:{q}"


# ---------------------------------------------------------------- the weights the NPU holds
def fp16_rne_flush(x):
    """weight_file.hpp f32_to_fp16: round to nearest even, fp16 subnormals flushed to zero."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    sign = (u >> 16) & 0x8000
    exp = ((u >> 23) & 0xFF) - 127 + 15
    man = u & 0x7FFFFF
    man = man + 0x0FFF + ((man >> 13) & 1)
    carry = (man & 0x800000) != 0
    man = np.where(carry, 0, man)
    exp = np.where(carry, exp + 1, exp)
    h = sign | (np.clip(exp, 0, 31) << 10) | ((man >> 13) & 0x3FF)
    h = np.where(exp <= 0, sign, h)
    h = np.where(exp >= 31, sign | 0x7C00, h)
    return h.astype(np.uint16).view(np.float16).astype(np.float32)


def bf16_rne(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def q4_1_pool(typ, raw_blocks, v):
    """[rows, K] -> what gemv_q4 computes with: bf16(d) * q + bf16(m), d/m/q as std_perm_gguf derives them."""
    rows, K = v.shape
    if typ == "Q4_1":
        b = raw_blocks.reshape(rows, K // 32, 20)
        d = b[..., 0:2].copy().view(np.float16).astype(np.float32)[..., 0]
        m = b[..., 2:4].copy().view(np.float16).astype(np.float32)[..., 0]
        nib = b[..., 4:]
        q = np.concatenate([nib & 0xF, nib >> 4], axis=-1).astype(np.float32)
    elif typ == "Q4_0":
        b = raw_blocks.reshape(rows, K // 32, 18)
        d = b[..., 0:2].copy().view(np.float16).astype(np.float32)[..., 0]
        m = -8.0 * d
        nib = b[..., 2:]
        q = np.concatenate([nib & 0xF, nib >> 4], axis=-1).astype(np.float32)
    else:  # Q8_0 / Q4_K / Q6_K: re-quantized to q4_1 from the exact row (pools.cpp kq branch)
        blk = v.reshape(rows, K // 32, 32).astype(np.float32)
        lo, hi = blk.min(-1), blk.max(-1)
        d = fp16_rne_flush((hi - lo) / np.float32(15.0))
        m = fp16_rne_flush(lo)
        inv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, 1), 0).astype(np.float32)
        t = (blk - m[..., None]) * inv[..., None]
        q = np.clip(np.sign(t) * np.floor(np.abs(t) + 0.5), 0, 15).astype(np.float32)
    w = bf16_rne(d)[..., None] * q + bf16_rne(m)[..., None]
    return w.reshape(rows, K).astype(np.float32)


def load_gguf_into(model, path, prefix):
    """Write the weights the engine holds for this GGUF into `model`, LEAN_ROWS rows at a time (fits a 4B on the host)."""
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize
    params = dict(model.named_parameters())
    kinds, loaded = {}, set()
    with torch.no_grad():
        for t in GGUFReader(str(path)).tensors:
            typ = t.tensor_type.name
            shape = [int(x) for x in reversed(t.shape)]
            if t.name == "token_embd.weight":
                key, how = prefix + "embed_tokens.weight", "exact"
                kinds["token_embd"] = typ
            elif t.name == "output_norm.weight":
                key, how = prefix + "norm.weight", "bf16"
            elif t.name.startswith("blk."):
                _, _l, role, _w = t.name.split(".")
                key = f"{prefix}layers.{_l}.{GG2HF[role]}.weight"
                how = "pool" if role in MATMUL else "bf16"
                if role in MATMUL:
                    kinds.setdefault(role, set()).add(typ)
            else:
                kinds.setdefault("other", set()).add(t.name)
                continue
            dst = params[key]
            if len(shape) == 1:
                v = dequantize(t.data, t.tensor_type).reshape(shape).astype(np.float32)
                dst.copy_(torch.from_numpy(bf16_rne(v) if how == "bf16" else v))
            else:
                raw = np.asarray(t.data).reshape(shape[0], -1)
                for a in range(0, shape[0], LEAN_ROWS):
                    blk = raw[a:a + LEAN_ROWS]
                    v = dequantize(blk, t.tensor_type).reshape(blk.shape[0], shape[1]).astype(np.float32)
                    if how == "pool":
                        v = q4_1_pool(typ, np.asarray(blk, dtype=np.uint8) if typ in ("Q4_1", "Q4_0") else None, v)
                    elif how == "bf16":
                        v = bf16_rne(v)
                    dst[a:a + blk.shape[0]].copy_(torch.from_numpy(np.ascontiguousarray(v)))
            loaded.add(key)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return {"kinds": {k: sorted(v) if isinstance(v, set) else v for k, v in kinds.items()},
            "not_loaded": sorted(k for k in params if k not in loaded and "lm_head" not in k)}


def load_model(a, variant, causal):
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    src = a.reranker if causal else a.embedder
    cls = AutoModelForCausalLM if causal else AutoModel
    if variant == "fp32":
        return cls.from_pretrained(str(src), dtype=torch.float32).eval(), {}
    model = cls.from_config(AutoConfig.from_pretrained(str(src)), dtype=torch.float32).eval()
    return model, load_gguf_into(model, variant.split(":", 1)[1], "model." if causal else "")


def tag_of(variant):
    return "fp32" if variant == "fp32" else Path(variant.split(":", 1)[1]).parent.name


def tok(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), padding_side="left")


# ---------------------------------------------------------------- the fixed text sets
def scifact(sci):
    corpus = [json.loads(l) for l in (sci / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = {q["_id"]: q["text"] for q in map(json.loads, (sci / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
    qrels = {}
    for line in (sci / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        qid, did, s = line.split("\t")
        if int(s) > 0:
            qrels.setdefault(qid, set()).add(did)
    return corpus, queries, qrels


def cmd_texts(a):
    t = tok(a.embedder)
    enc = lambda s: t(s, add_special_tokens=False)["input_ids"] + [EOS]  # noqa: E731
    out = {"eos": EOS, "card": {"texts": [instruct(CARD_TASK, q) for q in CARD_Q] + CARD_D}}
    out["card"]["ids"] = [enc(s) for s in out["card"]["texts"]]
    corpus, queries, qrels = scifact(a.scifact)
    qids = list(qrels)[:N_QUERIES]
    need = sorted({d for q in qids for d in qrels[q]})
    rng = random.Random(20261016)
    others = [c["_id"] for c in corpus if c["_id"] not in set(need)]
    rng.shuffle(others)
    doc_ids = need + others[:N_DOCS - len(need)]
    by_id = {c["_id"]: c for c in corpus}
    docs = [t((by_id[d]["title"] + " " + by_id[d]["text"]).strip(), add_special_tokens=False)["input_ids"][:DOC_TOKENS - 1] + [EOS]
            for d in doc_ids]
    rng.shuffle(order := list(range(len(doc_ids))))
    doc_ids, docs = [doc_ids[i] for i in order], [docs[i] for i in order]
    out["scifact"] = {"qids": qids, "queries": [enc(instruct(SCI_TASK, queries[q])) for q in qids],
                      "doc_ids": doc_ids, "docs": docs, "qrels": {q: sorted(qrels[q]) for q in qids}}
    long = []
    for c in corpus[:60]:
        long += t((c["title"] + " " + c["text"]).strip() + "\n", add_special_tokens=False)["input_ids"]
    out["prompts10"] = [long[i * 37:i * 37 + n - 1] + [EOS] for i, n in enumerate(PROMPT_LENS)]
    a.ref.mkdir(parents=True, exist_ok=True)
    (a.ref / "texts.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"texts: card 4, prompts10 lengths {[len(p) for p in out['prompts10']]}, scifact {len(qids)} queries over {len(docs)} docs")


def texts(a):
    return json.loads((a.ref / "texts.json").read_text(encoding="utf-8"))


def rr_ids(t, instruction, q, d):
    pre = t.encode(RR_PREFIX, add_special_tokens=False)
    suf = t.encode(RR_SUFFIX, add_special_tokens=False)
    body = t(f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}", add_special_tokens=True)["input_ids"]
    return pre + body + suf


def rerank_seqs(a):
    t = tok(a.reranker)
    pairs = [(qi, di) for qi in range(2) for di in range(2)]
    return pairs, [rr_ids(t, CARD_TASK, CARD_Q[qi], CARD_D[di]) for qi, di in pairs]


# ---------------------------------------------------------------- CPU references
def encode(model, seqs, tokens_per_batch=4096):
    import torch
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    res = [None] * len(seqs)
    i = 0
    with torch.inference_mode():
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= tokens_per_batch:
                j += 1
            group = [seqs[k] for k in order[i:j]]
            L = max(map(len, group))
            ids = torch.full((len(group), L), EOS, dtype=torch.long)
            att = torch.zeros((len(group), L), dtype=torch.long)
            for r, s in enumerate(group):
                ids[r, L - len(s):] = torch.tensor(s)
                att[r, L - len(s):] = 1
            pos = (att.cumsum(1) - 1).clamp(min=0)          # every text from position 0, as the engine runs it
            h = model(input_ids=ids, attention_mask=att, position_ids=pos).last_hidden_state[:, -1].float().numpy()
            for r, k in enumerate(order[i:j]):
                res[k] = h[r]
            i = j
    return np.stack(res)


def cmd_embed(a):
    T = texts(a)
    model, info = load_model(a, a.variant, causal=False)
    print(f"[{tag_of(a.variant)}] {json.dumps(info)}", flush=True)
    npz = a.ref / f"embed_{tag_of(a.variant)}.npz"
    res = dict(np.load(npz)) if npz.is_file() else {}
    want = {"card": T["card"]["ids"], "prompts10": T["prompts10"],
            "sci_queries": T["scifact"]["queries"], "sci_docs": T["scifact"]["docs"]}
    for name in a.sets.split(","):
        h = encode(model, want[name])
        res[name + "_hidden"] = h
        res[name] = h / np.linalg.norm(h, axis=1, keepdims=True)
        np.savez(npz, **res)
        print(f"[{tag_of(a.variant)}] {name}: {len(want[name])} texts", flush=True)
    if "card" in res:
        c = res["card"]
        print(f"[{tag_of(a.variant)}] card similarity {np.round(c[:2] @ c[2:].T, 5).tolist()}")


def cmd_rerank(a):
    import torch
    t = tok(a.reranker)
    yes, no = t.convert_tokens_to_ids("yes"), t.convert_tokens_to_ids("no")
    model, info = load_model(a, a.variant, causal=True)
    pairs, seqs = rerank_seqs(a)
    out = []
    with torch.inference_mode():
        for (qi, di), s in zip(pairs, seqs):
            lg = model(input_ids=torch.tensor([s])).logits[0, -1].float()
            ly, ln = float(lg[yes]), float(lg[no])
            p = torch.softmax(torch.tensor([ln, ly]), 0)[1].item()
            out.append({"query": qi, "doc": di, "logit_yes": ly, "logit_no": ln, "score": p})
            print(f"[{tag_of(a.variant)}] q{qi} d{di}: yes {ly:.4f} no {ln:.4f} score {p:.6f} ({len(s)} tokens)", flush=True)
    a.ref.mkdir(parents=True, exist_ok=True)
    (a.ref / f"rerank_{tag_of(a.variant)}.json").write_text(json.dumps({"info": info, "yes": yes, "no": no, "pairs": out}),
                                                            encoding="utf-8")


def ndcg10(scores, doc_ids, qrels):
    vals = []
    for qi, q in enumerate(qrels):
        rel = set(qrels[q])
        top = np.argsort(-scores[qi])[:10]
        dcg = sum(1.0 / np.log2(r + 2) for r, d in enumerate(top) if doc_ids[d] in rel)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(10, len(rel))))
        vals.append(dcg / idcg)
    return float(np.mean(vals))


def cmd_compare(a):
    T = texts(a)
    base = np.load(a.ref / "embed_fp32.npz")
    qrels, doc_ids = T["scifact"]["qrels"], T["scifact"]["doc_ids"]
    bc = base["card"][:2] @ base["card"][2:].T
    have_sci = "sci_docs" in base
    print(f"fp32: card {np.round(bc, 5).tolist()}" +
          (f"; SciFact nDCG@10 {ndcg10(base['sci_queries'] @ base['sci_docs'].T, doc_ids, qrels):.4f}" if have_sci else ""))
    for p in sorted(a.ref.glob("embed_*.npz")):
        if p.stem == "embed_fp32":
            continue
        v = np.load(p)
        line = (f"{p.stem[6:]}: prompts10 cosine vs fp32 min {np.sum(v['prompts10'] * base['prompts10'], axis=1).min():.5f}; "
                f"card max |d| vs fp32 {np.abs(v['card'][:2] @ v['card'][2:].T - bc).max():.4f}")
        if have_sci and "sci_docs" in v:
            cos = np.sum(v["sci_docs"][:N_PASSAGES] * base["sci_docs"][:N_PASSAGES], axis=1)
            line += (f"; 200 passages cosine mean {cos.mean():.5f} min {cos.min():.5f}; "
                     f"SciFact nDCG@10 {ndcg10(v['sci_queries'] @ v['sci_docs'].T, doc_ids, qrels):.4f}")
        print(line)


# ---------------------------------------------------------------- NPU checks (open_qwen36_cli --hidden dumps)
def write_prompts(path, seqs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(",".join(map(str, s)) for s in seqs) + "\n", encoding="utf-8")
    print(f"{len(seqs)} prompts, lengths {[len(s) for s in seqs]} -> {path}")


def cmd_prompts(a):
    T = texts(a)
    write_prompts(a.out / "embed" / "prompts.txt", T["prompts10"] + T["card"]["ids"])
    write_prompts(a.out / "rerank" / "prompts.txt", rerank_seqs(a)[1])


def dumps(d, n):
    return np.stack([np.fromfile(d / f"h_{i}.bin", dtype=np.float32) for i in range(n)])


def cmd_npu_embed(a):
    T = texts(a)
    seqs = T["prompts10"] + T["card"]["ids"]
    u = dumps(a.out, len(seqs))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    card = u[10:12] @ u[12:14].T
    worst = 1.0
    for tag in ("fp32", a.same):
        r = np.load(a.ref / f"embed_{tag}.npz")
        ref = np.concatenate([r["prompts10"], r["card"]])
        cos = np.sum(u * ref, axis=1)
        if tag == a.same:
            worst = cos.min()
        print(f"NPU vs {tag}: per prompt (tokens:cos) " + ", ".join(f"{len(s)}:{c:.5f}" for s, c in zip(seqs, cos)) +
              f"; min {cos.min():.5f}; card max |d| {np.abs(card - ref[10:12] @ ref[12:14].T).max():.4f}")
    print(f"OPEN-HIDDEN-STATE (cosine >= 0.999 against {a.same}): {'PASS' if worst >= 0.999 else 'FAIL'}")
    return 0 if worst >= 0.999 else 1


def cmd_npu_rerank(a):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    pairs, _ = rerank_seqs(a)
    t = tok(a.reranker)
    yes_id, no_id = t.convert_tokens_to_ids("yes"), t.convert_tokens_to_ids("no")
    emb = next(x for x in GGUFReader(str(a.gguf)).tensors if x.name == "token_embd.weight")
    hidden = int(emb.shape[0])
    rows = dequantize(emb.data, emb.tensor_type).reshape(-1, hidden)
    yes, no = rows[yes_id].astype(np.float64), rows[no_id].astype(np.float64)
    refs = {tag: {(p["query"], p["doc"]): p for p in json.loads((a.ref / f"rerank_{tag}.json").read_text(encoding="utf-8"))["pairs"]}
            for tag in ("fp32", a.same) if (a.ref / f"rerank_{tag}.json").is_file()}
    for i, (qi, di) in enumerate(pairs):
        h = np.fromfile(a.out / f"h_{i}.bin", dtype=np.float32).astype(np.float64)
        s = 1.0 / (1.0 + np.exp(h @ no - h @ yes))
        print(f"q{qi} d{di}: NPU score {s:.6f}" + "".join(f" | {tag} {r[(qi, di)]['score']:.6f} (|d| {abs(s - r[(qi, di)]['score']):.4f})"
                                                         for tag, r in refs.items()))
    return 0


# ---------------------------------------------------------------- the spec, from the GGUF header
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_kv(path):
    """GGUF metadata; a float32 scalar comes back as the shortest decimal that is the same f32 (1e-06, not 9.99999997e-07)."""
    with open(path, "rb") as f:
        buf = f.read(64 << 20)
    pos = 0

    def take(n):
        nonlocal pos
        pos += n
        return buf[pos - n:pos]

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
            return [string() for _ in range(n)] if et == 8 else [value(et) for _ in range(n)]
        v = struct.unpack(SCALAR[t], take(struct.calcsize(SCALAR[t])))[0]
        return float(str(np.float32(v))) if t == 6 else v

    if take(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    u32()
    u64()
    kv = {}
    for _ in range(u64()):
        k = string()
        kv[k] = value(u32())
    return kv


def cmd_spec(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "open_kernels"))
    from recipes.spec import ModelSpec
    s = ModelSpec.from_gguf_metadata(gguf_kv(a.gguf))
    d = s.to_dict()
    d["extra"] = {"architecture": "qwen3", "source": "gguf", "model": a.name}
    spec = ModelSpec.from_dict(d)
    print(f"{a.gguf}: vocab {spec.vocab}, norm_eps {spec.norm_eps!r} -> {spec.spec_hash()}")
    if a.write:
        Path(a.write).write_text(spec.to_json(), encoding="utf-8", newline="\n")
        print(f"wrote {a.write}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", type=Path, help="the official Qwen3-Embedding safetensors dir")
    p.add_argument("--reranker", type=Path, help="the official Qwen3-Reranker safetensors dir")
    p.add_argument("--ref", type=Path, help="where the reference texts and vectors live")
    p.add_argument("--scifact", type=Path, help="BEIR SciFact dir (corpus.jsonl, queries.jsonl, qrels/test.tsv)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("texts", help="write <ref>/texts.json: the card, ten fixed prompts, a SciFact subset")
    for name in ("embed", "rerank"):
        s = sub.add_parser(name, help=f"CPU {name} reference; variant fp32 or gguf:<file> (the weights as the engine holds them)")
        s.add_argument("variant")
        if name == "embed":
            s.add_argument("--sets", default="card,prompts10,sci_queries,sci_docs")
    sub.add_parser("compare", help="every GGUF reference in <ref> against fp32")
    s = sub.add_parser("prompts", help="write the CLI's prompt files under <out>/embed and <out>/rerank")
    s.add_argument("out", type=Path)
    for name in ("npu-embed", "npu-rerank"):
        s = sub.add_parser(name, help="score the h_<i>.bin files open_qwen36_cli --hidden <dir>/h wrote")
        s.add_argument("out", type=Path)
        s.add_argument("--same", required=True, help="the reference tag of the same GGUF (its parent dir name)")
        if name == "npu-rerank":
            s.add_argument("--gguf", type=Path, required=True, help="the reranker GGUF (for its yes / no rows)")
    s = sub.add_parser("spec", help="the ModelSpec a GGUF derives, epsilon read as float32")
    s.add_argument("gguf", type=Path)
    s.add_argument("--name", default="Qwen3-Embedding-0.6B-NPU2")
    s.add_argument("--write", help="write the spec JSON here")
    a = p.parse_args(argv)
    return {"texts": cmd_texts, "embed": cmd_embed, "rerank": cmd_rerank, "compare": cmd_compare, "prompts": cmd_prompts,
            "npu-embed": cmd_npu_embed, "npu-rerank": cmd_npu_rerank, "spec": cmd_spec}[a.cmd](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
