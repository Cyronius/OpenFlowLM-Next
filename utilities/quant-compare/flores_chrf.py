"""Translation quality of a served model on FLORES-200 devtest, scored with chrF (2026-10-16); see README.md."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# Hy-MT2 model card: a Chinese prompt with a Chinese language name for Chinese, English otherwise
PROMPTS = {
    "zho_Hans": "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}",
    "deu_Latn": "Translate the following text into German. Note that you should only output the translated "
                "result without any additional explanation:\n\n{src}",
    "eng_Latn": "Translate the following text into English. Note that you should only output the translated "
                "result without any additional explanation:\n\n{src}",
}


def lines(flores: Path, lang: str, n: int) -> list[str]:
    return (flores / f"{lang}.devtest").read_text(encoding="utf-8").splitlines()[:n]


def chat(url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "temperature": 0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0, "repeat_penalty": 1.0,
            "max_tokens": max_tokens}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def gen(a) -> int:
    flores = Path(a.flores)
    out = Path(a.out)
    done = {}
    if out.is_file():
        for ln in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            done[(r["pair"], r["i"])] = r
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for pair in a.pairs.split(","):
            s, t = pair.split("-")
            for i, src in enumerate(lines(flores, s, a.n)):
                if (pair, i) in done:
                    continue
                hyp = chat(a.url, a.model, PROMPTS[t].format(src=src), a.max_tokens).strip()
                f.write(json.dumps({"pair": pair, "i": i, "hyp": hyp}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0:
                    print(f"{pair} {i}: {time.time() - t0:.0f} s", flush=True)
    print(f"done in {time.time() - t0:.0f} s")
    return 0


def load(p: str) -> dict:
    out: dict = {}
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        out.setdefault(r["pair"], {})[r["i"]] = r["hyp"]
    return out


def score(a) -> int:
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    flores = Path(a.flores)
    runs = {"a": load(a.a)} | ({"b": load(a.b)} if a.b else {})
    for pair in sorted(runs["a"]):
        t = pair.split("-")[1]
        idx = sorted(set.intersection(*(set(r.get(pair, {})) for r in runs.values())))
        n = len(idx)
        allref = lines(flores, t, max(idx) + 1)
        refs = [allref[i] for i in idx]
        row = [f"{pair} (n={n})"]
        for k, run in runs.items():
            hyp = [run[pair][i] for i in idx]
            row.append(f"{k} chrF {chrf.corpus_score(hyp, [refs]).score:.2f}")
        if a.b:
            ha = [runs["a"][pair][i] for i in idx]
            hb = [runs["b"][pair][i] for i in idx]
            same = sum(x == y for x, y in zip(ha, hb))
            row.append(f"identical {same}/{n}")
            row.append(f"a vs b chrF {chrf.corpus_score(ha, [hb]).score:.2f}")
            # paired bootstrap over sentences: how far a - b could move on another draw of this size
            import random
            rng = random.Random(0)
            d = []
            for _ in range(a.bootstrap):
                s = [rng.randrange(n) for _ in range(n)]
                r = [[refs[i] for i in s]]
                d.append(chrf.corpus_score([ha[i] for i in s], r).score - chrf.corpus_score([hb[i] for i in s], r).score)
            d.sort()
            if d:
                row.append(f"a - b 95% [{d[int(0.025 * len(d))]:+.2f}, {d[int(0.975 * len(d)) - 1]:+.2f}]")
        print("  ".join(row))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--url", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--flores", required=True, help="FLORES-200 devtest directory (<lang>.devtest files)")
    g.add_argument("--pairs", default="eng_Latn-zho_Hans,eng_Latn-deu_Latn")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--flores", required=True)
    s.add_argument("--a", required=True)
    s.add_argument("--b")
    s.add_argument("--bootstrap", type=int, default=1000, help="paired bootstrap resamples for a - b (0: none)")
    a = ap.parse_args()
    return gen(a) if a.cmd == "gen" else score(a)


if __name__ == "__main__":
    sys.exit(main())
