# Expected phonemes for g2p_test.cpp: misaki 0.9.4 (spaCy tagger, no fallback) over a small lexicon excerpt (2026-10-16).
"""Writes g2p_gold.json and g2p_silver.json (the entries of misaki's us_gold/us_silver.json the cases
need; misaki is Apache-2.0) and g2p_cases.json (text -> misaki's phonemes with that excerpt).

    python gen_g2p_fixtures.py --pylib <dir holding misaki> --out <testdata dir>
"""
import argparse
import json
import os
import re
import sys

CASES = [
    "Hello, world!",
    "The apple and the cat.",
    "I walked to the park.",
    "She walks to work; walking is good.",
    "In 1984, I had 3 cats.",
    "It costs $5.50, or 50% more.",
    "The 21st century began in 2001.",
    "Pi is about 3.14.",
    "NASA and the U.S. agreed.",
    "Zorblax is unknown.",
    "A well-known fact.",
    "Don't stop—keep going...",
    "We met at 10 and left at 11:30.",
    "He paid £20 for 2 books (used).",
]

NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred "
                "thousand million billion point minus and first second third fourth fifth eleventh twelfth "
                "twentieth twenty-first dollar dollars cent cents pound pounds pence euro euros percent oh O").split()


def keys_for(word):
    out = {word, word.lower(), word.capitalize(), word.upper()}
    w = word.lower()
    for suf in ("s", "es", "ed", "d", "ing", "'s"):
        if w.endswith(suf):
            out.update({w[: -len(suf)], w[: -len(suf)] + "e"})
    if w.endswith("ies"):
        out.add(w[:-3] + "y")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylib", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.pylib)
    sys.stdout.reconfigure(encoding="utf-8")
    from misaki import en

    g2p = en.G2P(trf=False, british=False, fallback=None, unk="")
    full_gold, full_silver = g2p.lexicon.golds, g2p.lexicon.silvers
    import importlib.resources
    from misaki import data
    raw_gold = json.load(importlib.resources.open_text(data, "us_gold.json"))
    raw_silver = json.load(importlib.resources.open_text(data, "us_silver.json"))
    want = set(NUMBER_WORDS) | {chr(c) for c in range(65, 91)}
    for text in CASES:
        for tok in re.findall(r"[A-Za-z0-9'.\-]+", text):
            for part in re.split(r"[-.]", tok):
                if part:
                    want |= keys_for(part)
            want |= keys_for(tok)
    want |= {"am", "an", "to", "the", "in", "a", "I", "used", "by", "versus", "don't", "he's"}
    gold = {k: v for k, v in raw_gold.items() if k in want}
    silver = {k: v for k, v in raw_silver.items() if k in want and k not in gold}
    g2p.lexicon.golds = en.Lexicon.grow_dictionary(gold)
    g2p.lexicon.silvers = en.Lexicon.grow_dictionary(silver)
    cases = []
    for text in CASES:
        ps, _ = g2p(text)
        cases.append({"text": text, "phonemes": ps.strip()})
        print(f"{text!r} -> {ps.strip()!r}")
    with open(os.path.join(args.out, "g2p_gold.json"), "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=0, sort_keys=True)
    with open(os.path.join(args.out, "g2p_silver.json"), "w", encoding="utf-8") as f:
        json.dump(silver, f, ensure_ascii=False, indent=0, sort_keys=True)
    spell = {w: g2p.lexicon.get_NNP(w)[0] for w in ("Zorblax", "Kubernetes")}
    with open(os.path.join(args.out, "g2p_cases.json"), "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "spell": spell}, f, ensure_ascii=False, indent=1)
    print(f"{len(gold)} gold and {len(silver)} silver entries, {len(cases)} cases")


if __name__ == "__main__":
    main()
