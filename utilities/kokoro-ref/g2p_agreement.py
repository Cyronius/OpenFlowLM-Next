# The C++ phonemizer against misaki 0.9.4 with spaCy, line by line, and the unknown-word rate (2026-10-17).
"""    python g2p_agreement.py --pylib <dir with misaki> --corpus g2p_corpus.txt --cli <kokoro_cli.exe> --model <dir>

misaki runs with its spaCy tagger and no fallback; the C++ runs in skip mode (unknown words
dropped), so both strings drop the same words when they agree on what is unknown."""
import argparse
import difflib
import os
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylib", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--british", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.pylib)
    sys.stdout.reconfigure(encoding="utf-8")
    from misaki import en

    lines = [l.rstrip("\n") for l in open(args.corpus, encoding="utf-8") if l.strip()]
    g2p = en.G2P(trf=False, british=args.british, fallback=None, unk="")
    g2p_q = en.G2P(trf=False, british=args.british, fallback=None, unk="❓")
    env = dict(os.environ, OFLM_TTS_UNKNOWN="skip")
    voice = "bf_emma" if args.british else "af_heart"
    out = subprocess.run([args.cli, "--model", args.model, "--voice", voice, "--g2p-lines", args.corpus],
                         capture_output=True, env=env, check=True).stdout.decode("utf-8").splitlines()
    exact = words_same = words_all = unk_m = unk_c = n_words = 0
    for i, text in enumerate(lines):
        ps_m = g2p(text)[0].strip()
        ps_q, toks = g2p_q(text)
        unknown_m = [t.text for t in toks if t.phonemes is None or "❓" in t.phonemes]
        n_unk_m = ps_q.count("❓")
        ps_c, _, unknown_c = out[i].partition("\t")
        unknown_c = [u for u in unknown_c.split("|") if u]
        n_words += len(re.findall(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*", text))
        unk_m += n_unk_m
        unk_c += len(unknown_c)
        a, b = ps_m.split(), ps_c.split()
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        same = sum(blk.size for blk in sm.get_matching_blocks())
        words_same += same
        words_all += max(len(a), len(b))
        if ps_m == ps_c:
            exact += 1
        else:
            print(f"[{i + 1}] {text}\n   misaki: {ps_m}\n   c++   : {ps_c}")
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    print(f"     {tag}: {' '.join(a[i1:i2])!r} -> {' '.join(b[j1:j2])!r}")
        if unknown_m or unknown_c:
            print(f"   unknown: misaki {unknown_m} | c++ {unknown_c}")
    print(f"\n{len(lines)} lines, {n_words} words: identical lines {exact}/{len(lines)}; "
          f"phoneme words in agreement {words_same}/{words_all} ({100.0 * words_same / max(words_all, 1):.1f}%)")
    print(f"unknown words: misaki {unk_m} ({100.0 * unk_m / n_words:.1f}% of words), c++ {unk_c} "
          f"({100.0 * unk_c / n_words:.1f}%)")


if __name__ == "__main__":
    main()
