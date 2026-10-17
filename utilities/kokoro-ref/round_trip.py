# Speech round trip: transcribe WAVs through a server's /v1/audio/transcriptions and score the word error rate (2026-10-17).
"""    python round_trip.py --port 52640 --text round_trip.txt --wav engine-{i}.wav [--wav ref-{i}.wav ...]

Each --wav is a pattern with {i} for the 1-based line number of --text."""
import argparse
import json
import re
import sys
import time
import urllib.request
import uuid


def transcribe(port, path, model):
    data = open(path, "rb").read()
    b = "----rt" + uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8")).get("text", "")


def words(t):
    t = t.lower().replace("2024", "twenty twenty four")
    return re.sub(r"[^a-z' ]", " ", t).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--wav", action="append", required=True)
    ap.add_argument("--model", default="whisper-v3:turbo-q8_0")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    lines = [l.strip() for l in open(args.text, encoding="utf-8") if l.strip()]
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/version", timeout=5)
            break
        except Exception:
            time.sleep(5)
    for pattern in args.wav:
        errs = total = 0
        print(f"== {pattern}")
        for i, text in enumerate(lines, 1):
            said = transcribe(args.port, pattern.format(i=i), args.model)
            e, n = wer(text, said)
            errs += e
            total += n
            print(f"  [{i}] {e}/{n} errors: {said.strip()}")
        print(f"  word error rate {errs}/{total} = {errs / total:.3f}")


if __name__ == "__main__":
    main()
