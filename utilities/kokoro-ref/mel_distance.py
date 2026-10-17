# Log-mel distance and waveform SNR between WAV files of the same utterance (2026-10-17).
"""    python mel_distance.py ref.wav a.wav [b.wav ...]

80 mel bands (0-12 kHz), 1024-point Hann STFT, hop 256 at 24 kHz; reports the mean and RMS of
|10 log10 mel_a - 10 log10 mel_ref| in dB over frames with energy, and the waveform SNR."""
import sys
import wave

import numpy as np


def load(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    return rate, x


def mel_filters(rate, n_fft, n_mels, fmax):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    pts = mel2hz(np.linspace(hz2mel(0.0), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / rate)
    fb = np.zeros((n_mels, bins.size))
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


def logmel(x, rate):
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    frames = [x[i:i + n_fft] * win for i in range(0, max(len(x) - n_fft, 0) + 1, hop)]
    spec = np.abs(np.fft.rfft(np.array(frames), axis=1)) ** 2
    return 10 * np.log10(spec @ mel_filters(rate, n_fft, 80, 12000.0).T + 1e-10)


def main():
    rate, ref = load(sys.argv[1])
    m_ref = logmel(ref, rate)
    active = m_ref.max(axis=1) > m_ref.max() - 60
    for path in sys.argv[2:]:
        r2, x = load(path)
        n = min(len(x), len(ref))
        m = logmel(x[:n], r2)
        k = min(len(m), len(m_ref))
        d = np.abs(m[:k] - m_ref[:k])[active[:k]]
        snr = 10 * np.log10(np.sum(ref[:n] ** 2) / max(np.sum((x[:n] - ref[:n]) ** 2), 1e-20))
        print(f"{path}: {len(x)} samples vs {len(ref)}; log-mel |diff| mean {d.mean():.3f} dB, "
              f"RMS {np.sqrt((d ** 2).mean()):.3f} dB; waveform SNR {snr:.1f} dB")


if __name__ == "__main__":
    main()
