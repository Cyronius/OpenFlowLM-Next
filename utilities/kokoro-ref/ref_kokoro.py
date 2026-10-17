# Deterministic PyTorch reference for the open Kokoro engine: per-block dumps and reference audio (2026-10-16).
"""Runs hexgrad's `kokoro` package on the CPU with the harmonic source's noise drawn from the same
splitmix64/Box-Muller stream the C++ engine uses (seed 0 turns it off), and writes every block's
output for one phoneme string, so `kokoro_cli --check` can compare block by block.

    python ref_kokoro.py --pylib <dir with kokoro, misaki> --config config.json \
        --weights kokoro-v1_0.pth|kokoro-v1_0.safetensors --voice af_heart.safetensors \
        --phonemes "..." | --text "..." --seed 1234 --out ref.safetensors [--wav ref.wav]
"""
import argparse
import json
import math
import struct
import sys
import wave

import numpy as np


def import_kokoro(pylib):
    """kokoro's model modules without its package __init__, which imports KPipeline and through it
    misaki's espeak fallback (phonemizer, espeak-ng) that the reference does not use."""
    import os
    import types
    sys.path.insert(0, pylib)
    if "kokoro" not in sys.modules:
        pkg = types.ModuleType("kokoro")
        pkg.__path__ = [os.path.join(pylib, "kokoro")]
        sys.modules["kokoro"] = pkg


def gaussian(seed, n):
    """splitmix64 uniforms, Box-Muller pairs; mirrors open_tts::Gaussian."""
    m = (n + 1) // 2 * 2
    k = np.arange(1, m + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + k * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)
    u1, u2 = u[0::2], u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    th = 2.0 * math.pi * u2
    out = np.empty(m, dtype=np.float64)
    out[0::2] = r * np.cos(th)
    out[1::2] = r * np.sin(th)
    return out[:n].astype(np.float32)


def load_model(args, torch):
    from kokoro.model import KModel
    if args.weights.endswith(".pth"):
        model = KModel(repo_id="hexgrad/Kokoro-82M", config=args.config, model=args.weights)
    else:
        # A flattened state dict: build the modules from the .pth-free path, then load by name.
        from safetensors.torch import load_file
        import tempfile, os
        sd = load_file(args.weights)
        comps = {}
        for k, v in sd.items():
            comp, rest = k.split(".", 1)
            comps.setdefault(comp, {})["module." + rest] = v
        tmp = os.path.join(tempfile.mkdtemp(), "w.pth")
        torch.save(comps, tmp)
        model = KModel(repo_id="hexgrad/Kokoro-82M", config=args.config, model=tmp)
    model.eval()
    if args.round in ("f16", "bf16"):
        dt = torch.float16 if args.round == "f16" else torch.bfloat16
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(dt).float())
    return model


def phonemize(args):
    from misaki import en
    g2p = en.G2P(trf=False, british=args.british, fallback=None, unk="")
    ps, _ = g2p(args.text)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylib", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--phonemes")
    ap.add_argument("--text")
    ap.add_argument("--british", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--round", choices=["none", "f16", "bf16"], default="none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav")
    args = ap.parse_args()
    import_kokoro(args.pylib)
    sys.stdout.reconfigure(encoding="utf-8")

    import torch
    import torch.nn.functional as F
    from safetensors.numpy import load_file as np_load, save_file as np_save

    torch.manual_seed(0)
    torch.set_num_threads(8)
    model = load_model(args, torch)
    ps = args.phonemes if args.phonemes is not None else phonemize(args)
    vocab = model.vocab
    ids = [0] + [vocab[c] for c in ps if c in vocab] + [0]
    (vname, vtab), = np_load(args.voice).items()
    vtab = vtab.reshape(-1, 256)
    style = torch.from_numpy(vtab[min(len(ps) - 1, vtab.shape[0] - 1)].copy())[None, :]
    print(f"phonemes ({len(ps)} chars, {len(ids)} ids): {ps}")

    dump = {"ids": np.array(ids, dtype=np.float32), "style": style.numpy()[0],
            "seed": np.array([args.seed], dtype=np.float64), "speed": np.array([args.speed], dtype=np.float32)}
    with torch.no_grad():
        input_ids = torch.LongTensor([ids])
        T = len(ids)
        text_mask = torch.zeros(1, T, dtype=torch.bool)
        lengths = torch.LongTensor([T])
        bert = model.bert(input_ids, attention_mask=(~text_mask).int())
        dump["bert"] = bert[0].numpy()
        d_en = model.bert_encoder(bert).transpose(-1, -2)
        dump["d_en"] = d_en[0].T.contiguous().numpy()
        s = style[:, 128:]
        d = model.predictor.text_encoder(d_en, s, lengths, text_mask)
        dump["pred.d"] = d[0].contiguous().numpy()
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / args.speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        dump["pred.dur"] = pred_dur.numpy().astype(np.float32)
        dump["pred.dur_float"] = duration[0].numpy()
        indices = torch.repeat_interleave(torch.arange(T), pred_dur)
        aln = torch.zeros((T, indices.shape[0]))
        aln[indices, torch.arange(indices.shape[0])] = 1
        aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln
        F0, N = model.predictor.F0Ntrain(en, s)
        dump["pred.F0"] = F0[0].numpy()
        dump["pred.N"] = N[0].numpy()
        t_en = model.text_encoder(input_ids, lengths, text_mask)
        dump["t_en"] = t_en[0].contiguous().numpy()
        asr = t_en @ aln

        dec = model.decoder
        sd = style[:, :128]
        F0c = dec.F0_conv(F0.unsqueeze(1))
        Nc = dec.N_conv(N.unsqueeze(1))
        xx = torch.cat([asr, F0c, Nc], axis=1)
        xx = dec.encode(xx, sd)
        asr_res = dec.asr_res(asr)
        res = True
        for block in dec.decode:
            if res:
                xx = torch.cat([xx, asr_res, F0c, Nc], axis=1)
            xx = block(xx, sd)
            if block.upsample_type != "none":
                res = False
        dump["dec.x"] = xx[0].contiguous().numpy()

        gen = dec.generator
        f0u = gen.f0_upsamp(F0[:, None]).transpose(1, 2)
        sg = gen.m_source.l_sin_gen
        fn = torch.multiply(f0u, torch.FloatTensor([[range(1, sg.harmonic_num + 2)]]))
        sine = sg._f02sine(fn) * sg.sine_amp
        uv = sg._f02uv(f0u)
        noise_amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
        if args.seed:
            noise = torch.from_numpy(gaussian(args.seed, sine.numel())).reshape(sine.shape)
        else:
            noise = torch.zeros_like(sine)
        sine = sine * uv + noise_amp * noise
        har_source = gen.m_source.l_tanh(gen.m_source.l_linear(sine)).transpose(1, 2).squeeze(1)
        dump["gen.har_source"] = har_source[0].numpy()
        spec, phase = gen.stft.transform(har_source)
        har = torch.cat([spec, phase], dim=1)
        dump["gen.har"] = har[0].contiguous().numpy()
        for i in range(gen.num_upsamples):
            xx = F.leaky_relu(xx, negative_slope=0.1)
            xs_ = gen.noise_res[i](gen.noise_convs[i](har), sd)
            xx = gen.ups[i](xx)
            if i == gen.num_upsamples - 1:
                xx = gen.reflection_pad(xx)
            xx = xx + xs_
            acc = None
            for j in range(gen.num_kernels):
                r = gen.resblocks[i * gen.num_kernels + j](xx, sd)
                acc = r if acc is None else acc + r
            xx = acc / gen.num_kernels
            dump[f"gen.stage{i}"] = xx[0].contiguous().numpy()
        xx = F.leaky_relu(xx)
        post = gen.conv_post(xx)
        dump["gen.conv_post"] = post[0].contiguous().numpy()
        spec2 = torch.exp(post[:, :gen.post_n_fft // 2 + 1, :])
        phase2 = torch.sin(post[:, gen.post_n_fft // 2 + 1:, :])
        audio = gen.stft.inverse(spec2, phase2).squeeze()
        dump["audio"] = audio.numpy()

        if args.seed == 0:
            # Without the noise the package's own forward must give the same audio: the dump is the model.
            torch.manual_seed(0)
            orig = sg.forward
            sg.forward = lambda f0: (lambda sw, u: (sw * u, u, torch.zeros_like(sw)))(sg._f02sine(
                torch.multiply(f0, torch.FloatTensor([[range(1, sg.harmonic_num + 2)]]))) * sg.sine_amp, sg._f02uv(f0))
            full = model(ps, style, args.speed)
            sg.forward = orig
            print(f"package forward vs step-by-step: max |diff| = {float((full - audio).abs().max()):.3g}")

    dump = {k: np.ascontiguousarray(v, dtype=np.float64 if k == "seed" else np.float32) for k, v in dump.items()}
    np_save(dump, args.out, metadata={"phonemes": ps, "seed": str(args.seed), "speed": str(args.speed),
                                       "round": args.round, "voice": args.voice})
    a = dump["audio"]
    print(f"frames {int(dump['pred.dur'].sum())}, samples {a.size} ({a.size / 24000:.2f} s), peak {np.abs(a).max():.3f}")
    if args.wav:
        pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(args.wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)


if __name__ == "__main__":
    main()
