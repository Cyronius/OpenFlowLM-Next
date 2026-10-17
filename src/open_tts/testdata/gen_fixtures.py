# Small random-weight Kokoro blocks, run by hexgrad's own modules, for ops_test.cpp (2026-10-16).
"""Writes blocks.safetensors: each block's state dict, its input and its output, from kokoro 0.9.4's
modules with torch.manual_seed(0). No model weights are involved.

    python gen_fixtures.py --pylib <dir holding kokoro> --out blocks.safetensors
"""
import argparse
import importlib.util
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylib", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location(
        "ref_kokoro", os.path.join(os.path.dirname(__file__), "..", "..", "..", "utilities", "kokoro-ref", "ref_kokoro.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    ref.import_kokoro(args.pylib)
    import torch
    import torch.nn as nn
    from torch.nn.utils import weight_norm
    from safetensors.numpy import save_file
    from kokoro.istftnet import AdainResBlk1d, AdaINResBlock1, TorchSTFT, SourceModuleHnNSF
    from kokoro.modules import LayerNorm, AdaLayerNorm

    torch.manual_seed(0)
    out = {}

    def put_state(name, module):
        for k, v in module.state_dict().items():
            out[f"{name}.{k}"] = v.detach().float().numpy().copy()

    def randomize(module, scale=0.5):
        with torch.no_grad():
            for p in module.parameters():
                p.copy_(torch.randn_like(p) * scale)
            for name, p in module.named_parameters():
                if name.endswith("weight_g") or "alpha" in name:
                    p.copy_(p.abs() + 0.5)

    with torch.no_grad():
        s = torch.randn(1, 5)
        out["style"] = s[0].numpy().copy()
        for name, block, cin, t in [("rb_up", AdainResBlk1d(6, 4, 5, upsample=True), 6, 7),
                                    ("rb", AdainResBlk1d(4, 4, 5), 4, 9)]:
            randomize(block)
            block.eval()
            x = torch.randn(1, cin, t)
            put_state(name, block)
            out[f"{name}.in"] = x[0].numpy().copy()
            out[f"{name}.out"] = block(x, s)[0].numpy().copy()

        rb1 = AdaINResBlock1(4, 7, [1, 3, 5], 5)
        randomize(rb1)
        x = torch.randn(1, 4, 30)
        put_state("rb1", rb1)
        out["rb1.in"] = x[0].numpy().copy()
        out["rb1.out"] = rb1(x, s)[0].numpy().copy()

        lstm = nn.LSTM(6, 3, 1, batch_first=True, bidirectional=True)
        randomize(lstm)
        x = torch.randn(1, 9, 6)
        put_state("lstm", lstm)
        out["lstm.in"] = x[0].numpy().copy()
        out["lstm.out"] = lstm(x)[0][0].numpy().copy()

        ups = weight_norm(nn.ConvTranspose1d(4, 3, 20, 10, padding=5))
        randomize(ups)
        x = torch.randn(1, 4, 5)
        put_state("ups", ups)
        out["ups.in"] = x[0].numpy().copy()
        out["ups.out"] = ups(x)[0].numpy().copy()

        nconv = nn.Conv1d(22, 4, kernel_size=12, stride=6, padding=3)
        randomize(nconv)
        x = torch.randn(1, 22, 61)
        put_state("nconv", nconv)
        out["nconv.in"] = x[0].numpy().copy()
        out["nconv.out"] = nconv(x)[0].numpy().copy()

        ln = LayerNorm(5)
        randomize(ln)
        x = torch.randn(1, 5, 7)
        put_state("ln", ln)
        out["ln.in"] = x[0].numpy().copy()
        out["ln.out"] = ln(x)[0].numpy().copy()

        aln = AdaLayerNorm(5, 6)
        randomize(aln)
        x = torch.randn(1, 6, 7)   # [B, C, T], the way DurationEncoder hands it over
        put_state("aln", aln)
        out["aln.in"] = x[0].T.contiguous().numpy().copy()
        out["aln.out"] = aln(x.transpose(-1, -2), s)[0].numpy().copy()

        stft = TorchSTFT(20, 5, 20)
        x = torch.randn(1, 600) * 0.3
        mag, ph = stft.transform(x)
        out["stft.in"] = x[0].numpy().copy()
        out["stft.mag"] = mag[0].numpy().copy()
        out["stft.phase"] = ph[0].numpy().copy()
        m2 = torch.exp(torch.randn(1, 11, 121) * 0.5)
        p2 = torch.sin(torch.randn(1, 11, 121) * 2)
        out["istft.mag"] = m2[0].numpy().copy()
        out["istft.phase"] = p2[0].numpy().copy()
        out["istft.out"] = stft.inverse(m2, p2)[0, 0].numpy().copy()

        src = SourceModuleHnNSF(24000, 300, harmonic_num=8, voiced_threshod=10)
        randomize(src)
        put_state("src", src)
        f0 = torch.tensor([[0.0, 5.0, 120.0, 180.5, 233.25, 260.0, 9.5, 301.0]])
        out["src.f0"] = f0[0].numpy().copy()
        sg = src.l_sin_gen
        f0u = nn.Upsample(scale_factor=300)(f0[:, None]).transpose(1, 2)
        for seed in (0, 7):
            fn = torch.multiply(f0u, torch.FloatTensor([[range(1, sg.harmonic_num + 2)]]))
            sine = sg._f02sine(fn) * sg.sine_amp
            uv = sg._f02uv(f0u)
            amp = uv * sg.noise_std + (1 - uv) * sg.sine_amp / 3
            noise = torch.from_numpy(ref.gaussian(seed, sine.numel())).reshape(sine.shape) if seed else 0 * sine
            sine = sine * uv + amp * noise
            out[f"src.out{seed}"] = src.l_tanh(src.l_linear(sine))[0, :, 0].numpy().copy()
        out["gauss7"] = ref.gaussian(7, 11)

        v = torch.linspace(-6, 6, 97)
        out["gelu.in"] = v.numpy().copy()
        out["gelu.out"] = (0.5 * v * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (v + 0.044715 * torch.pow(v, 3.0))))).numpy()

    save_file({k: np.ascontiguousarray(a, dtype=np.float32) for k, a in out.items()}, args.out)
    print(f"wrote {len(out)} tensors to {args.out}")


if __name__ == "__main__":
    main()
