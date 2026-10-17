r"""transformers references for Gemma 3's image path (OPEN-VISION-GEMMA3).

    python open_kernels/model/gemma3_vision_ref.py tiny --fixture DIR [--swap-mlp-names]
    python open_kernels/model/gemma3_vision_ref.py tower --fixture DIR --hf CKPT --image IMG
    python open_kernels/model/gemma3_vision_ref.py logits --out DIR --hf CKPT --image IMG --ids-file IDS

`tiny` builds a small random SigLIP tower and Gemma 3 projector in transformers, writes them as an
mmproj GGUF the way llama.cpp's converter does (F16 matrices, F32 vectors and embeddings, the
soft-embedding norm stored as 1 + w), and writes pixels.bin / tower.bin / ref.bin beside it:
`vit_test --siglip DIR/mmproj.gguf DIR`. `--swap-mlp-names` names fc1 `ffn_down`, as Google's and
the older ggml-org files do. `tower` writes the same fixture from a real checkpoint's tower for a
real mmproj. `logits` runs the whole model on one image and a prompt (bf16) and writes the
next-token logits with Gemma 3's mask (token_type_ids) and with a causal one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def bf16_round(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16).astype(np.uint32).view(np.float32)


def app_pixels(image_path: str, size: int) -> np.ndarray:
    """[3, size, size] as the app hands them over: bilinear resize, (x / 255 - 0.5) * 2, bf16."""
    from PIL import Image
    im = Image.open(image_path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    a = np.asarray(im, np.float32).transpose(2, 0, 1)
    return bf16_round((a * np.float32(1.0 / 255.0) - np.float32(0.5)) * np.float32(2.0))


def build(vc, mm_tokens, text_hidden):
    import torch
    from transformers import Gemma3Config, Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    tc = Gemma3TextConfig(hidden_size=text_hidden, intermediate_size=4 * text_hidden, num_hidden_layers=1,
                          num_attention_heads=1, num_key_value_heads=1, head_dim=text_hidden, vocab_size=8)
    cfg = Gemma3Config(text_config=tc, vision_config=vc, mm_tokens_per_image=mm_tokens)
    with torch.device("cpu"):
        return SiglipVisionModel(vc).eval().float(), Gemma3MultiModalProjector(cfg).eval().float()


def forward(tower, proj, pixels: np.ndarray):
    import torch
    with torch.no_grad():
        x = torch.from_numpy(pixels[None]).float()
        h = tower(pixel_values=x).last_hidden_state
        y = proj(h)
    return h[0].numpy().astype(np.float32), y[0].numpy().astype(np.float32)


def write_mmproj(path: Path, vc, tower, proj, pool: int, swap_mlp_names: bool) -> None:
    import gguf
    sd = {k: v.detach().float().numpy() for k, v in tower.state_dict().items()}
    w = gguf.GGUFWriter(str(path), "clip")
    w.add_string("clip.projector_type", "gemma3")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_uint32("clip.vision.image_size", vc.image_size)
    w.add_uint32("clip.vision.patch_size", vc.patch_size)
    w.add_uint32("clip.vision.embedding_length", vc.hidden_size)
    w.add_uint32("clip.vision.feed_forward_length", vc.intermediate_size)
    w.add_uint32("clip.vision.projection_dim", proj.mm_input_projection_weight.shape[1])
    w.add_uint32("clip.vision.block_count", vc.num_hidden_layers)
    w.add_uint32("clip.vision.attention.head_count", vc.num_attention_heads)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", vc.layer_norm_eps)
    w.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    w.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    w.add_bool("clip.use_gelu", True)
    if pool != 4:
        w.add_uint32("clip.vision.projector.scale_factor", pool)
    p = "vision_model." if "vision_model.embeddings.patch_embedding.weight" in sd else ""
    f16 = lambda a: a.astype(np.float16)
    w.add_tensor("v.patch_embd.weight", sd[p + "embeddings.patch_embedding.weight"].astype(np.float32))
    w.add_tensor("v.patch_embd.bias", sd[p + "embeddings.patch_embedding.bias"].astype(np.float32))
    w.add_tensor("v.position_embd.weight", sd[p + "embeddings.position_embedding.weight"].astype(np.float32))
    fc1, fc2 = ("ffn_down", "ffn_up") if swap_mlp_names else ("ffn_up", "ffn_down")
    for i in range(vc.num_hidden_layers):
        b = f"{p}encoder.layers.{i}."
        g = f"v.blk.{i}."
        for src, dst in (("layer_norm1", "ln1"), ("layer_norm2", "ln2")):
            w.add_tensor(g + dst + ".weight", sd[b + src + ".weight"].astype(np.float32))
            w.add_tensor(g + dst + ".bias", sd[b + src + ".bias"].astype(np.float32))
        for src, dst in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                         ("self_attn.out_proj", "attn_out"), ("mlp.fc1", fc1), ("mlp.fc2", fc2)):
            w.add_tensor(g + dst + ".weight", f16(sd[b + src + ".weight"]))
            w.add_tensor(g + dst + ".bias", sd[b + src + ".bias"].astype(np.float32))
    w.add_tensor("v.post_ln.weight", sd[p + "post_layernorm.weight"].astype(np.float32))
    w.add_tensor("v.post_ln.bias", sd[p + "post_layernorm.bias"].astype(np.float32))
    w.add_tensor("mm.input_projection.weight", f16(proj.mm_input_projection_weight.detach().float().numpy()))
    w.add_tensor("mm.soft_emb_norm.weight", (proj.mm_soft_emb_norm.weight.detach().float().numpy() + 1.0).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def round_f16_matrices(tower, proj) -> None:
    """The weights the GGUF stores as F16, rounded the same way in the oracle, so the fixture measures the port."""
    import torch
    with torch.no_grad():
        for name, prm in tower.named_parameters():
            if prm.dim() == 2 and "position_embedding" not in name:
                prm.copy_(prm.half().float())
        proj.mm_input_projection_weight.copy_(proj.mm_input_projection_weight.half().float())


def cmd_tiny(a) -> int:
    import torch
    from transformers import SiglipVisionConfig
    torch.manual_seed(a.seed)
    vc = SiglipVisionConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
                            image_size=16, patch_size=4, layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh",
                            vision_use_head=False)
    tower, proj = build(vc, mm_tokens=4, text_hidden=24)
    with torch.no_grad():
        for prm in list(tower.parameters()) + list(proj.parameters()):
            prm.normal_(0.0, 0.3)
    round_f16_matrices(tower, proj)
    out = Path(a.fixture)
    out.mkdir(parents=True, exist_ok=True)
    write_mmproj(out / "mmproj.gguf", vc, tower, proj, pool=2, swap_mlp_names=a.swap_mlp_names)
    rng = np.random.default_rng(a.seed)
    px = bf16_round(rng.uniform(-1, 1, (3, vc.image_size, vc.image_size)).astype(np.float32))
    h, y = forward(tower, proj, px)
    px.tofile(out / "pixels.bin")
    h.tofile(out / "tower.bin")
    y.tofile(out / "ref.bin")
    print(f"tiny fixture -> {out} (fc1 named {'ffn_down' if a.swap_mlp_names else 'ffn_up'}): "
          f"tower {h.shape}, projected {y.shape}; vit_test --siglip {out / 'mmproj.gguf'} {out}")
    return 0


def load_real(hf: Path):
    """The checkpoint's tower and projector in fp32, from its own safetensors."""
    import torch
    from safetensors import safe_open
    from transformers import SiglipVisionConfig
    cfg = json.loads((hf / "config.json").read_text())
    vc = SiglipVisionConfig(**cfg["vision_config"])
    tower, proj = build(vc, cfg.get("mm_tokens_per_image", 256), cfg["text_config"]["hidden_size"])
    idx = json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"]
    want_t, want_p = tower.state_dict(), proj.state_dict()
    got_t, got_p = {}, {}
    lead = "vision_model." if any(k.startswith("vision_model.") for k in want_t) else ""
    for name, shard in idx.items():
        for pre, store in (("vision_tower.", got_t), ("model.vision_tower.", got_t),
                           ("multi_modal_projector.", got_p), ("model.multi_modal_projector.", got_p)):
            if name.startswith(pre):
                key = name[len(pre):]
                if store is got_t and not lead and key.startswith("vision_model."):
                    key = key[len("vision_model."):]
                with safe_open(str(hf / shard), framework="pt") as f:
                    store[key] = f.get_tensor(name).float()
    tower.load_state_dict(got_t, strict=True)
    proj.load_state_dict(got_p, strict=True)
    assert set(got_t) == set(want_t) and set(got_p) == set(want_p)
    return vc, tower, proj, cfg


def cmd_tower(a) -> int:
    vc, tower, proj, _ = load_real(Path(a.hf))
    px = app_pixels(a.image, vc.image_size)
    h, y = forward(tower, proj, px)
    out = Path(a.fixture)
    out.mkdir(parents=True, exist_ok=True)
    px.tofile(out / "pixels.bin")
    h.tofile(out / "tower.bin")
    y.tofile(out / "ref.bin")
    print(f"real fixture -> {out}: tower {h.shape}, projected {y.shape} (|y| max {np.abs(y).max():.3f})")
    return 0


def cmd_logits(a) -> int:
    import torch
    from transformers import Gemma3ForConditionalGeneration
    hf = Path(a.hf)
    cfg = json.loads((hf / "config.json").read_text())
    ids = [int(t) for t in Path(a.ids_file).read_text().replace("\n", ",").split(",") if t.strip()]
    image_id = cfg["image_token_index"]
    n_img = sum(1 for t in ids if t == image_id)
    if n_img != cfg.get("mm_tokens_per_image", 256):
        sys.exit(f"{a.ids_file} holds {n_img} image tokens; one image is {cfg.get('mm_tokens_per_image', 256)}")
    model = Gemma3ForConditionalGeneration.from_pretrained(str(hf), dtype=torch.bfloat16).eval()
    px = app_pixels(a.image, cfg["vision_config"]["image_size"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.tensor([ids])
    tt = (x == image_id).long()
    pv = torch.from_numpy(px[None]).to(torch.bfloat16)
    sfx = "_fp32feat" if a.features else ""
    with torch.no_grad():
        if a.features:
            # the fp32 tower's rows instead of the bf16 model's own (the bf16 tower drifts over 27 blocks)
            emb = model.get_input_embeddings()(torch.where(x == image_id, torch.zeros_like(x), x))
            f = torch.from_numpy(np.fromfile(a.features, np.float32).reshape(-1, emb.shape[-1])).to(emb.dtype)
            emb[0, (x[0] == image_id)] = f
            kw0 = {"inputs_embeds": emb}
        else:
            feats = model.model.get_image_features(pv, return_dict=True).pooler_output
            np.asarray(feats[0].float().numpy(), np.float32).tofile(out / "hf_features_bf16.bin")
            kw0 = {"input_ids": x, "pixel_values": pv}
        for tag, kw in (("bidir", {"token_type_ids": tt}), ("causal", {})):
            lg = model(**kw0, **kw).logits[0, -1].float().numpy()
            lg.astype(np.float32).tofile(out / f"hf_logits_{tag}{sfx}.bin")
            top = np.argsort(-lg)[:5]
            print(f"{tag}{sfx}: top-5 {top.tolist()} logits {[round(float(lg[t]), 3) for t in top]}")
    a_, b_ = (np.fromfile(out / f"hf_logits_{t}{sfx}.bin", np.float32) for t in ("bidir", "causal"))
    print(f"bidir vs causal: corr {np.corrcoef(a_, b_)[0, 1]:.6f}, max |diff| {np.abs(a_ - b_).max():.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("--fixture", required=True)
    t.add_argument("--swap-mlp-names", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("tower")
    r.add_argument("--fixture", required=True)
    r.add_argument("--hf", required=True)
    r.add_argument("--image", required=True)
    g = sub.add_parser("logits")
    g.add_argument("--out", required=True)
    g.add_argument("--hf", required=True)
    g.add_argument("--image", required=True)
    g.add_argument("--ids-file", required=True)
    g.add_argument("--features", default=None, help="f32 [256, hidden] rows to inject instead of the model's own tower")
    a = ap.parse_args()
    return {"tiny": cmd_tiny, "tower": cmd_tower, "logits": cmd_logits}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
