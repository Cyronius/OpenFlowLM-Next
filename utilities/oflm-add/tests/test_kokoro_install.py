# Traces: OPEN-TTS-INSTALL (canonical spec: specs/open-tts/spec.md)
import hashlib
import json
import struct
import sys
import types
from pathlib import Path

import pytest

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402

V10 = {"hidden_dim": 512, "style_dim": 128, "n_layer": 3, "max_dur": 50, "n_token": 178, "n_mels": 80,
       "istftnet": {"upsample_rates": [10, 6], "gen_istft_n_fft": 20, "gen_istft_hop_size": 5},
       "plbert": {"hidden_size": 768}, "vocab": {"a": 43}}


def _safetensors(path, dtype="F32"):
    header = json.dumps({"w": {"dtype": dtype, "shape": [1], "data_offsets": [0, 4 if dtype == "F32" else 2]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * (4 if dtype == "F32" else 2))


def _repo(tmp, name="Kokoro-82M-test", weights="kokoro-v1_0.safetensors", dtype="F32", cfg=V10):
    d = tmp / name
    (d / "voices").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    if weights:
        _safetensors(d / weights, dtype)
    for v in ("af_heart", "bm_george", "jf_alpha", "zf_xiaobei"):
        _safetensors(d / "voices" / f"{v}.safetensors")
    return d


def _args(tmp, **kw):
    a = dict(tag=None, config=str(tmp / "user" / "model_list.json"), models_root=str(tmp / "models"),
             dry_run=False, force=False, no_verify=False, quiet=True, family=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


@pytest.fixture
def fake_lexicons(monkeypatch):
    blobs = {n: f"{{\"{n}\": \"x\"}}".encode() for n in oflm_add.MISAKI_LEXICONS}
    monkeypatch.setattr(oflm_add, "MISAKI_LEXICONS", {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()})

    def fetch(target, force=False, quiet=False):
        (target / "lexicon").mkdir(parents=True, exist_ok=True)
        for n, b in blobs.items():
            (target / "lexicon" / n).write_bytes(b)
        for n, sha in oflm_add.MISAKI_LEXICONS.items():
            assert hashlib.sha256((target / "lexicon" / n).read_bytes()).hexdigest() == sha
        return [f"lexicon/{n}" for n in oflm_add.MISAKI_LEXICONS]

    monkeypatch.setattr(oflm_add, "fetch_misaki_lexicons", fetch)


def test_installs_english_voices_lexicons_and_a_tts_entry(tmp_path, fake_lexicons):
    repo = _repo(tmp_path)
    oflm_add.install_kokoro(_args(tmp_path), str(repo), repo, False, repo.name, {"models": {}})
    target = tmp_path / "models" / repo.name
    assert (target / "kokoro-v1_0.safetensors").is_file()
    assert sorted(p.name for p in (target / "voices").iterdir()) == ["af_heart.safetensors", "bm_george.safetensors"]
    entry = json.loads((tmp_path / "user" / "model_list.json").read_text())["models"]["kokoro"]["82m"]
    assert entry["tts"] is True and entry["url"] == ""
    assert entry["details"]["family"] == "kokoro" and entry["details"]["format"] == "safetensors"
    assert entry["label"] == ["audio", "text-to-speech"]
    assert "voices/af_heart.safetensors" in entry["files"] and "lexicon/us_gold.json" in entry["files"]
    assert not any(f.startswith("voices/jf_") or f.startswith("voices/zf_") for f in entry["files"])


def test_refuses_the_pth_only_repo_with_a_pointer(tmp_path, fake_lexicons):
    repo = _repo(tmp_path, weights=None)
    (repo / "kokoro-v1_0.pth").write_bytes(b"pickle")
    with pytest.raises(SystemExit, match="mlx-community/Kokoro-82M-bf16"):
        oflm_add.install_kokoro(_args(tmp_path), str(repo), repo, False, repo.name, {"models": {}})


def test_refuses_a_gguf_only_repo_naming_gguf(tmp_path, fake_lexicons):
    repo = _repo(tmp_path, weights=None)
    (repo / "kokoro-82m-f16.gguf").write_bytes(b"GGUF")
    with pytest.raises(SystemExit, match="GGUF.*does not read yet"):
        oflm_add.install_kokoro(_args(tmp_path), str(repo), repo, False, repo.name, {"models": {}})


def test_refuses_another_geometry(tmp_path, fake_lexicons):
    repo = _repo(tmp_path, cfg=dict(V10, hidden_dim=256))
    with pytest.raises(SystemExit, match="hidden_dim=256"):
        oflm_add.install_kokoro(_args(tmp_path), str(repo), repo, False, repo.name, {"models": {}})


def test_refuses_a_quantized_checkpoint(tmp_path, fake_lexicons):
    repo = _repo(tmp_path, dtype="I8")
    with pytest.raises(SystemExit, match="F32, F16 or BF16"):
        oflm_add.install_kokoro(_args(tmp_path), str(repo), repo, False, repo.name, {"models": {}})


def test_lexicon_hash_mismatch_is_refused(tmp_path, monkeypatch):
    target = tmp_path / "t"
    (target / "lexicon").mkdir(parents=True)
    for n in oflm_add.MISAKI_LEXICONS:
        (target / "lexicon" / n).write_bytes(b"not misaki")
    with pytest.raises(SystemExit, match="sha256"):
        oflm_add.fetch_misaki_lexicons(target)
