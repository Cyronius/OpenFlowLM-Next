# Traces: OPEN-WHISPER-INSTALL (canonical spec: specs/open-whisper/spec.md)
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402

HANDY = ["whisper-large-v3-turbo-F16.gguf", "whisper-large-v3-turbo-Q4_K_M.gguf",
         "whisper-large-v3-turbo-Q5_K_M.gguf", "whisper-large-v3-turbo-Q6_K.gguf",
         "whisper-large-v3-turbo-Q8_0.gguf", "README.md"]
TOKENS = ["a", "b", "ab", "<|endoftext|>", "<|startoftranscript|>"]
MERGES = ["a b"]


def turbo_kv(**override):
    kv = {"general.architecture": "whisper", "general.repo_url": "https://huggingface.co/openai/whisper-large-v3-turbo",
          "tokenizer.ggml.tokens": TOKENS, "tokenizer.ggml.merges": MERGES, "tokenizer.ggml.eos_token_id": 3,
          "stt.whisper.decoder_start_token_id": 4, "stt.whisper.no_timestamps_token_id": 2,
          "stt.whisper.suppress_tokens": [1], "stt.whisper.begin_suppress_tokens": [0, 3]}
    for key, (value, gkeys) in oflm_add.WHISPER_GEOMETRY.items():
        for g in gkeys:
            kv[g] = value
    kv.update(override)
    return kv


def write_gguf(path, kv):
    def s(x):
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for k, v in kv.items():
        out += s(k)
        if isinstance(v, str):
            out += struct.pack("<I", 8) + s(v)
        elif isinstance(v, list) and v and isinstance(v[0], str):
            out += struct.pack("<IIQ", 9, 8, len(v)) + b"".join(s(x) for x in v)
        elif isinstance(v, list):
            out += struct.pack("<IIQ", 9, 5, len(v)) + b"".join(struct.pack("<i", x) for x in v)
        else:
            out += struct.pack("<II", 4, v)
    Path(path).write_bytes(out)


def write_sidecars(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, (v, _) in oflm_add.WHISPER_GEOMETRY.items()}
    cfg.update(model_type="whisper", bos_token_id=3, eos_token_id=3, torch_dtype="float16")
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "generation_config.json").write_text(json.dumps(
        {"decoder_start_token_id": 4, "no_timestamps_token_id": 2, "eos_token_id": 3, "suppress_tokens": [1],
         "begin_suppress_tokens": [0, 3], "task_to_id": {}}))
    (d / "tokenizer.json").write_text(json.dumps(
        {"model": {"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": MERGES},
         "added_tokens": [{"id": 3, "content": "<|endoftext|>"}, {"id": 4, "content": "<|startoftranscript|>"}]}))
    (d / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>"}))
    return cfg


def test_default_quant_is_q8_0_and_quant_picks_by_name():
    assert oflm_add.choose_whisper_gguf(HANDY)[0] == "whisper-large-v3-turbo-Q8_0.gguf"
    assert oflm_add.choose_whisper_gguf(HANDY, "q4_k_m")[0] == "whisper-large-v3-turbo-Q4_K_M.gguf"
    assert oflm_add.choose_whisper_gguf(HANDY, "F16")[0] == "whisper-large-v3-turbo-F16.gguf"
    chosen, refused = oflm_add.choose_whisper_gguf(HANDY, "Q5_0")
    assert chosen is None and "no Q5_0 file" in refused["--quant"]


def test_gguf_checks_architecture_geometry_and_types():
    q8, q5k, iq2 = 8, 13, 16
    assert oflm_add.check_whisper_gguf(turbo_kv(), [("x", q8, [1280]), ("y", q5k, [1280])]) == []
    assert "IQ2_XXS" in oflm_add.check_whisper_gguf(turbo_kv(), [("x", iq2, [1280])])[0]
    assert "not 'whisper'" in oflm_add.check_whisper_gguf(turbo_kv(**{"general.architecture": "llama"}), [])[0]
    assert "stt.whisper.encoder.n_layers is 24" in oflm_add.check_whisper_gguf(
        turbo_kv(**{"stt.whisper.encoder.n_layers": 24}), [])[0]


def test_sidecars_must_describe_the_gguf(tmp_path):
    write_sidecars(tmp_path)
    assert oflm_add.check_whisper_sidecars(turbo_kv(), tmp_path) == []
    assert "merges" in oflm_add.check_whisper_sidecars(
        turbo_kv(**{"tokenizer.ggml.merges": ["b a"]}), tmp_path)[0]
    assert "vocabulary" in oflm_add.check_whisper_sidecars(
        turbo_kv(**{"tokenizer.ggml.tokens": ["a", "b", "ab", "<|startoftranscript|>", "<|endoftext|>"]}),
        tmp_path)[0]
    assert "decoder_start_token_id" in oflm_add.check_whisper_sidecars(
        turbo_kv(**{"stt.whisper.decoder_start_token_id": 3}), tmp_path)[0]
    assert "suppress_tokens" in oflm_add.check_whisper_sidecars(
        turbo_kv(**{"stt.whisper.suppress_tokens": [2]}), tmp_path)[0]


def test_tokenizer_config_gains_the_ids_the_host_reads(tmp_path):
    write_sidecars(tmp_path)
    assert oflm_add.whisper_tokenizer_config(tmp_path)
    tc = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tc["bos_token_id"] == 3 and tc["eos_token_id"] == [3]
    assert not oflm_add.whisper_tokenizer_config(tmp_path)


def test_kernels_are_found_by_geometry_not_by_name(tmp_path):
    cfg = write_sidecars(tmp_path / "m")
    marker = tmp_path / "xclbins" / "Whisper-V3-Turbo-NPU2" / "open_kernels" / "whisper_kernels.json"
    marker.parent.mkdir(parents=True)
    body = {"format": oflm_add.WHISPER_KERNELS_FORMAT, "complete": True,
            "hf_config_check": {"model_type": "whisper", "d_model": 1280, "vocab_size": 51866}}
    marker.write_text(json.dumps(body))
    assert oflm_add.find_whisper_kernels([tmp_path / "xclbins"], cfg) == marker.parent
    marker.write_text(json.dumps({**body, "complete": False}))
    assert oflm_add.find_whisper_kernels([tmp_path / "xclbins"], cfg) is None
    marker.write_text(json.dumps({**body, "hf_config_check": {"d_model": 1024}}))
    assert oflm_add.find_whisper_kernels([tmp_path / "xclbins"], cfg) is None


def test_repo_url_names_the_base_checkpoint():
    assert oflm_add.hf_repo_from_url("https://huggingface.co/openai/whisper-large-v3-turbo") == \
        "openai/whisper-large-v3-turbo"
    assert oflm_add.hf_repo_from_url("https://example.com/x/y") is None


def test_another_whisper_size_is_refused_before_anything_is_copied(tmp_path):
    import pytest
    src = tmp_path / "whisper-small"
    write_sidecars(src)
    cfg = json.loads((src / "config.json").read_text())
    (src / "config.json").write_text(json.dumps({**cfg, "d_model": 768, "encoder_layers": 12}))
    (src / "model.safetensors").write_bytes(b"\0" * 8)
    args = SimpleNamespace(tag=None, quant=None, config=str(tmp_path / "reg.json"), models_root=str(tmp_path / "m"),
                           xclbin_dir=None, force=False, no_verify=False, quiet=True, dry_run=False)
    with pytest.raises(SystemExit, match="d_model"):
        oflm_add.install_whisper(args, str(src), src, False, "whisper-small", {}, None, {})
    assert not (tmp_path / "m").exists()


def test_local_gguf_install_registers_the_open_engine_entry(tmp_path, monkeypatch):
    src = tmp_path / "whisper-gguf"
    src.mkdir()
    write_gguf(src / "whisper-large-v3-turbo-Q8_0.gguf", turbo_kv())
    base = tmp_path / "base"
    write_sidecars(base)

    def fetch(repo, fname, dest, modelscope=False, force=False):
        assert repo == "openai/whisper-large-v3-turbo"
        dest.write_bytes((base / fname).read_bytes())
        return dest

    monkeypatch.setattr(oflm_add, "fetch_from_repo", fetch)
    monkeypatch.setattr(oflm_add, "find_system_xclbin_root", lambda: None)
    args = SimpleNamespace(tag=None, quant=None, config=str(tmp_path / "reg" / "model_list.json"),
                           models_root=str(tmp_path / "models"), xclbin_dir=str(tmp_path / "x"),
                           force=False, no_verify=False, quiet=True, dry_run=False)
    system = {"model_path": "models", "models": {"whisper-v3": {"turbo": {
        "name": "Whisper-V3-Turbo-NPU2", "url": "u", "files": ["model.q4nx"], "asr": True,
        "oflm_min_version": "0.9.14", "details": {"family": "whisper-v3", "format": "NPU2"}}}}}
    oflm_add.install_whisper(args, str(src), src, False, "whisper-gguf", {}, None, system)

    entry = json.loads(Path(args.config).read_text(encoding="utf-8"))["models"]["whisper-v3"]["turbo-q8_0"]
    assert entry["name"] == "whisper-gguf" and entry["asr"] is True and entry["url"] == ""
    assert entry["details"] == {"family": "whisper-v3", "format": "NPU2", "weights": "gguf",
                                "quantization_level": "Q8_0"}
    assert sorted(entry["files"]) == sorted(oflm_add.WHISPER_SIDECARS + ["model.gguf"])
    installed = tmp_path / "models" / "whisper-gguf"
    assert all((installed / f).is_file() for f in entry["files"])
    assert json.loads((installed / "tokenizer_config.json").read_text())["eos_token_id"] == [3]
