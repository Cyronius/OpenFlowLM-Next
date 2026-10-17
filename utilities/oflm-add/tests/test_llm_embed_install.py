# Traces: SERVER-EMBED-QWEN3, SERVER-RERANK (canonical spec: specs/server-api/spec.md)
"""oflm-add installs Qwen3-Embedding / Qwen3-Reranker GGUFs as embedding-slot models, not chat models."""
import json
import struct
import sys
from pathlib import Path

import pytest

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402

SYSTEM_LIST = Path(__file__).resolve().parents[3] / "src" / "model_list.json"


def _gguf(path, arch="qwen3"):
    def blob(b):
        return struct.pack("<Q", len(b)) + b

    out = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 1, 1))
    out += blob(b"general.architecture") + struct.pack("<I", 8) + blob(arch.encode())
    out += blob(b"blk.0.attn_q.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 32, 32)
    out += struct.pack("<IQ", 8, 0)                               # Q8_0
    path.write_bytes(bytes(out))


def _repo(tmp_path, name, gguf_name):
    repo = tmp_path / name
    repo.mkdir()
    _gguf(repo / gguf_name)
    (repo / "config.json").write_text(json.dumps({"model_type": "qwen3", "vocab_size": 151669}))
    (repo / "tokenizer.json").write_text("{}")
    (repo / "tokenizer_config.json").write_text("{}")
    (repo / "README.md").write_text("# local test repo\n")
    return repo


def _kernels(tmp_path):
    k = tmp_path / "kernels"
    k.mkdir()
    (k / "manifest.json").write_text(json.dumps({"gguf": {}}))
    return k


def test_names_derive_the_embedding_families_not_the_chat_one():
    assert oflm_add.derive_family({}, "Qwen3-Embedding-0.6B-GGUF") == "qwen3-embedding"
    assert oflm_add.derive_family({}, "Qwen3-Reranker-0.6B-Q8_0-GGUF") == "qwen3-reranker"
    assert oflm_add.derive_family({}, "Qwen3-Reranker-0.6B-i1-GGUF") == "qwen3-reranker"
    assert oflm_add.derive_family({}, "Qwen3-0.6B-GGUF") == "qwen3"
    assert oflm_add.derive_tag("Qwen3-Embedding-0.6B-GGUF") == "qwen3-embedding:0.6b"


def test_without_open_kernels_the_install_says_what_to_pass(tmp_path, monkeypatch):
    if not SYSTEM_LIST.is_file():
        pytest.skip("no src/model_list.json in this checkout")
    repo = _repo(tmp_path, "Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf")
    monkeypatch.setattr("sys.argv", ["oflm-add", str(repo), "--system-list", str(SYSTEM_LIST),
                                     "--config", str(tmp_path / "user_list.json"),
                                     "--models-root", str(tmp_path / "models"), "--quiet"])
    with pytest.raises(SystemExit) as e:
        oflm_add.main()
    assert "--open-kernels" in str(e.value)
    assert not (tmp_path / "user_list.json").exists()


def test_the_embedder_takes_no_config_from_its_pretraining_base(tmp_path, monkeypatch):
    # Qwen's own GGUF README names Qwen3-0.6B-Base, whose config.json says vocabulary 151936.
    if not SYSTEM_LIST.is_file():
        pytest.skip("no src/model_list.json in this checkout")
    repo = tmp_path / "Qwen3-Embedding-0.6B-GGUF"
    repo.mkdir()
    _gguf(repo / "Qwen3-Embedding-0.6B-Q8_0.gguf")
    (repo / "README.md").write_text("---\nbase_model:\n- Qwen/Qwen3-0.6B-Base\n---\n# x\n")
    asked = []

    def fetch(base, f, dest, modelscope=False, force=False):
        asked.append((base, f))
        dest.write_text("{}" if f.endswith(".json") else "")
        return dest

    monkeypatch.setattr(oflm_add, "fetch_from_repo", fetch)
    monkeypatch.setattr(oflm_add, "readme_base_chain", lambda repo_id, modelscope=False, max_depth=3: ["Qwen/Qwen3-0.6B-Base"])
    monkeypatch.setattr("sys.argv", ["oflm-add", str(repo), "--system-list", str(SYSTEM_LIST),
                                     "--config", str(tmp_path / "user_list.json"),
                                     "--models-root", str(tmp_path / "models"),
                                     "--open-kernels", str(_kernels(tmp_path)), "--no-xclbin", "--quiet"])
    oflm_add.main()
    target = tmp_path / "models" / repo.name
    assert ("Qwen/Qwen3-0.6B-Base", "tokenizer.json") in asked
    assert not any(f == "config.json" for _, f in asked)
    assert not (target / "config.json").exists()


@pytest.mark.parametrize("name, gguf, tag, label", [
    ("Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf", ("qwen3-embedding", "0.6b"), "embeddings"),
    ("Qwen3-Reranker-0.6B-Q8_0-GGUF", "qwen3-reranker-0.6b-q8_0.gguf", ("qwen3-reranker", "0.6b"), "rerank"),
])
def test_installs_with_its_label_and_kernels(tmp_path, monkeypatch, name, gguf, tag, label):
    if not SYSTEM_LIST.is_file():
        pytest.skip("no src/model_list.json in this checkout")
    repo = _repo(tmp_path, name, gguf)
    kernels = _kernels(tmp_path)
    # ggml-org's repo name ends in "-Q8_0-GGUF", which derive_tag reads as part of the size, so the tag is given
    monkeypatch.setattr("sys.argv", ["oflm-add", str(repo), "--system-list", str(SYSTEM_LIST),
                                     "--config", str(tmp_path / "user_list.json"),
                                     "--models-root", str(tmp_path / "models"), "--tag", ":".join(tag),
                                     "--open-kernels", str(kernels), "--no-xclbin", "--quiet"])
    oflm_add.main()
    target = tmp_path / "models" / name
    assert (target / "model.gguf").is_file()
    entry = json.loads((tmp_path / "user_list.json").read_text())["models"][tag[0]][tag[1]]
    assert entry["label"] == [label]
    assert entry["details"]["family"] == tag[0] and entry["details"]["weights"] == "gguf"
    assert (target / "open_kernels").resolve() == kernels.resolve()
