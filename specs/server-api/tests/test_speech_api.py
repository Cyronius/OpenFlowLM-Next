# Traces: SERVER-SPEECH, SERVER-SPEECH-PARAMS, SERVER-SPEECH-VOICES, SERVER-SPEECH-LOAD (canonical spec: specs/server-api/spec.md)
# Needs `oflm serve --tts 1` at OFLM_TEST_BASE_URL (default localhost:52625) with kokoro:82m installed by oflm-add.
import json
import os
import struct
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("OFLM_TEST_BASE_URL", "http://localhost:52625")
TTS_MODEL = os.environ.get("OFLM_TEST_TTS_MODEL", "kokoro:82m")
TIMEOUT = 600
SENTENCE = "The quick brown fox jumps over the lazy dog."


def _speech(**fields):
    body = {"model": TTS_MODEL, "input": SENTENCE, "voice": "af_heart"}
    body.update(fields)
    body = {k: v for k, v in body.items() if v is not None}
    req = urllib.request.Request(f"{BASE}/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def _error(raw):
    try:
        return json.loads(raw).get("error") or {}
    except (ValueError, AttributeError):
        return {}


def _wav(data):
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", data[:16]
    fmt = struct.unpack("<HHIIHH", data[20:36])
    assert data[36:40] == b"data"
    n = struct.unpack("<I", data[40:44])[0]
    return fmt, data[44:44 + n]


def _get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
        return json.loads(r.read())


def test_speaks_a_wav():
    status, ctype, data = _speech()
    assert status == 200, data[:300]
    assert "audio/wav" in ctype
    (tag, channels, rate, _, _, bits), pcm = _wav(data)
    assert (tag, channels, rate, bits) == (1, 1, 24000, 16)
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert len(samples) > 12000, "less than half a second"
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32768
    assert rms > 0.005, f"silent (RMS {rms})"


def test_same_request_same_bytes_and_pcm_is_the_wav_body():
    a, b = _speech()[2], _speech()[2]
    assert a == b
    status, ctype, pcm = _speech(response_format="pcm")
    assert status == 200 and "audio/pcm" in ctype
    assert pcm == a[44:]


def test_speed_shortens():
    slow = len(_speech()[2])
    fast = len(_speech(speed=2.0)[2])
    assert fast <= 0.8 * slow, (fast, slow)


@pytest.mark.parametrize("fields,param,code", [
    ({"input": None}, "input", "missing_required_parameter"),
    ({"input": "   "}, "input", "invalid_value"),
    ({"voice": "no_such_voice"}, "voice", "invalid_value"),
    ({"response_format": "mp3"}, "response_format", "not_implemented"),
    ({"response_format": "opus"}, "response_format", "not_implemented"),
    ({"speed": 5}, "speed", "invalid_value"),
    ({"stream_format": "sse"}, "stream_format", "not_implemented"),
    ({"model": "tts-1"}, "model", "model_not_found"),
    ({"model": "oflm-test-no-such-model:0b"}, "model", "model_not_found"),
])
def test_refusals_name_the_field(fields, param, code):
    status, _, raw = _speech(**fields)
    err = _error(raw)
    assert status == 400, raw[:300]
    assert err.get("param") == param and err.get("code") == code, err
    assert _speech()[0] == 200, "a refusal must leave the server serving"


def test_unknown_voice_lists_the_voices():
    err = _error(_speech(voice="no_such_voice")[2])
    assert "af_heart" in err.get("message", "")


def test_openai_voice_name_resolves():
    status, _, data = _speech(voice="alloy")
    assert status == 200
    assert data == _speech(voice="af_alloy")[2]


def test_listed_in_v1_models_not_in_api_tags():
    assert TTS_MODEL in [m["id"] for m in _get("/v1/models")["data"]]
    assert TTS_MODEL not in [m["name"] for m in _get("/api/tags")["models"]]
