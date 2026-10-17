# Traces: SERVER-SPEECH (canonical spec: specs/server-api/spec.md)
from __future__ import annotations

import io
import os
import sys
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oflm_test import resolve_suites  # noqa: E402
from oflm_test.tasks import SpeechTask  # noqa: E402


def _args(**overrides):
    defaults = dict(llm=False, embedding=False, audio=False, vision=False, tools=False,
                    api=False, tts=False, all=False)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSpeechSuiteSelection(unittest.TestCase):
    def test_tts_alone_runs_only_tts(self):
        suites, _ = resolve_suites(_args(tts=True))
        self.assertEqual([n for n, on in suites.items() if on], ["tts"])

    def test_all_leaves_tts_out(self):
        suites, _ = resolve_suites(_args(all=True))
        self.assertFalse(suites["tts"])
        self.assertTrue(suites["llm"])

    def test_embedding_with_tts_names_tts_in_the_note(self):
        suites, note = resolve_suites(_args(embedding=True, tts=True))
        self.assertEqual([n for n, on in suites.items() if on], ["embedding"])
        self.assertIn("--tts", note)

    def test_args_without_a_tts_attribute_still_resolve(self):
        args = SimpleNamespace(llm=True, embedding=False, audio=False, vision=False,
                               tools=False, api=False, all=False)
        suites, _ = resolve_suites(args)
        self.assertFalse(suites["tts"])


class TestSpeechModelTag(unittest.TestCase):
    def _task(self, model_filter):
        with patch.object(SpeechTask, "get_csv_filename", return_value=os.devnull):
            return SpeechTask("http://127.0.0.1:1/v1", model_filter=model_filter)

    def test_default_tag(self):
        self.assertEqual(self._task(None).model_tag, "kokoro:82m")

    def test_chat_filter_does_not_become_the_speech_model(self):
        self.assertEqual(self._task(["llama3.2:1b"]).model_tag, "kokoro:82m")

    def test_kokoro_tag_in_filter_is_used(self):
        self.assertEqual(self._task(["llama3.2:1b", "kokoro:82m-f32"]).model_tag, "kokoro:82m-f32")


class TestRoundTripVerdict(unittest.TestCase):
    def _task(self):
        with patch.object(SpeechTask, "get_csv_filename", return_value=os.devnull):
            task = SpeechTask("http://127.0.0.1:1/v1")
        task._wav_bytes = b"RIFF0000WAVE"
        return task

    def _http_error(self, code, body):
        return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body))

    def test_no_asr_model_is_a_skip(self):
        body = b'{"error": {"code": "model_not_found", "message": "no speech model"}}'
        with patch("urllib.request.urlopen", side_effect=self._http_error(400, body)):
            verdict, status, _ = self._task()._probe_round_trip()
        self.assertEqual((verdict[0], status), ("SKIPPED", 400))

    def test_server_error_is_a_fail(self):
        with patch("urllib.request.urlopen", side_effect=self._http_error(500, b'{"error": {"code": "internal"}}')):
            verdict, status, _ = self._task()._probe_round_trip()
        self.assertEqual((verdict[0], status), ("FAIL", 500))

    def test_bad_request_without_json_is_a_fail(self):
        with patch("urllib.request.urlopen", side_effect=self._http_error(400, b"bad wav")):
            verdict, _, _ = self._task()._probe_round_trip()
        self.assertEqual(verdict[0], "FAIL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
