"""Unit tests for RerankTask, checks R1-R8, against a fake /v1/rerank: no OFLM server needed.

Run with:
    python3 -m pytest tests/test_rerank_checks.py -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oflm_test.tasks import RerankTask  # noqa: E402

MODEL = "qwen3-reranker:0.6b"


class FakeReranker:
    """A /v1/rerank that scores by shared words; each flag breaks one thing a check should catch."""

    def __init__(self, request_order=False, unsorted=False, noisy=False, stateful=False,
                 any_model=False, lax=False, ignore_top_n=False, wrong_name=False):
        self.flags = dict(request_order=request_order, unsorted=unsorted, noisy=noisy, stateful=stateful,
                          any_model=any_model, lax=lax, ignore_top_n=ignore_top_n, wrong_name=wrong_name)
        self.calls = 0

    @staticmethod
    def _error(param):
        body = {"error": {"message": f"bad {param}", "type": "invalid_request_error", "param": param,
                          "code": "invalid_value"}}
        return 400, body, json.dumps(body)

    def __call__(self, path, body, **_):
        f = self.flags
        self.calls += 1
        assert path == "/rerank"
        if not isinstance(body.get("query"), str) and not f["lax"]:
            return self._error("query")
        if not isinstance(body.get("documents"), list) and not f["lax"]:
            return self._error("documents")
        if body.get("model") not in (None, MODEL) and not f["any_model"]:
            err = {"error": {"message": "not loaded", "type": "invalid_request_error", "param": "model",
                             "code": "model_not_found"}}
            return 400, err, json.dumps(err)
        query = set(str(body.get("query", "")).lower().replace("?", "").split())
        docs = body.get("documents") if isinstance(body.get("documents"), list) else []
        scores, carry = [], 0.0
        for d in docs:
            words = set(d.lower().replace(".", "").split())
            s = len(query & words) / (len(query) + 1)
            if f["stateful"]:
                s, carry = min(1.0, s + carry), 0.3
            if f["noisy"]:
                s = min(1.0, s + 0.01 * (self.calls % 3))
            scores.append(s)
        order = list(range(len(docs)))
        if not f["request_order"]:
            order.sort(key=lambda i: -scores[i])
        if f["unsorted"]:
            order = sorted(order, key=lambda i: scores[i])
        top_n = body.get("top_n")
        if top_n and not f["ignore_top_n"]:
            order = order[:top_n]
        results = []
        for i in order:
            r = {"index": i, "relevance_score": scores[i]}
            if body.get("return_documents"):
                r["document"] = {"text": docs[i]}
            results.append(r)
        out = {"object": "list", "model": "someone-else" if f["wrong_name"] else MODEL, "results": results,
               "usage": {"prompt_tokens": 10, "total_tokens": 10}}
        return 200, out, json.dumps(out)


def _task(fake):
    with patch.object(RerankTask, "_get_oflm_version", return_value="0.9.99"), \
         patch.object(RerankTask, "_fetch_all_models", return_value=[MODEL, "llama3.2:1b"]), \
         patch("os.makedirs"):
        task = RerankTask(base_url="http://127.0.0.1:52625/v1", backend_os="linux")
    task._post_json = fake
    return task


def _verdict(task, name):
    check = {
        "R1": task._check_response_structure, "R2": task._check_relevance_ordering, "R3": task._check_sorted,
        "R4": task._check_repeatability, "R5": task._check_document_independence, "R6": task._check_top_n,
        "R7": task._check_model_identity, "R8": task._check_request_validation,
    }[name]
    return check(MODEL)[0]


class TestRerankChecks(unittest.TestCase):

    def test_only_rerankers_are_tested(self):
        self.assertEqual(_task(FakeReranker()).models, [MODEL])

    def test_eight_checks_named_in_order(self):
        for number, name in enumerate(RerankTask.CHECK_NAMES, 1):
            self.assertTrue(name.startswith(f"R{number} "), name)
        self.assertEqual(len(RerankTask.CHECK_NAMES), 8)

    def test_a_correct_server_passes_every_check(self):
        task = _task(FakeReranker())
        for n in range(1, 9):
            with self.subTest(check=f"R{n}"):
                self.assertEqual(_verdict(task, f"R{n}"), "PASS")

    def test_each_defect_fails_its_check(self):
        cases = {
            "R2": dict(request_order=True), "R3": dict(unsorted=True), "R4": dict(noisy=True),
            "R5": dict(stateful=True), "R6": dict(ignore_top_n=True), "R7": dict(any_model=True),
            "R8": dict(lax=True),
        }
        for check, flags in cases.items():
            with self.subTest(check=check, flags=flags):
                self.assertEqual(_verdict(_task(FakeReranker(**flags)), check), "FAIL")

    def test_a_wrong_model_name_is_a_soft_fail(self):
        self.assertEqual(_verdict(_task(FakeReranker(wrong_name=True)), "R7"), "SOFT-FAIL")


if __name__ == "__main__":
    unittest.main()
