"""
Phase 8.1 acceptance test: real eval set + task_accuracy metric.

Gate:
  - 30 probes evaluated under B0 (clean) and B3 (quant_noise→RS recover)
  - accuracy_b0 and accuracy_b3 in [0, 1]
  - result carries .accuracy attribute (not .nll — new metric)
  - does NOT assert B0 > B3 — gate may FAIL honestly (honesty spine)
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.harness.eval_set import (
    EVAL_PROBES,
    EvalResult,
    normalized_match,
    run_task_eval_b0,
    run_task_eval_b3,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@pytest.fixture(scope="module")
def model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
    model.eval()
    yield model, tok
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def b0_result(model_and_tok):
    model, tok = model_and_tok
    return run_task_eval_b0(model, tok, DEVICE)


@pytest.fixture(scope="module")
def b3_result(model_and_tok):
    model, tok = model_and_tok
    return run_task_eval_b3(model, tok, DEVICE, DTYPE, noise_level=0.3)


class TestEvalSet:
    def test_probe_count_is_30(self):
        assert len(EVAL_PROBES) == 30

    def test_probes_have_required_fields(self):
        for p in EVAL_PROBES:
            assert "prompt" in p and isinstance(p["prompt"], str) and p["prompt"]
            assert "expected" in p and isinstance(p["expected"], str) and p["expected"]

    def test_normalized_match_exact(self):
        assert normalized_match("Paris", "Paris")
        assert normalized_match("paris.", "Paris")
        assert normalized_match("Paris is a city", "Paris")

    def test_normalized_match_rejects_wrong(self):
        assert not normalized_match("London", "Paris")
        assert not normalized_match("", "Paris")


class TestB0Eval:
    def test_returns_eval_result(self, b0_result):
        assert isinstance(b0_result, EvalResult)

    def test_accuracy_in_range(self, b0_result):
        assert 0.0 <= b0_result.accuracy <= 1.0

    def test_probe_count(self, b0_result):
        assert len(b0_result.probe_results) == 30

    def test_has_accuracy_not_nll(self, b0_result):
        assert hasattr(b0_result, "accuracy")
        assert not hasattr(b0_result, "nll")

    def test_condition_label(self, b0_result):
        assert b0_result.condition == "B0"

    def test_accuracy_logged(self, b0_result):
        n_correct = sum(r.correct for r in b0_result.probe_results)
        print(f"\nB0 task_accuracy: {b0_result.accuracy:.3f}  ({n_correct}/30 correct)")
        for r in b0_result.probe_results:
            mark = "OK" if r.correct else "WRONG"
            print(f"  [{mark}] expected={r.expected!r} got={r.predicted!r}")


class TestB3Eval:
    def test_returns_eval_result(self, b3_result):
        assert isinstance(b3_result, EvalResult)

    def test_accuracy_in_range(self, b3_result):
        assert 0.0 <= b3_result.accuracy <= 1.0

    def test_probe_count(self, b3_result):
        assert len(b3_result.probe_results) == 30

    def test_condition_label(self, b3_result):
        assert b3_result.condition == "B3"

    def test_accuracy_logged(self, b0_result, b3_result):
        n0 = sum(r.correct for r in b0_result.probe_results)
        n3 = sum(r.correct for r in b3_result.probe_results)
        print(f"\nB0 accuracy: {b0_result.accuracy:.3f}  ({n0}/30)")
        print(f"B3 accuracy: {b3_result.accuracy:.3f}  ({n3}/30)")
        print(f"Delta B0-B3: {b0_result.accuracy - b3_result.accuracy:+.3f}")
        # No assertion on ordering — gate may FAIL honestly
