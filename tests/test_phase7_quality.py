"""
Phase 7.4 — acceptance test: real quality metric + R-D gate.

Tests assert verdict LINE EXISTS (never ==PASS per honesty gate).
Gate is allowed to FAIL honestly. Rigged PASS = fatal (S9).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.harness.phase7_quality import run_phase7_quality

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@pytest.fixture(scope="module")
def quality_result():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
    model.eval()
    result = run_phase7_quality(model, tok, DEVICE, DTYPE)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result


class TestQualityMetric:
    def test_four_baselines_present(self, quality_result):
        names = [b.name for b in quality_result.baselines]
        assert "B0_no_protection" in names
        assert "B1_all_resident" in names
        assert "B2_erasure_parity" in names
        assert "B3_full_AEPK" in names

    def test_all_nll_finite(self, quality_result):
        import math
        for b in quality_result.baselines:
            assert math.isfinite(b.nll), f"{b.name} NLL not finite: {b.nll}"

    def test_b0_b1_equal_nll(self, quality_result):
        b0 = next(b for b in quality_result.baselines if b.name == "B0_no_protection")
        b1 = next(b for b in quality_result.baselines if b.name == "B1_all_resident")
        assert b0.nll == b1.nll, "B0 and B1 must be identical (both clean KV)"

    def test_storage_bits_positive(self, quality_result):
        for b in quality_result.baselines:
            assert b.storage_bits > 0, f"{b.name} storage_bits must be positive"

    def test_b3_storage_nonzero(self, quality_result):
        b3 = next(b for b in quality_result.baselines if b.name == "B3_full_AEPK")
        # B3 residency plan moves pages to CODED/EVICTED, may use less storage than B0.
        # The parity overhead is accounted in storage_b3 via plan.total_storage_bits +
        # parity_pages_bits. Assert it's nonzero and finite.
        assert b3.storage_bits > 0, "B3 storage must be positive (residency + parity)"
        assert b3.residual_mse >= 0.0, "B3 residual MSE must be non-negative"


class TestTaskProbe:
    def test_b0_task_probe_ran(self, quality_result):
        """Task probe ran and returned a boolean (correctness doesn't matter for gate)."""
        assert isinstance(quality_result.task_probe_correct_b0, bool)

    def test_b3_task_probe_ran(self, quality_result):
        assert isinstance(quality_result.task_probe_correct_b3, bool)


class TestRDGate:
    def test_gate_verdict_line_exists(self, quality_result):
        """Honesty gate: assert the verdict LINE exists, never ==PASS."""
        assert quality_result.gate_verdict in ("PASS", "FAIL"), (
            f"gate_verdict must be 'PASS' or 'FAIL', got: {quality_result.gate_verdict!r}"
        )

    def test_report_has_verdict(self, quality_result):
        verdict_lines = [l for l in quality_result.report_lines if "GATE VERDICT" in l]
        assert len(verdict_lines) >= 1, "report must contain a GATE VERDICT line"

    def test_compute_caveat_in_report(self, quality_result):
        caveat_lines = [l for l in quality_result.report_lines if "COMPUTE CAVEAT" in l]
        assert len(caveat_lines) >= 1, "report must contain COMPUTE CAVEAT line"


class TestReportGeneration:
    def test_writes_real_model_report(self, quality_result):
        """Write real-model section to results/REPORT_phase7.md (separate from simulator REPORT.md
        which test_report.py regenerates and would overwrite an append)."""
        import os
        report_path = os.path.join(
            os.path.dirname(__file__), "..", "results", "REPORT_phase7.md"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(quality_result.report_lines))
            f.write("\n")
        assert os.path.exists(report_path)
        with open(report_path, encoding="utf-8") as f:
            content = f.read()
        assert "GATE VERDICT" in content, "real-model report must contain GATE VERDICT line"
