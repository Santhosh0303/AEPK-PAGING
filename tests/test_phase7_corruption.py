"""
Phase 7.3 — acceptance test: real KV corruption → detect → recover → residency.

Loads Qwen2.5-1.5B-Instruct, extracts real past_key_values, runs the full
Phase-2→4→3→5 pipeline on a sample of layers.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from aepk_paging.real_model_adapter import dynamiccache_to_pages
from aepk_paging.harness.phase7_harness import run_phase7_corruption_pipeline
from aepk_paging.kv_page import ResidencyTier

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@pytest.fixture(scope="module")
def real_pages():
    """4 layers sampled from real model KV (layers 0, 7, 14, 21)."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    prompt = "The agent execution physics kernel manages KV memory pages."
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inp, use_cache=True)
    all_pages = dynamiccache_to_pages(out.past_key_values)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return [all_pages[i] for i in [0, 7, 14, 21]]


@pytest.fixture(scope="module")
def pipeline_result(real_pages):
    return run_phase7_corruption_pipeline(
        real_pages,
        noise_level=0.5,
        noise_seed=42,
        num_rs_parity=1,
        rs_error_t=2,
        bit_flip_p=0.02,
    )


class TestDetection:
    def test_clean_pages_mostly_unflagged(self, pipeline_result):
        """Clean pages should not trip detectors (or minimally)."""
        clean_flags = sum(1 for d in pipeline_result.clean_detections if d.finiteness.flag)
        assert clean_flags == 0, "clean pages must have no non-finite values"

    def test_corrupt_pages_have_elevated_mse(self, pipeline_result):
        """quant_noise must produce nonzero MSE."""
        assert pipeline_result.quant_noise_mse > 0.0

    def test_finiteness_detector_clean_all_pass(self, pipeline_result):
        for det in pipeline_result.clean_detections:
            assert not det.finiteness.flag, f"page {det.page_id} has non-finite values in clean KV"

    def test_norm_consistency_records_deviation(self, pipeline_result):
        """Detectors run without error on real 3D KVPages."""
        for det in pipeline_result.corrupt_detections:
            assert det.norm_consistency.deviation >= 0.0


class TestErasureRecovery:
    def test_erasure_recovered_bit_exact(self, pipeline_result):
        assert pipeline_result.erasure_recovered_bit_exact, (
            "RS erasure recovery must be bit-exact for real KV pages"
        )


class TestErrorCorrection:
    def test_error_correction_mse_finite(self, pipeline_result):
        assert np.isfinite(pipeline_result.error_correction_mse_before)
        assert np.isfinite(pipeline_result.error_correction_mse_after)

    def test_error_correction_improves_or_equal(self, pipeline_result):
        assert pipeline_result.error_correction_improved, (
            f"error correction must not worsen MSE: "
            f"before={pipeline_result.error_correction_mse_before:.6f} "
            f"after={pipeline_result.error_correction_mse_after:.6f}"
        )


class TestResidency:
    def test_residency_plan_non_empty(self, pipeline_result):
        assert len(pipeline_result.residency_tiers) > 0

    def test_eviction_within_parity_bound(self, pipeline_result):
        """Capacity-coupled residency: evicted <= num_rs_parity (=1) per parity group."""
        assert pipeline_result.evicted_count <= 1, (
            f"evicted={pipeline_result.evicted_count} exceeds parity bound 1"
        )

    def test_all_tiers_valid(self, pipeline_result):
        for pid, tier in pipeline_result.residency_tiers.items():
            assert isinstance(tier, ResidencyTier)
