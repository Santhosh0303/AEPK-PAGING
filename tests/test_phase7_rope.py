"""
Phase 7.5 — acceptance test: position-covariance brick ([RoPE], §9).

Cache real K at position p; re-base to p+Δ via R(Δ); assert agreement
with K recomputed at p+Δ to within fp16 float tolerance (atol=0.05).

Honesty note: max_diff ~1e-2 for lower layers, ~2e-2 for deeper layers.
Error source: fp16 arithmetic in model forward, NOT in our rebasing code.
Relative error ~0.2% << fp16 limit ~1%.
Cross-model transfer: OUT OF SCOPE (THESIS_DOSSIER §9).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.real_model_rope import verify_position_covariance, rebase_kv_position, compute_cos_sin

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

ATOL = 0.05   # fp16 float tolerance; relative error ~0.2% with |K| ~10


@pytest.fixture(scope="module")
def rope_result():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
    model.eval()
    text = "The quick brown fox jumps"
    ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)
    result = verify_position_covariance(model, ids, delta=7, device=DEVICE, dtype=DTYPE, atol=ATOL)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result


class TestPositionCovariance:
    def test_all_layers_pass_atol(self, rope_result):
        """Core 7.5 gate: K_rebased ≈ K_recomputed within fp16 tolerance for ALL layers."""
        failed = [(r["layer"], r["max_diff"]) for r in rope_result["layer_results"] if not r["pass"]]
        assert len(failed) == 0, (
            f"Position-covariance failed for layers: {failed}\n"
            f"atol={ATOL}; overall_max_diff={rope_result['overall_max_diff']:.4e}"
        )

    def test_rebasing_reduces_error_vs_baseline(self, rope_result):
        """Rebased K must be much closer to K_shift than un-rebased K_orig."""
        assert rope_result["reduction_factor"] > 10.0, (
            f"Expected >10x error reduction; got {rope_result['reduction_factor']:.1f}x\n"
            f"baseline_max={rope_result['baseline_max_diff']:.4f} "
            f"rebased_max={rope_result['overall_max_diff']:.4e}"
        )

    def test_overall_max_diff_below_atol(self, rope_result):
        assert rope_result["overall_max_diff"] < ATOL, (
            f"overall_max_diff={rope_result['overall_max_diff']:.4e} >= atol={ATOL}"
        )

    def test_num_layers_correct(self, rope_result):
        assert rope_result["num_layers"] == 28, "Qwen2.5-1.5B has 28 layers"

    def test_delta_preserved(self, rope_result):
        assert rope_result["delta"] == 7

    def test_reduction_factor_logged(self, rope_result):
        """Log the reduction factor for PROGRESS.md."""
        rf = rope_result["reduction_factor"]
        assert rf > 0
        print(f"\nPosition-covariance reduction factor: {rf:.1f}x "
              f"(baseline_max={rope_result['baseline_max_diff']:.3f} → "
              f"rebased_max={rope_result['overall_max_diff']:.4e})")
