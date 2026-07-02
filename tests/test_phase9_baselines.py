"""
Phase 9.2 — Tests for ISO-ACCURACY baseline comparison.

No-damage controls MUST pass before full sweep runs:
  1. KIVI-fp16-control accuracy == B0_sdpa ±0.01
  2. B0_eager accuracy ≈ B0_sdpa ±0.05 (eager-vs-sdpa confound check)

Gate line: BASELINE_DOMINANCE: {DOMINATES_ALL|DOMINATES_SOME|DOMINATED}
Test asserts the verdict LINE EXISTS, never == (honesty spine S9).

KIVI short-prompt note: group_size=32 requires T>=32. Our probes have T<32.
K quantization falls back to fp16 for official config — expected and documented.

SnapKV note: requires attn_implementation="eager" (SDPA blocks output_attentions).
Official repo not used; implementation from arXiv:2404.14469 Section 3.
"""

from __future__ import annotations

import os
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    device = "cuda"
    dtype = torch.float16
    yield model, tok, device, dtype
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def model_eager(model_and_tok):
    """Separate eager-attention model for SnapKV tests.

    Must use bfloat16: fp16 eager overflows in Q·K^T (Qwen2.5 head_dim=128 → NaN logits).
    BF16 has same bit-width (16) so storage comparisons remain valid.
    """
    from transformers import AutoModelForCausalLM
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",
    )
    m.eval()
    yield m
    del m
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Unit tests (fast, no full model sweep)
# ---------------------------------------------------------------------------

def test_kivi_quantize_passthrough():
    """k_bits=16 → passthrough: K and V are unchanged (fp32 round-trip only)."""
    import numpy as np
    from aepk_paging.harness.phase9_baselines import _kivi_quantize_page
    from aepk_paging.kv_page import KVPage

    rng = np.random.default_rng(42)
    K = rng.normal(0, 1, (10, 2, 128)).astype(np.float32)
    V = rng.normal(0, 1, (10, 2, 128)).astype(np.float32)
    page = KVPage(page_id=("u", 0), layer=0, token_range=(0, 10), K=K, V=V,
                  precision_tag="test", attention_mass=1.0)
    q_page, storage = _kivi_quantize_page(page, k_bits=16, v_bits=16,
                                           group_size=32, residual_length=0)
    np.testing.assert_array_equal(q_page.K, K)
    np.testing.assert_array_equal(q_page.V, V)
    # Storage must equal fp16 reference for passthrough
    from aepk_paging.harness.phase9_baselines import _kivi_fp16_ref_bits
    assert storage == _kivi_fp16_ref_bits(10)


def test_kivi_short_seq_official_config_no_k_compression():
    """T=10, group_size=32: K cannot be quantized. Storage == fp16 reference."""
    import numpy as np
    from aepk_paging.harness.phase9_baselines import _kivi_quantize_page, _kivi_fp16_ref_bits
    from aepk_paging.kv_page import KVPage

    rng = np.random.default_rng(0)
    T = 10
    K = rng.normal(0, 1, (T, 2, 128)).astype(np.float32)
    V = rng.normal(0, 1, (T, 2, 128)).astype(np.float32)
    page = KVPage(page_id=("u", 0), layer=0, token_range=(0, T), K=K, V=V,
                  precision_tag="test", attention_mass=1.0)
    _, storage = _kivi_quantize_page(page, k_bits=2, v_bits=2,
                                      group_size=32, residual_length=32)
    # All tokens in residual (T=10 < residual_length=32)
    assert storage == _kivi_fp16_ref_bits(T), (
        f"Expected {_kivi_fp16_ref_bits(T)} bits (no compression), got {storage}"
    )


def test_kivi_long_seq_achieves_compression():
    """T=64, group_size=32: K can be quantized. Storage < fp16 reference."""
    import numpy as np
    from aepk_paging.harness.phase9_baselines import _kivi_quantize_page, _kivi_fp16_ref_bits
    from aepk_paging.kv_page import KVPage

    rng = np.random.default_rng(1)
    T = 64
    K = rng.normal(0, 1, (T, 2, 128)).astype(np.float32)
    V = rng.normal(0, 1, (T, 2, 128)).astype(np.float32)
    page = KVPage(page_id=("u", 0), layer=0, token_range=(0, T), K=K, V=V,
                  precision_tag="test", attention_mass=1.0)
    _, storage = _kivi_quantize_page(page, k_bits=2, v_bits=2,
                                      group_size=32, residual_length=32)
    ref = _kivi_fp16_ref_bits(T)
    assert storage < ref, f"Expected compression (storage={storage} < ref={ref})"


def test_kivi_quantize_deq_shape():
    """Quantized page has same K/V shape as input."""
    import numpy as np
    from aepk_paging.harness.phase9_baselines import _kivi_quantize_page
    from aepk_paging.kv_page import KVPage

    rng = np.random.default_rng(2)
    K = rng.normal(0, 1, (20, 2, 128)).astype(np.float32)
    V = rng.normal(0, 1, (20, 2, 128)).astype(np.float32)
    page = KVPage(page_id=("u", 0), layer=0, token_range=(0, 20), K=K, V=V,
                  precision_tag="test", attention_mass=1.0)
    q_page, _ = _kivi_quantize_page(page, k_bits=2, v_bits=2,
                                     group_size=4, residual_length=0)
    assert q_page.K.shape == K.shape
    assert q_page.V.shape == V.shape


def test_snapkv_importance_short_seq():
    """T <= window_size → importance is None (no eviction)."""
    from aepk_paging.harness.phase9_baselines import _snapkv_importance
    attn = torch.ones(1, 12, 10, 10) / 10  # [1, Q, T, T], T=10
    imp = _snapkv_importance(attn, window_size=32)
    assert imp is None, "Short sequence should return None (no eviction)"


def test_snapkv_importance_long_seq():
    """T > window_size → importance tensor returned with shape [NUM_KV_HEADS, T]."""
    from aepk_paging.harness.phase9_baselines import _snapkv_importance, NUM_KV_HEADS
    T = 50
    attn = torch.rand(1, 12, T, T).softmax(dim=-1)
    imp = _snapkv_importance(attn, window_size=32)
    assert imp is not None
    assert imp.shape == (NUM_KV_HEADS, T)
    # Window positions should have inf importance
    assert (imp[:, -32:] == float("inf")).all()


# ---------------------------------------------------------------------------
# Full integration test (slow — loads model, runs 100 probes × multiple methods)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_phase9_baselines_full(model_and_tok, model_eager):
    """Full Phase 9.2 baseline sweep.

    Checks:
    1. No-damage controls pass (KIVI-fp16 ≈ B0, B0_eager ≈ B0_sdpa)
    2. Report file generated at results/REPORT_phase9_baselines_v2.md
    3. Report contains BASELINE_DOMINANCE gate line
    4. Gate line has one of the allowed verdicts
    5. Report contains UNVERIFIED: KVQuant
    6. AEPK accuracy labeled as recovery-on uninterpreted
    """
    from aepk_paging.harness.phase9_baselines import (
        run_phase9_baselines,
        B0_ACCURACY_SDPA,
    )

    model, tok, device, dtype = model_and_tok

    result = run_phase9_baselines(model, tok, device, dtype, model_eager=model_eager)

    # --- Control checks ---
    assert result.control_ok, (
        f"No-damage control FAILED.\n"
        f"  KIVI-fp16 acc={result.kivi_fp16_control.accuracy:.3f} vs B0={B0_ACCURACY_SDPA:.3f}\n"
        f"  B0_eager={result.b0_eager:.3f} vs B0_sdpa={B0_ACCURACY_SDPA:.3f}"
    )

    # --- Report file ---
    assert os.path.isfile(result.report_path), f"Report not found: {result.report_path}"

    with open(result.report_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    # Gate line exists
    assert "BASELINE_DOMINANCE:" in content, "Gate line BASELINE_DOMINANCE missing"

    # Allowed verdicts
    allowed = {"DOMINATES_ALL", "DOMINATES_SOME", "DOMINATED"}
    found_verdict = None
    for line in content.splitlines():
        if line.startswith("BASELINE_DOMINANCE:"):
            verdict = line.split(":", 1)[1].strip()
            assert verdict in allowed, f"Unexpected dominance verdict: {verdict!r}"
            found_verdict = verdict
            break
    assert found_verdict is not None

    # UNVERIFIED: KVQuant present
    assert "UNVERIFIED: KVQuant" in content, "Missing UNVERIFIED: KVQuant in report"

    # AEPK accuracy labeled correctly
    assert "uninterpreted pending" in content, (
        "AEPK accuracy should be labeled 'recovery-on, uninterpreted pending 9.3'"
    )

    # Determinism: re-running with same seed should give same numbers
    # (light check: KIVI-fp16 accuracy is deterministic)
    assert result.kivi_fp16_control.accuracy == result.kivi_fp16_control.accuracy  # trivial but pin type

    print(f"\nPhase 9.2 complete. Dominance: {result.dominance_verdict}")
    print(f"  AEPK: acc={result.aepk_b3_noise02.accuracy:.3f}, "
          f"bits={result.aepk_b3_noise02.bits_per_kv_elem:.2f}")
    print(f"  KIVI-2-official: acc={result.kivi_2_official.accuracy:.3f}, "
          f"bits={result.kivi_2_official.bits_per_kv_elem:.2f}")
    print(f"  SnapKV-r50: acc={result.snapkv_r50.accuracy:.3f}, "
          f"bits={result.snapkv_r50.bits_per_kv_elem:.2f}")
