"""
Phase 8.5 acceptance test: scale generalization check.

Gates:
  - 4 ScalePoints returned (2 models × 2 context lengths)
  - each ScalePoint has valid NLL (>0), storage_bits > 0, on_pareto is bool
  - generalizes_verdict in {"GENERALIZES_ALL", "GENERALIZES_SOME", "NONE"}
  - report written and contains "PHASE 8.5 GENERALIZATION VERDICT"
  - does NOT assert GENERALIZES_ALL (honesty spine)
  - long context has more B0 storage than short context for same model (sanity)
  - 0.5B short storage < 1.5B short storage (smaller model = smaller KV)
  - DynamicCache .layers API consistent across both models
"""

import os
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from aepk_paging.harness.phase8_scale import (
    run_scale_check,
    ScaleCheckResult,
    ScalePoint,
    MODEL_IDS,
    CTX_CONFIGS,
    SCALE_NOISE_LEVEL,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

EXPECTED_MODELS = set(MODEL_IDS)
EXPECTED_CTX = {label for label, _ in CTX_CONFIGS}


@pytest.fixture(scope="module")
def scale_result():
    return run_scale_check(DEVICE, DTYPE)


class TestScaleStructure:
    def test_returns_scale_check_result(self, scale_result):
        assert isinstance(scale_result, ScaleCheckResult)

    def test_four_scale_points(self, scale_result):
        assert len(scale_result.scale_points) == 4

    def test_all_model_ctx_combinations_present(self, scale_result):
        combos = {(pt.model_id, pt.ctx_label) for pt in scale_result.scale_points}
        expected = {(m, c) for m in MODEL_IDS for c, _ in CTX_CONFIGS}
        assert combos == expected, f"Missing: {expected - combos}"

    def test_generalizes_verdict_valid(self, scale_result):
        assert scale_result.generalizes_verdict in (
            "GENERALIZES_ALL", "GENERALIZES_SOME", "NONE"
        ), f"Got: {scale_result.generalizes_verdict!r}"


class TestScaleValues:
    def test_all_nlls_positive(self, scale_result):
        for pt in scale_result.scale_points:
            assert pt.b0_nll > 0.0, f"{pt.model_id}/{pt.ctx_label}: b0_nll={pt.b0_nll}"
            assert pt.b3_nll > 0.0, f"{pt.model_id}/{pt.ctx_label}: b3_nll={pt.b3_nll}"

    def test_all_storage_positive(self, scale_result):
        for pt in scale_result.scale_points:
            assert pt.b0_storage_bits > 0
            assert pt.b3_storage_bits > 0

    def test_on_pareto_is_bool(self, scale_result):
        for pt in scale_result.scale_points:
            assert isinstance(pt.on_pareto, bool)

    def test_nll_delta_definition(self, scale_result):
        for pt in scale_result.scale_points:
            assert abs(pt.nll_delta - (pt.b3_nll - pt.b0_nll)) < 1e-5

    def test_long_ctx_more_storage_than_short_same_model(self, scale_result):
        """Longer prefix → more KV tokens → more raw storage."""
        by_key = {(pt.model_id, pt.ctx_label): pt for pt in scale_result.scale_points}
        for model_id in MODEL_IDS:
            short_pt = by_key[(model_id, "short")]
            long_pt  = by_key[(model_id, "long")]
            assert long_pt.b0_storage_bits > short_pt.b0_storage_bits, (
                f"{model_id}: long ({long_pt.b0_storage_bits}) should > "
                f"short ({short_pt.b0_storage_bits})"
            )

    def test_0_5b_smaller_storage_than_1_5b_same_ctx(self, scale_result):
        """Smaller model → fewer/smaller KV pages → less raw storage."""
        by_key = {(pt.model_id, pt.ctx_label): pt for pt in scale_result.scale_points}
        for ctx_label, _ in CTX_CONFIGS:
            pt_05 = by_key[("Qwen/Qwen2.5-0.5B-Instruct", ctx_label)]
            pt_15 = by_key[("Qwen/Qwen2.5-1.5B-Instruct", ctx_label)]
            assert pt_05.b0_storage_bits < pt_15.b0_storage_bits, (
                f"ctx={ctx_label}: 0.5B ({pt_05.b0_storage_bits}) should < "
                f"1.5B ({pt_15.b0_storage_bits})"
            )

    def test_prefix_token_count_long_gt_short(self, scale_result):
        by_key = {(pt.model_id, pt.ctx_label): pt for pt in scale_result.scale_points}
        for model_id in MODEL_IDS:
            assert (
                by_key[(model_id, "long")].prefix_token_count >
                by_key[(model_id, "short")].prefix_token_count
            )


class TestGeneralizesLogic:
    def test_verdict_matches_pareto_count(self, scale_result):
        n = sum(pt.on_pareto for pt in scale_result.scale_points)
        total = len(scale_result.scale_points)
        if scale_result.generalizes_verdict == "GENERALIZES_ALL":
            assert n == total
        elif scale_result.generalizes_verdict == "GENERALIZES_SOME":
            assert 0 < n < total
        else:
            assert n == 0


class TestReport:
    def test_report_written(self, scale_result):
        assert os.path.exists(scale_result.report_path), (
            f"Report not at {scale_result.report_path}"
        )

    def test_report_contains_verdict(self, scale_result):
        with open(scale_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "PHASE 8.5 GENERALIZATION VERDICT:" in content

    def test_report_contains_both_models(self, scale_result):
        with open(scale_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "1.5B" in content
        assert "0.5B" in content

    def test_results_logged(self, scale_result):
        print(f"\n{'='*70}")
        print(f"Phase 8.5 Scale Generalization Results (noise={SCALE_NOISE_LEVEL})")
        print(f"{'='*70}")
        print(f"{'Model':<30} | {'Ctx':<5} | {'pfx_tok':>7} | {'B0_NLL':>7} | "
              f"{'B3_NLL':>7} | {'dNLL':>7} | {'sav%':>6} | {'Pareto':>6}")
        print("-" * 80)
        for pt in scale_result.scale_points:
            short = pt.model_id.split("/")[-1]
            print(f"{short:<30} | {pt.ctx_label:<5} | {pt.prefix_token_count:>7} | "
                  f"{pt.b0_nll:>7.4f} | {pt.b3_nll:>7.4f} | {pt.nll_delta:>+7.4f} | "
                  f"{pt.storage_savings_pct:>+5.1f}% | {'YES' if pt.on_pareto else 'no':>6}")
        print(f"\nGeneralization verdict: {scale_result.generalizes_verdict}")
        print(f"Report: {scale_result.report_path}")
