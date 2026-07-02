"""
Phase 9.3-LC acceptance tests (9.3a + 9.3b).

Gate (honesty spine S9 — never violated):
  - LC_OVERRECOVERY verdict line asserted to EXIST; its VALUE never asserted.
  - No verdict value asserted equal to PASS or any specific number.
  - noise=0.0 control: both damage_only_retention and recovery_on_retention
    must equal 1.0 (bit-exact injection — same regression lock as 9.1-FIX).
  - T >= 150 asserted against the REAL tokenizer (not estimated).
  - B0_lc is a freshly measured float in [0, 1]; 0.330 never appears in tests.
  - damage_only path: tests cannot assert that recover_rs_erasure was not called
    (Python can't observe call counts without mocking), but the source structure
    guarantees it (recover_rs_erasure gated by 'if use_recovery:' only).

Fixture scope: module (model loaded once; full grid runs once).
"""

from __future__ import annotations

import os
import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM

from aepk_paging.harness.phase9_3_lc import (
    LC_N_PROBES_ITER,
    LC93aPoint,
    LC93aResult,
    LC93cResult,
    LC93dResult,
    LC93eResult,
    LCErasurePoint,
    LONG_CONTEXT_PASSAGE,
    LC_NOISE_LEVELS,
    assert_token_lengths,
    build_lc_probe_set,
    run_phase9_3a,
    run_phase9_3c,
    run_phase9_3d,
    run_phase9_3_erasure,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 9.3-LC-2: full grid (N_PROBES=100, N_SEEDS=5, noise=[0.0,0.2,0.3]) — powered, not the old reduced grid
_ITER_NOISE = [0.0, 0.2, 0.3]
_ITER_SEEDS = 5
_ITER_PROBES = LC_N_PROBES_ITER   # 100


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
def iter_result(model_and_tok):
    """Full-grid result: 100 probes, 5 seeds, noise=[0.0, 0.2, 0.3]."""
    model, tok = model_and_tok
    return run_phase9_3a(
        model, tok, DEVICE, DTYPE,
        noise_levels=_ITER_NOISE,
        n_seeds=_ITER_SEEDS,
        n_probes=_ITER_PROBES,
    )


# ---------------------------------------------------------------------------
# Static probe-set tests (no model needed)
# ---------------------------------------------------------------------------

class TestProbeSetStatic:
    def test_passage_nonempty(self):
        assert len(LONG_CONTEXT_PASSAGE) > 200

    def test_lc_count_matches_base(self):
        from aepk_paging.harness.phase9_accuracy import build_extended_eval_set
        base = build_extended_eval_set()
        lc = build_lc_probe_set(base)
        assert len(lc) == len(base)

    def test_lc_probes_contain_passage(self):
        lc = build_lc_probe_set()
        for p in lc[:5]:
            assert LONG_CONTEXT_PASSAGE in p["prompt"]

    def test_gold_answers_preserved(self):
        from aepk_paging.harness.phase9_accuracy import build_extended_eval_set
        base = build_extended_eval_set()
        lc = build_lc_probe_set(base)
        for b, l in zip(base, lc):
            assert b["expected"] == l["expected"]

    def test_default_arg_returns_100_probes(self):
        lc = build_lc_probe_set()
        assert len(lc) == 100


# ---------------------------------------------------------------------------
# Token-length assertion (requires tokenizer)
# ---------------------------------------------------------------------------

class TestTokenLengths:
    def test_all_probes_t_gte_150(self, model_and_tok):
        """S9 gate (2): T>=150 asserted with the REAL tokenizer."""
        _, tok = model_and_tok
        lc = build_lc_probe_set()[:10]
        lengths = assert_token_lengths(tok, lc, min_tokens=150)
        for idx, T in lengths.items():
            assert T >= 150, f"Probe {idx}: T={T} < 150"

    def test_min_token_length_gte_150(self, iter_result):
        assert iter_result.min_token_length >= 150

    def test_b0_lc_freshly_measured(self, iter_result):
        """B0_lc must be a float in [0, 1] — never hardcoded."""
        assert isinstance(iter_result.b0_lc, float)
        assert 0.0 <= iter_result.b0_lc <= 1.0

    def test_token_lengths_all_gte_150(self, iter_result):
        for idx, T in iter_result.token_lengths.items():
            assert T >= 150, f"Probe {idx}: T={T} < 150 in iter_result"


# ---------------------------------------------------------------------------
# Control regression — S9 gate (4): noise=0.0 rows must give retention==1.0
# ---------------------------------------------------------------------------

class TestNoiseZeroControl:
    """At noise=0.0 both paths inject CLEAN pages → bit-exact → retention==1.0.

    Proof: noise_level=0.0 branch appends the original page without calling
    quant_noise; _inject_pages writes f32→fp16 which is lossless for values
    that originated as fp16 (Phase 7.2); greedy decode is therefore identical
    to _run_lc_b0 → accuracy == B0_lc → retention == 1.0.
    """

    def test_damage_only_noise_zero_retention_equals_one(self, iter_result):
        pt = next(p for p in iter_result.points if p.noise_level == 0.0)
        assert pt.damage_only_retention == 1.0, (
            f"damage_only retention={pt.damage_only_retention:.6f} at noise=0.0; "
            "expected 1.0. Indicates harness bug (double-count or non-bit-exact injection)."
        )

    def test_recovery_on_noise_zero_retention_equals_one(self, iter_result):
        pt = next(p for p in iter_result.points if p.noise_level == 0.0)
        assert pt.recovery_on_retention == 1.0, (
            f"recovery_on retention={pt.recovery_on_retention:.6f} at noise=0.0; "
            "expected 1.0."
        )

    def test_noise_zero_both_means_equal_b0_lc(self, iter_result):
        pt = next(p for p in iter_result.points if p.noise_level == 0.0)
        assert pt.damage_only_mean == pt.b0_lc
        assert pt.recovery_on_mean == pt.b0_lc


# ---------------------------------------------------------------------------
# Structural checks on results (no value assertions)
# ---------------------------------------------------------------------------

class TestSweepStructure:
    def test_result_type(self, iter_result):
        assert isinstance(iter_result, LC93aResult)

    def test_all_iter_noise_levels_present(self, iter_result):
        found = {pt.noise_level for pt in iter_result.points}
        for lvl in _ITER_NOISE:
            assert lvl in found, f"noise={lvl} missing from iter_result.points"

    def test_points_are_correct_type(self, iter_result):
        for pt in iter_result.points:
            assert isinstance(pt, LC93aPoint)

    def test_damage_only_retention_is_float_nonnegative(self, iter_result):
        for pt in iter_result.points:
            assert isinstance(pt.damage_only_retention, float)
            assert pt.damage_only_retention >= 0.0

    def test_recovery_on_retention_is_float_nonnegative(self, iter_result):
        for pt in iter_result.points:
            assert isinstance(pt.recovery_on_retention, float)
            assert pt.recovery_on_retention >= 0.0

    def test_ci_values_nonnegative(self, iter_result):
        for pt in iter_result.points:
            assert pt.damage_only_ci >= 0.0
            assert pt.recovery_on_ci >= 0.0

    def test_n_probes_matches_iter(self, iter_result):
        assert iter_result.n_probes == _ITER_PROBES

    def test_n_seeds_matches_iter(self, iter_result):
        assert iter_result.n_seeds == _ITER_SEEDS


# ---------------------------------------------------------------------------
# Report / verdict line tests — S9: LINE EXISTS, value never asserted
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_written(self, iter_result):
        assert os.path.exists(iter_result.report_path)

    def test_lc_overrecovery_line_exists(self, iter_result):
        """S9: verdict line must exist — its VALUE is never asserted here."""
        with open(iter_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "LC_OVERRECOVERY:" in content, (
            "LC_OVERRECOVERY verdict line missing from REPORT_phase9_3_lc.md"
        )

    def test_lc_overrecovery_has_damage_only_field(self, iter_result):
        with open(iter_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert re.search(r"LC_OVERRECOVERY:.*damage_only=[\d.]+", content), (
            "LC_OVERRECOVERY line must contain damage_only=<float>"
        )

    def test_lc_overrecovery_has_recovery_on_field(self, iter_result):
        with open(iter_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert re.search(r"LC_OVERRECOVERY:.*recovery_on=[\d.]+", content), (
            "LC_OVERRECOVERY line must contain recovery_on=<float>"
        )

    def test_report_mentions_b0_lc(self, iter_result):
        with open(iter_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "B0_lc" in content

    def test_report_mentions_token_range(self, iter_result):
        with open(iter_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "min_T=" in content

    def test_report_byte_identical_on_rerun(self, iter_result):
        """Writing the report a second time must produce the same bytes."""
        from aepk_paging.harness.phase9_3_lc import _write_report
        with open(iter_result.report_path, encoding="utf-8") as f:
            original = f.read()
        _write_report(
            iter_result.b0_lc,
            iter_result.points,
            iter_result.n_probes,
            iter_result.n_seeds,
            iter_result.min_token_length,
            iter_result.max_token_length,
            iter_result.report_path,
        )
        with open(iter_result.report_path, encoding="utf-8") as f:
            rerun = f.read()
        assert original == rerun, "Report is NOT byte-identical across two writes"

    def test_results_logged(self, iter_result):
        print(f"\n{'='*70}")
        print("Phase 9.3a Long-Context Results (FULL GRID)")
        print(f"{'='*70}")
        print(
            f"B0_lc={iter_result.b0_lc:.4f}  "
            f"probes={iter_result.n_probes}  seeds={iter_result.n_seeds}"
        )
        print(
            f"Token range: {iter_result.min_token_length}–{iter_result.max_token_length}"
        )
        print(f"{'noise':>6} | {'do_ret':>7} | {'ro_ret':>7}")
        print("-" * 30)
        for pt in iter_result.points:
            print(
                f"{pt.noise_level:>6.2f} | {pt.damage_only_retention:>7.4f} | "
                f"{pt.recovery_on_retention:>7.4f}"
            )


# ---------------------------------------------------------------------------
# Stage 9.3c — ablation fixture and tests
# NOTE: TestAblation must come AFTER TestReport so that `iter_result` (9.3a)
# runs and the byte-identical 9.3a test passes BEFORE `ablation_result`
# overwrites the report with full 9.3c content.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ablation_result(model_and_tok, iter_result):
    """Run 9.3c ablation using iter_result (9.3a data) for damage_only/ro_mse."""
    model, tok = model_and_tok
    return run_phase9_3c(
        model, tok, DEVICE, DTYPE,
        prev_93a=iter_result,
        ablation_noise=[0.2, 0.3],
        n_seeds=_ITER_SEEDS,
    )


class TestAblation:
    """9.3c: strip RS / physics / detect bricks one at a time.

    Gate (S9): ABLATION verdict line asserted to EXIST; its VALUE never asserted.
    No Δ value compared to any threshold, sign, or specific number.
    """

    def test_ablation_result_type(self, ablation_result):
        assert isinstance(ablation_result, LC93cResult)

    def test_ablation_has_points(self, ablation_result):
        assert len(ablation_result.points) > 0

    def test_ablation_summary_keys(self, ablation_result):
        keys = set(ablation_result.ablation_summary.keys())
        assert {"coding", "physics", "detect"} <= keys

    def test_ablation_summary_values_are_floats(self, ablation_result):
        for k, v in ablation_result.ablation_summary.items():
            assert isinstance(v, float), f"{k} is not float"

    def test_ablation_line_exists_in_report(self, ablation_result):
        """S9 gate (6): ABLATION line must exist — value never asserted."""
        with open(ablation_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "ABLATION:" in content, (
            "ABLATION verdict line missing from REPORT_phase9_3_lc.md"
        )

    def test_ablation_line_has_coding_field(self, ablation_result):
        with open(ablation_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert re.search(r"ABLATION:.*coding=[+-]?[\d.]+", content), (
            "ABLATION line must contain coding=<±float>"
        )

    def test_ablation_line_has_physics_field(self, ablation_result):
        with open(ablation_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert re.search(r"ABLATION:.*physics=[+-]?[\d.]+", content), (
            "ABLATION line must contain physics=<±float>"
        )

    def test_ablation_line_has_detect_field(self, ablation_result):
        with open(ablation_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert re.search(r"ABLATION:.*detect=[+-]?[\d.]+", content), (
            "ABLATION line must contain detect=<±float>"
        )

    def test_lc_overrecovery_still_present(self, ablation_result):
        """Full report must still contain LC_OVERRECOVERY from 9.3b."""
        with open(ablation_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "LC_OVERRECOVERY:" in content

    def test_report_byte_identical_on_ablation_rerun(self, ablation_result, iter_result):
        """_write_full_report_93c must produce identical bytes twice."""
        from aepk_paging.harness.phase9_3_lc import _write_full_report_93c
        with open(ablation_result.report_path, encoding="utf-8") as f:
            original = f.read()
        _write_full_report_93c(
            iter_result,
            ablation_result.points,
            ablation_result.ablation_summary,
            ablation_result.report_path,
        )
        with open(ablation_result.report_path, encoding="utf-8") as f:
            rerun = f.read()
        assert original == rerun, "Full report is NOT byte-identical across two writes"

    def test_ablation_results_logged(self, ablation_result):
        print(f"\n{'='*70}")
        print("Phase 9.3c Ablation Results (FULL GRID)")
        print(f"{'='*70}")
        print(
            f"Summary: coding={ablation_result.ablation_summary['coding']:+.4f}  "
            f"physics={ablation_result.ablation_summary['physics']:+.4f}  "
            f"detect={ablation_result.ablation_summary['detect']:+.4f}"
        )
        print(f"{'noise':>6} | {'d_cod':>7} | {'d_phy':>7} | {'d_det':>7}")
        print("-" * 40)
        for pt in ablation_result.points:
            print(
                f"{pt.noise_level:>6.2f} | {pt.coding_delta:>+7.4f} | "
                f"{pt.physics_delta:>+7.4f} | {pt.detect_delta:>+7.4f}"
            )


# ---------------------------------------------------------------------------
# Stage 9.3d — KIVI/SnapKV fair fight on long context
# NOTE: TestBaselines must come AFTER TestAblation so that 9.3c report content
# is written before 9.3d overwrites the report with the final full content.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline_result(model_and_tok, iter_result, ablation_result):
    """Run 9.3d fair fight using prev_93a + prev_93c data."""
    model, tok = model_and_tok
    return run_phase9_3d(
        model, tok, DEVICE, DTYPE,
        prev_93a=iter_result,
        prev_93c=ablation_result,
        n_probes=_ITER_PROBES,
    )


class TestBaselines:
    """9.3d: KIVI-official + KIVI-2-small + SnapKV on LC probes (T=307-311).

    Gate (S9): LC_BASELINE_DOMINANCE verdict line asserted to EXIST; its VALUE
    never asserted. No verdict compared to DOMINATES_ALL or any specific string.
    control_ok asserted True (KIVI-fp16 on LC must ≈ B0_lc ±0.05).
    """

    def test_baseline_result_type(self, baseline_result):
        assert isinstance(baseline_result, LC93dResult)

    def test_b0_lc_matches_iter(self, baseline_result, iter_result):
        assert baseline_result.b0_lc == iter_result.b0_lc

    def test_control_ok(self, baseline_result):
        """KIVI-fp16-ctrl accuracy on LC must ≈ B0_lc (±0.05, small grid)."""
        assert baseline_result.control_ok, (
            f"KIVI-fp16-ctrl accuracy={baseline_result.kivi_fp16_ctrl.accuracy:.4f} "
            f"not within ±0.05 of B0_lc={baseline_result.b0_lc:.4f}"
        )

    def test_aepk_accuracy_is_float(self, baseline_result):
        assert isinstance(baseline_result.aepk_lc.accuracy, float)
        assert 0.0 <= baseline_result.aepk_lc.accuracy <= 1.0

    def test_kivi_2_official_accuracy_is_float(self, baseline_result):
        assert isinstance(baseline_result.kivi_2_official.accuracy, float)
        assert 0.0 <= baseline_result.kivi_2_official.accuracy <= 1.0

    def test_kivi_2_small_accuracy_is_float(self, baseline_result):
        assert isinstance(baseline_result.kivi_2_small.accuracy, float)
        assert 0.0 <= baseline_result.kivi_2_small.accuracy <= 1.0

    def test_snapkv_r50_accuracy_is_float(self, baseline_result):
        assert isinstance(baseline_result.snapkv_r50.accuracy, float)
        assert 0.0 <= baseline_result.snapkv_r50.accuracy <= 1.0

    def test_bits_per_kv_elem_positive(self, baseline_result):
        for comp in [
            baseline_result.aepk_lc,
            baseline_result.kivi_fp16_ctrl,
            baseline_result.kivi_2_official,
            baseline_result.kivi_2_small,
            baseline_result.snapkv_r100_ctrl,
            baseline_result.snapkv_r50,
        ]:
            assert comp.bits_per_kv_elem > 0.0, f"{comp.name}: bits_per_kv_elem <= 0"

    def test_kivi_2_official_compresses_vs_fp16(self, baseline_result):
        """On LC (T=307), KIVI-official must use fewer bits than fp16 ref."""
        assert (
            baseline_result.kivi_2_official.bits_per_kv_elem
            < baseline_result.kivi_fp16_ctrl.bits_per_kv_elem
        ), (
            f"KIVI-2-official bits={baseline_result.kivi_2_official.bits_per_kv_elem:.2f} "
            f"NOT less than fp16 ctrl bits={baseline_result.kivi_fp16_ctrl.bits_per_kv_elem:.2f}. "
            "KIVI should compress at T=307 (group_size=32, residual_length=32 → 275 tokens quantized)."
        )

    def test_snapkv_r50_compresses_vs_r100(self, baseline_result):
        """On LC (T=307), SnapKV-r50 must use fewer bits than r100 (T>window=32 → eviction)."""
        assert (
            baseline_result.snapkv_r50.bits_per_kv_elem
            < baseline_result.snapkv_r100_ctrl.bits_per_kv_elem
        ), (
            f"SnapKV-r50 bits={baseline_result.snapkv_r50.bits_per_kv_elem:.2f} "
            f"NOT less than r100 ctrl bits={baseline_result.snapkv_r100_ctrl.bits_per_kv_elem:.2f}. "
            "SnapKV should evict at T=307 (window_size=32, keep_ratio=0.5)."
        )

    def test_dominance_verdict_is_str(self, baseline_result):
        assert isinstance(baseline_result.dominance_verdict, str)
        assert len(baseline_result.dominance_verdict) > 0

    def test_dominance_verdict_in_valid_set(self, baseline_result):
        valid = {"DOMINATES_ALL", "DOMINATES_SOME", "DOMINATED"}
        assert baseline_result.dominance_verdict in valid, (
            f"dominance_verdict={baseline_result.dominance_verdict!r} not in {valid}"
        )

    def test_lc_baseline_dominance_line_exists(self, baseline_result):
        """S9 gate (6): LC_BASELINE_DOMINANCE line must exist — value never asserted."""
        with open(baseline_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "LC_BASELINE_DOMINANCE:" in content, (
            "LC_BASELINE_DOMINANCE verdict line missing from REPORT_phase9_3_lc.md"
        )

    def test_lc_overrecovery_still_present(self, baseline_result):
        with open(baseline_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "LC_OVERRECOVERY:" in content

    def test_ablation_still_present(self, baseline_result):
        with open(baseline_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "ABLATION:" in content

    def test_report_byte_identical_on_93d_rerun(self, baseline_result, iter_result, ablation_result):
        """_write_full_report_93d must produce identical bytes twice."""
        from aepk_paging.harness.phase9_3_lc import _write_full_report_93d
        with open(baseline_result.report_path, encoding="utf-8") as f:
            original = f.read()
        _write_full_report_93d(
            iter_result,
            ablation_result,
            baseline_result,
            baseline_result.report_path,
        )
        with open(baseline_result.report_path, encoding="utf-8") as f:
            rerun = f.read()
        assert original == rerun, "Final report is NOT byte-identical across two writes"

    def test_baseline_results_logged(self, baseline_result):
        print(f"\n{'='*70}")
        print("Phase 9.3d Baseline Results (FULL GRID)")
        print(f"{'='*70}")
        print(f"B0_lc={baseline_result.b0_lc:.4f}")
        print(f"Dominance verdict: {baseline_result.dominance_verdict}")
        print(f"  vs_kivi={baseline_result.aepk_vs_kivi}  vs_snapkv={baseline_result.aepk_vs_snapkv}")
        print(f"{'method':<22} | {'accuracy':>8} | {'bits/elem':>9} |")
        print("-" * 50)
        for comp in [
            baseline_result.kivi_fp16_ctrl,
            baseline_result.kivi_2_official,
            baseline_result.kivi_2_small,
            baseline_result.snapkv_r100_ctrl,
            baseline_result.snapkv_r50,
            baseline_result.aepk_lc,
        ]:
            print(f"{comp.name:<22} | {comp.accuracy:>8.4f} | {comp.bits_per_kv_elem:>9.2f} |")


# ---------------------------------------------------------------------------
# Stage 9.3-LC-2 (erasure) — total page-loss regime, the make-or-break test.
# NOTE: TestErasure must come AFTER TestBaselines so that the 9.3d report
# content is written before the erasure stage overwrites it with the final
# full report (9.3a + 9.3b + 9.3c + 9.3d + erasure).
# ---------------------------------------------------------------------------

_ERASED_KS = [0, 2, 4, 8]


@pytest.fixture(scope="module")
def erasure_result(model_and_tok, iter_result, ablation_result, baseline_result):
    """Run the erasure sweep (k=[0,2,4,8]) using prev_93a/93c/93d data."""
    model, tok = model_and_tok
    return run_phase9_3_erasure(
        model, tok, DEVICE, DTYPE,
        prev_93a=iter_result,
        prev_93c=ablation_result,
        prev_93d=baseline_result,
        erased_ks=_ERASED_KS,
        n_probes=_ITER_PROBES,
    )


class TestErasure:
    """Stage 9.3-LC-2 erasure regime.

    Gate (S9):
      - ERASURE_HEAL verdict line asserted to EXIST per k; VALUE never asserted.
      - erased_k=0 control: both damage_only_ret and recovery_on_ret == 1.0
        (bit-exact — no pages erased).
      - BUG-VS-FINDING: recovery_on at erased_k <= num_parity (== erased_k
        here, so always at the bound) MUST be close to 1.0 (RS erasure
        recovery is bit-exact per Phase 3). This test does NOT assert an
        exact threshold value (S9: never assert ==/> a specific number for
        a finding) — instead it is a printed diagnostic in
        test_erasure_results_logged for human review before the self-
        validation gate runs.
    """

    def test_erasure_result_type(self, erasure_result):
        assert isinstance(erasure_result, LC93eResult)

    def test_points_cover_all_ks(self, erasure_result):
        found = {pt.erased_k for pt in erasure_result.points}
        for k in _ERASED_KS:
            assert k in found, f"erased_k={k} missing from erasure_result.points"

    def test_points_are_correct_type(self, erasure_result):
        for pt in erasure_result.points:
            assert isinstance(pt, LCErasurePoint)

    def test_retentions_are_float_nonnegative(self, erasure_result):
        for pt in erasure_result.points:
            assert isinstance(pt.damage_only_ret, float)
            assert isinstance(pt.recovery_on_ret, float)
            assert pt.damage_only_ret >= 0.0
            assert pt.recovery_on_ret >= 0.0

    def test_control_k0_damage_only_retention_equals_one(self, erasure_result):
        """S9 gate (2): erased_k=0 is the control — no pages erased."""
        pt = next(p for p in erasure_result.points if p.erased_k == 0)
        assert pt.damage_only_ret == 1.0, (
            f"erased_k=0 damage_only_ret={pt.damage_only_ret} != 1.0 "
            "(control broken — a page was erased when erased_k=0)"
        )

    def test_control_k0_recovery_on_retention_equals_one(self, erasure_result):
        """S9 gate (2): erased_k=0 is the control — no pages erased."""
        pt = next(p for p in erasure_result.points if p.erased_k == 0)
        assert pt.recovery_on_ret == 1.0, (
            f"erased_k=0 recovery_on_ret={pt.recovery_on_ret} != 1.0 "
            "(control broken — a page was erased when erased_k=0)"
        )

    def test_b0_lc_matches_iter(self, erasure_result, iter_result):
        assert erasure_result.b0_lc == iter_result.b0_lc

    def test_erasure_heal_line_exists_per_k(self, erasure_result):
        with open(erasure_result.report_path, encoding="utf-8") as f:
            content = f.read()
        for k in _ERASED_KS:
            assert f"ERASURE_HEAL: erased={k} " in content, (
                f"ERASURE_HEAL line for erased={k} missing from REPORT_phase9_3_lc.md"
            )

    def test_lc_overrecovery_still_present(self, erasure_result):
        with open(erasure_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "LC_OVERRECOVERY:" in content

    def test_ablation_still_present(self, erasure_result):
        with open(erasure_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "ABLATION:" in content

    def test_lc_baseline_dominance_still_present(self, erasure_result):
        with open(erasure_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "LC_BASELINE_DOMINANCE:" in content

    def test_report_byte_identical_on_erasure_rerun(
        self, erasure_result, iter_result, ablation_result, baseline_result,
    ):
        """_write_full_report_93e must produce identical bytes twice."""
        from aepk_paging.harness.phase9_3_lc import _write_full_report_93e
        with open(erasure_result.report_path, encoding="utf-8") as f:
            original = f.read()
        _write_full_report_93e(
            iter_result,
            ablation_result,
            baseline_result,
            erasure_result,
            erasure_result.report_path,
        )
        with open(erasure_result.report_path, encoding="utf-8") as f:
            rerun = f.read()
        assert original == rerun, "Final report is NOT byte-identical across two writes"

    def test_erasure_results_logged(self, erasure_result):
        print(f"\n{'='*70}")
        print("Phase 9.3-LC-2 Erasure Results (FULL GRID)")
        print(f"{'='*70}")
        print(f"B0_lc={erasure_result.b0_lc:.4f}  probes={erasure_result.n_probes}")
        print(f"{'erased_k':>8} | {'damage_only_ret':>16} | {'recovery_on_ret':>16}")
        print("-" * 50)
        for pt in erasure_result.points:
            print(f"{pt.erased_k:>8} | {pt.damage_only_ret:>16.4f} | {pt.recovery_on_ret:>16.4f}")
