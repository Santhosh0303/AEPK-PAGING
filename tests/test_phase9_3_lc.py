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

Fixture scope: module (model loaded once; reduced grid runs once).
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
    LONG_CONTEXT_PASSAGE,
    LC_NOISE_LEVELS,
    assert_token_lengths,
    build_lc_probe_set,
    run_phase9_3a,
    run_phase9_3c,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Reduced grid for fast iteration — not the full grid
_ITER_NOISE = [0.0, 0.2, 0.3]
_ITER_SEEDS = 2
_ITER_PROBES = LC_N_PROBES_ITER   # 10


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
    """Reduced-grid result: 10 probes, 2 seeds, noise=[0.0, 0.2, 0.3]."""
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
        print("Phase 9.3a Long-Context Results (REDUCED GRID)")
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
        print("Phase 9.3c Ablation Results (REDUCED GRID)")
        print(f"{'='*70}")
        print(
            f"Summary: coding={ablation_result.ablation_summary['coding']:+.4f}  "
            f"physics={ablation_result.ablation_summary['physics']:+.4f}  "
            f"detect={ablation_result.ablation_summary['detect']:+.4f}"
        )
        print(f"{'noise':>6} | {'Δcod':>7} | {'Δphy':>7} | {'Δdet':>7}")
        print("-" * 40)
        for pt in ablation_result.points:
            print(
                f"{pt.noise_level:>6.2f} | {pt.coding_delta:>+7.4f} | "
                f"{pt.physics_delta:>+7.4f} | {pt.detect_delta:>+7.4f}"
            )
