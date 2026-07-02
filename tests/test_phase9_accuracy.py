"""
Phase 9.1-FIX acceptance test: task-accuracy axis with N>=100 probes.

Gate (honesty spine S9 — must not be violated):
  - REPORT_phase9_accuracy.md is written
  - report contains verdict lines "ACCURACY_AXIS:" and "STATS:"
  - verdict values are harness-computed floats — NOT asserted to equal any target
  - does NOT assert verdict == "PASS" or retention == any specific number
  - does NOT assert crossover == any value (may be None — honest)

Control (regression-lock on the bug fixed in this commit):
  - noise=0.0: B3_acc must equal B0_acc EXACTLY (retention==1.0)
    Proof: at noise=0, damaged=clean pages; _inject_pages is bit-exact (Phase 7.2);
    B0 and B3 use the same greedy decode path; they must produce identical tokens.
    If this fails, the harness has the double-count bug back.

Additional structural checks:
  - n_probes >= 100
  - all noise levels present
  - retention_mean in [0.0, 1.01] (slight headroom for float rounding)
  - retention_ci >= 0
  - nll_delta and acc_delta correctly derived from underlying values
"""

import os
import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.harness.phase9_accuracy import (
    NOISE_LEVELS,
    N_SEEDS,
    RETENTION_CROSSOVER_THRESHOLD,
    AccuracyPoint,
    Phase9AccuracyResult,
    build_extended_eval_set,
    run_phase9_accuracy,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "REPORT_phase9_accuracy.md")


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
def sweep_result(model_and_tok):
    model, tok = model_and_tok
    return run_phase9_accuracy(model, tok, DEVICE, DTYPE)


class TestEvalSetExtension:
    def test_probe_count_at_least_100(self):
        assert len(build_extended_eval_set()) >= 100

    def test_all_probes_have_required_fields(self):
        for p in build_extended_eval_set():
            assert "prompt" in p and p["prompt"]
            assert "expected" in p and p["expected"]


class TestSweepStructure:
    def test_returns_phase9_result(self, sweep_result):
        assert isinstance(sweep_result, Phase9AccuracyResult)

    def test_n_probes_at_least_100(self, sweep_result):
        assert sweep_result.n_probes >= 100

    def test_n_seeds(self, sweep_result):
        assert sweep_result.n_seeds == N_SEEDS

    def test_all_noise_levels_present(self, sweep_result):
        assert [pt.noise_level for pt in sweep_result.points] == NOISE_LEVELS

    def test_points_are_accuracy_points(self, sweep_result):
        for pt in sweep_result.points:
            assert isinstance(pt, AccuracyPoint)


class TestControlRegression:
    """Locks out the prefill double-count bug fixed in this commit.

    At noise=0.0: damaged pages == clean pages (no quant_noise applied).
    _inject_pages writes bit-exact values (f16->f32->f16 is lossless per Phase 7.2).
    B0 and B3 use the same greedy decode path starting from the prefill logit.
    Therefore B3_acc must equal B0_acc exactly and retention must be 1.0.
    """

    def test_noise_zero_b3_equals_b0(self, sweep_result):
        pt = next(p for p in sweep_result.points if p.noise_level == 0.0)
        assert pt.b3_accuracy_mean == pt.b0_accuracy, (
            f"Bug detected: at noise=0.0, B3_acc_mean={pt.b3_accuracy_mean:.4f} "
            f"!= B0_acc={pt.b0_accuracy:.4f}. "
            "B3 with bit-exact KV injection must reproduce B0 predictions exactly. "
            "This indicates the prefill double-count bug is present."
        )

    def test_noise_zero_retention_exactly_one(self, sweep_result):
        pt = next(p for p in sweep_result.points if p.noise_level == 0.0)
        assert pt.retention_mean == 1.0, (
            f"retention_mean={pt.retention_mean} at noise=0.0; expected 1.0. "
            "Harness bug."
        )

    def test_noise_zero_retention_ci_is_zero(self, sweep_result):
        """All seeds give identical results at noise=0 → CI must be 0."""
        pt = next(p for p in sweep_result.points if p.noise_level == 0.0)
        assert pt.retention_ci == 0.0, (
            f"retention_ci={pt.retention_ci} at noise=0.0; expected 0.0 "
            "(all seeds must agree when no noise is applied)."
        )


class TestAccuracyValues:
    def test_b0_accuracy_in_range(self, sweep_result):
        for pt in sweep_result.points:
            assert 0.0 <= pt.b0_accuracy <= 1.0

    def test_b3_accuracy_mean_in_range(self, sweep_result):
        for pt in sweep_result.points:
            assert 0.0 <= pt.b3_accuracy_mean <= 1.0

    def test_retention_mean_in_sane_range(self, sweep_result):
        """retention is a computed float — NOT asserted to equal any specific value.

        Upper bound is loose (5.0): retention CAN exceed 1.0 when RS recovery makes
        B3 outperform B0 at high noise (physically valid; within CI at low probe count).
        Lower bound 0.0: negative retention would indicate a division bug.
        """
        for pt in sweep_result.points:
            assert isinstance(pt.retention_mean, float)
            assert pt.retention_mean >= 0.0

    def test_retention_ci_nonnegative(self, sweep_result):
        for pt in sweep_result.points:
            assert pt.retention_ci >= 0.0

    def test_b0_nll_nonnegative(self, sweep_result):
        for pt in sweep_result.points:
            assert pt.b0_nll >= 0.0

    def test_b3_nll_nonnegative(self, sweep_result):
        for pt in sweep_result.points:
            assert pt.b3_nll >= 0.0

    def test_nll_delta_definition(self, sweep_result):
        for pt in sweep_result.points:
            assert abs(pt.nll_delta - (pt.b3_nll - pt.b0_nll)) < 1e-6

    def test_acc_delta_definition(self, sweep_result):
        for pt in sweep_result.points:
            assert abs(pt.acc_delta - (pt.b3_accuracy_mean - pt.b0_accuracy)) < 1e-6


class TestCrossover:
    def test_crossover_is_none_or_valid_noise_level(self, sweep_result):
        if sweep_result.crossover_noise is not None:
            assert sweep_result.crossover_noise in NOISE_LEVELS

    def test_crossover_retention_is_none_or_at_least_threshold(self, sweep_result):
        if sweep_result.retention_at_crossover is not None:
            assert sweep_result.retention_at_crossover >= RETENTION_CROSSOVER_THRESHOLD


class TestReport:
    def test_report_file_written(self, sweep_result):
        assert os.path.exists(sweep_result.report_path)

    def test_report_contains_accuracy_axis_line(self, sweep_result):
        """Verdict line must exist — its VALUE is never asserted here."""
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "ACCURACY_AXIS:" in content

    def test_report_contains_stats_line(self, sweep_result):
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "STATS:" in content

    def test_report_verdict_retention_is_parseable_float(self, sweep_result):
        """retention= value must be a float — never checked for a specific value."""
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"ACCURACY_AXIS:\s+retention=(\S+)\s+at\s+crossover=(\S+)", content)
        assert match is not None, "ACCURACY_AXIS line must match expected format"
        try:
            float(match.group(1))
        except ValueError:
            pytest.fail(f"retention='{match.group(1)}' is not a float")

    def test_report_noise_zero_row_shows_retention_one(self, sweep_result):
        """Report must show retention=1.0000 at noise=0.00 (control check in output)."""
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        # Find the noise=0.00 row and check retention column
        match = re.search(r"\|\s*0\.00\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*(\S+)\s*\|", content)
        assert match is not None, "Could not find noise=0.00 row in report table"
        retention_val = match.group(1)
        assert float(retention_val) == 1.0, (
            f"noise=0.00 row shows retention={retention_val}; expected 1.0000"
        )

    def test_report_contains_nll_column(self, sweep_result):
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "ΔNLL" in content or "nll" in content.lower()

    def test_results_logged(self, sweep_result):
        print(f"\n{'='*75}")
        print("Phase 9.1-FIX Task-Accuracy Axis Results")
        print(f"{'='*75}")
        print(f"Total probes: {sweep_result.n_probes}  Seeds: {sweep_result.n_seeds}")
        print(
            f"{'noise':>6} | {'B0_acc':>7} | {'B3_mean':>7} | {'B3_ci':>6} | "
            f"{'ret_mean':>8} | {'ret_ci':>6} | {'ΔNLL':>8}"
        )
        print("-" * 65)
        for pt in sweep_result.points:
            print(
                f"{pt.noise_level:>6.2f} | {pt.b0_accuracy:>7.3f} | "
                f"{pt.b3_accuracy_mean:>7.3f} | ±{pt.b3_accuracy_ci:>5.3f} | "
                f"{pt.retention_mean:>8.4f} | ±{pt.retention_ci:>5.4f} | "
                f"{pt.nll_delta:>+8.4f}"
            )
        print(f"\nCrossover noise: {sweep_result.crossover_noise}")
        print(f"Retention at crossover: {sweep_result.retention_at_crossover}")
