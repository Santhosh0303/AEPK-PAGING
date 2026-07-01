"""
Phase 9.1 acceptance test: task-accuracy axis with N>=100 probes.

Gate (honesty spine S9 — must not be violated):
  - REPORT_phase9_accuracy.md is written
  - report contains the verdict line "ACCURACY_AXIS:"
  - retention is a harness-computed float (NOT asserted to equal any value)
  - retention in a sane range [0.0, 2.0] — allows edge-case b0_acc~0
  - n_probes >= 100
  - does NOT assert verdict == "PASS" or retention == any number
  - does NOT assert crossover == any value (may be None — honest)
"""

import os
import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.harness.phase9_accuracy import (
    NOISE_LEVELS,
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
        probes = build_extended_eval_set()
        assert len(probes) >= 100, f"Expected >=100 probes, got {len(probes)}"

    def test_all_probes_have_required_fields(self):
        for p in build_extended_eval_set():
            assert "prompt" in p and p["prompt"]
            assert "expected" in p and p["expected"]


class TestSweepStructure:
    def test_returns_phase9_result(self, sweep_result):
        assert isinstance(sweep_result, Phase9AccuracyResult)

    def test_n_probes_at_least_100(self, sweep_result):
        assert sweep_result.n_probes >= 100, (
            f"Expected >=100 probes, got {sweep_result.n_probes}"
        )

    def test_all_noise_levels_present(self, sweep_result):
        actual = [pt.noise_level for pt in sweep_result.points]
        assert actual == NOISE_LEVELS, f"Expected {NOISE_LEVELS}, got {actual}"

    def test_points_are_accuracy_points(self, sweep_result):
        for pt in sweep_result.points:
            assert isinstance(pt, AccuracyPoint)


class TestAccuracyValues:
    def test_b0_accuracy_in_range(self, sweep_result):
        for pt in sweep_result.points:
            assert 0.0 <= pt.b0_accuracy <= 1.0, (
                f"b0_accuracy={pt.b0_accuracy} at noise={pt.noise_level}"
            )

    def test_b3_accuracy_in_range(self, sweep_result):
        for pt in sweep_result.points:
            assert 0.0 <= pt.b3_accuracy <= 1.0

    def test_retention_in_sane_range(self, sweep_result):
        """retention must be a computed float — NOT asserted to equal any specific value."""
        for pt in sweep_result.points:
            assert isinstance(pt.retention, float), (
                f"retention must be float, got {type(pt.retention)}"
            )
            assert 0.0 <= pt.retention <= 2.0, (
                f"retention={pt.retention} out of sane range at noise={pt.noise_level}"
            )

    def test_retention_definition(self, sweep_result):
        """retention = b3_acc / b0_acc (or 1.0 if b0_acc==0)."""
        for pt in sweep_result.points:
            if pt.b0_accuracy > 0.0:
                expected_r = pt.b3_accuracy / pt.b0_accuracy
                assert abs(pt.retention - expected_r) < 1e-9, (
                    f"retention={pt.retention} but b3/b0={expected_r}"
                )

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
            assert abs(pt.acc_delta - (pt.b3_accuracy - pt.b0_accuracy)) < 1e-6


class TestCrossover:
    def test_crossover_is_none_or_valid_noise_level(self, sweep_result):
        if sweep_result.crossover_noise is not None:
            assert sweep_result.crossover_noise in NOISE_LEVELS

    def test_crossover_retention_is_none_or_float(self, sweep_result):
        if sweep_result.retention_at_crossover is not None:
            assert isinstance(sweep_result.retention_at_crossover, float)
            assert sweep_result.retention_at_crossover >= RETENTION_CROSSOVER_THRESHOLD


class TestReport:
    def test_report_file_written(self, sweep_result):
        assert os.path.exists(sweep_result.report_path), (
            f"Report not written: {sweep_result.report_path}"
        )

    def test_report_contains_accuracy_axis_line(self, sweep_result):
        """The verdict line must exist — its VALUE is never asserted here."""
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "ACCURACY_AXIS:" in content, (
            "Report must contain 'ACCURACY_AXIS:' verdict line"
        )

    def test_report_verdict_line_has_retention_and_crossover(self, sweep_result):
        """Verdict line must encode a numeric retention and a crossover value."""
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        # Pattern: ACCURACY_AXIS: retention=<value> at crossover=<value>
        match = re.search(
            r"ACCURACY_AXIS:\s+retention=(\S+)\s+at\s+crossover=(\S+)",
            content,
        )
        assert match is not None, (
            "ACCURACY_AXIS line must match 'retention=<x> at crossover=<y>'"
        )
        ret_str = match.group(1)
        # retention must be parseable as float — never checked for a specific value
        try:
            float(ret_str)
        except ValueError:
            pytest.fail(f"retention='{ret_str}' is not a float")

    def test_report_contains_nll_column(self, sweep_result):
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "ΔNLL" in content or "nll" in content.lower()

    def test_report_contains_divergence_note(self, sweep_result):
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "diverge" in content.lower() or "NLL" in content

    def test_results_logged(self, sweep_result):
        print(f"\n{'='*70}")
        print("Phase 9.1 Task-Accuracy Axis Results")
        print(f"{'='*70}")
        print(f"Total probes: {sweep_result.n_probes}")
        print(
            f"{'noise':>6} | {'B0_acc':>7} | {'B3_acc':>7} | {'Δacc':>7} | "
            f"{'retention':>9} | {'ΔNLL':>8}"
        )
        print("-" * 60)
        for pt in sweep_result.points:
            print(
                f"{pt.noise_level:>6.2f} | {pt.b0_accuracy:>7.3f} | "
                f"{pt.b3_accuracy:>7.3f} | {pt.acc_delta:>+7.3f} | "
                f"{pt.retention:>9.4f} | {pt.nll_delta:>+8.4f}"
            )
        print(f"\nCrossover noise: {sweep_result.crossover_noise}")
        print(f"Retention at crossover: {sweep_result.retention_at_crossover}")
