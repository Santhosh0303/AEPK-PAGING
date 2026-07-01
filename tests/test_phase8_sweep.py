"""
Phase 8.2 acceptance test: quant_noise sweep across all 6 levels.

Gate:
  - sweep_result has a SweepPoint for each noise_level
  - overall_verdict in ("PASS", "FAIL") — harness-computed, not hardcoded
  - REPORT_phase8_sweep.md written and contains "SWEEP VERDICT"
  - crossover_level reported (None if no crossover — honesty spine)
  - does NOT assert verdict == "PASS" (gate may FAIL honestly)
"""

import os
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.harness.phase8_sweep import (
    NOISE_LEVELS,
    NLL_THRESHOLD,
    SweepResult,
    SweepPoint,
    run_sweep,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "REPORT_phase8_sweep.md")


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
    return run_sweep(model, tok, DEVICE, DTYPE, noise_levels=NOISE_LEVELS)


class TestSweepStructure:
    def test_returns_sweep_result(self, sweep_result):
        assert isinstance(sweep_result, SweepResult)

    def test_all_noise_levels_present(self, sweep_result):
        levels = [sp.noise_level for sp in sweep_result.sweep_points]
        assert levels == NOISE_LEVELS, f"Expected {NOISE_LEVELS}, got {levels}"

    def test_sweep_points_are_frozen(self, sweep_result):
        for sp in sweep_result.sweep_points:
            assert isinstance(sp, SweepPoint)

    def test_overall_verdict_is_valid(self, sweep_result):
        assert sweep_result.overall_verdict in ("PASS", "FAIL"), (
            f"verdict must be PASS or FAIL, got {sweep_result.overall_verdict!r}"
        )

    def test_pareto_frontier_subset_of_levels(self, sweep_result):
        for level in sweep_result.pareto_frontier:
            assert level in NOISE_LEVELS

    def test_crossover_is_max_pareto(self, sweep_result):
        if sweep_result.crossover_level is not None:
            assert sweep_result.crossover_level == max(sweep_result.pareto_frontier)
        else:
            assert sweep_result.pareto_frontier == []


class TestSweepValues:
    def test_b0_nll_consistent_across_levels(self, sweep_result):
        """B0 is clean KV — NLL should be identical for all points."""
        nlls = [sp.b0_nll for sp in sweep_result.sweep_points]
        assert max(nlls) - min(nlls) < 1e-4, f"B0 NLL varied: {nlls}"

    def test_b0_accuracy_consistent(self, sweep_result):
        accs = [sp.b0_accuracy for sp in sweep_result.sweep_points]
        assert max(accs) - min(accs) < 1e-6, f"B0 accuracy varied: {accs}"

    def test_nll_delta_definition(self, sweep_result):
        for sp in sweep_result.sweep_points:
            assert abs(sp.nll_delta - (sp.b3_nll - sp.b0_nll)) < 1e-6

    def test_acc_delta_definition(self, sweep_result):
        for sp in sweep_result.sweep_points:
            assert abs(sp.acc_delta - (sp.b3_accuracy - sp.b0_accuracy)) < 1e-6

    def test_pareto_definition_correct(self, sweep_result):
        """on_pareto iff ΔNLL ≤ threshold AND b3 saves storage."""
        for sp in sweep_result.sweep_points:
            expected = sp.nll_delta <= NLL_THRESHOLD and sp.b3_storage_bits < sp.b0_storage_bits
            assert sp.on_pareto == expected, (
                f"noise={sp.noise_level}: on_pareto={sp.on_pareto} but "
                f"nll_delta={sp.nll_delta:.4f} threshold={NLL_THRESHOLD} "
                f"storage_diff={sp.b3_storage_bits - sp.b0_storage_bits}"
            )

    def test_accuracy_in_range(self, sweep_result):
        for sp in sweep_result.sweep_points:
            assert 0.0 <= sp.b0_accuracy <= 1.0
            assert 0.0 <= sp.b3_accuracy <= 1.0

    def test_nll_nonnegative(self, sweep_result):
        for sp in sweep_result.sweep_points:
            assert sp.b0_nll >= 0.0
            assert sp.b3_nll >= 0.0


class TestReport:
    def test_report_file_written(self, sweep_result):
        assert os.path.exists(sweep_result.report_path), (
            f"REPORT_phase8_sweep.md not written to {sweep_result.report_path}"
        )

    def test_report_contains_verdict_line(self, sweep_result):
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "PHASE 8 SWEEP VERDICT:" in content

    def test_report_contains_pareto_section(self, sweep_result):
        with open(sweep_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "Pareto frontier" in content
        assert "Crossover level" in content

    def test_sweep_results_logged(self, sweep_result):
        print(f"\n{'='*60}")
        print(f"Phase 8.2 Sweep Results")
        print(f"{'='*60}")
        print(f"{'noise':>6} | {'B0_NLL':>7} | {'B3_NLL':>7} | {'dNLL':>7} | "
              f"{'B0_acc':>7} | {'B3_acc':>7} | {'dacc':>7} | {'savings%':>9} | Pareto")
        print("-" * 80)
        for sp in sweep_result.sweep_points:
            print(
                f"{sp.noise_level:>6.2f} | {sp.b0_nll:>7.4f} | {sp.b3_nll:>7.4f} | "
                f"{sp.nll_delta:>+7.4f} | {sp.b0_accuracy:>7.3f} | {sp.b3_accuracy:>7.3f} | "
                f"{sp.acc_delta:>+7.3f} | {sp.storage_savings_pct:>+8.1f}% | "
                f"{'YES' if sp.on_pareto else 'no'}"
            )
        print(f"\nPareto frontier: {sweep_result.pareto_frontier}")
        print(f"Crossover level: {sweep_result.crossover_level}")
        print(f"Overall verdict: {sweep_result.overall_verdict}")
