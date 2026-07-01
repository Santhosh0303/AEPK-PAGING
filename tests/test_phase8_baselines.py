"""
Phase 8.4 acceptance test: UQ + H2O baselines vs AEPK adaptive.

Gates:
  - 5 baseline points returned (UQ-8bit, UQ-4bit, H2O-25pct, H2O-50pct, H2O-75pct)
  - Each baseline has valid NLL (>0), accuracy in [0,1], storage_bits > 0
  - dominance dict covers all 5 baselines
  - overall_verdict in {"AEPK_DOMINATES_ALL", "AEPK_DOMINATES_SOME", "NONE"}
  - Report written and contains "PHASE 8.4 DOMINANCE VERDICT"
  - Does NOT assert AEPK_DOMINATES_ALL (honesty spine — verdict may be NONE)
  - UQ-4bit storage < UQ-8bit storage (sanity: fewer bits per element)
  - H2O-75pct storage < H2O-25pct storage (sanity: more eviction → less stored)
"""

import os
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.harness.phase8_adaptive import run_adaptive_sweep
from aepk_paging.harness.phase8_baselines import run_baselines, BaselinePoint, Phase8BaselinesResult
from aepk_paging.harness.phase8_sweep import NOISE_LEVELS

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Uniform frontier from Phase 8.2 (used only to initialise the adaptive sweep delta calc)
UNIFORM_FRONTIER = [0.0, 0.05, 0.1, 0.2]

EXPECTED_BASELINE_NAMES = {"UQ-8bit", "UQ-4bit", "H2O-25pct", "H2O-50pct", "H2O-75pct"}


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
def adaptive_result(model_and_tok):
    """Run Phase 8.3 adaptive sweep to get AEPK reference points."""
    model, tok = model_and_tok
    return run_adaptive_sweep(
        model, tok, DEVICE, DTYPE,
        uniform_frontier=UNIFORM_FRONTIER,
        noise_levels=NOISE_LEVELS,
    )


@pytest.fixture(scope="module")
def baselines_result(model_and_tok, adaptive_result):
    """Run Phase 8.4 baselines harness using Phase 8.3 adaptive points as reference."""
    model, tok = model_and_tok
    return run_baselines(
        model, tok, DEVICE, DTYPE,
        aepk_adaptive_points=adaptive_result.adaptive_sweep_points,
    )


class TestBaselinesStructure:
    def test_returns_phase8_baselines_result(self, baselines_result):
        assert isinstance(baselines_result, Phase8BaselinesResult)

    def test_five_baseline_points(self, baselines_result):
        assert len(baselines_result.baselines) == 5

    def test_expected_names_present(self, baselines_result):
        names = {b.name for b in baselines_result.baselines}
        assert names == EXPECTED_BASELINE_NAMES, f"Got names: {names}"

    def test_dominance_dict_covers_all(self, baselines_result):
        names = {b.name for b in baselines_result.baselines}
        assert set(baselines_result.dominance.keys()) == names

    def test_overall_verdict_valid(self, baselines_result):
        assert baselines_result.overall_verdict in (
            "AEPK_DOMINATES_ALL", "AEPK_DOMINATES_SOME", "NONE"
        ), f"Got: {baselines_result.overall_verdict!r}"


class TestBaselineValues:
    def test_all_nlls_positive(self, baselines_result):
        for b in baselines_result.baselines:
            assert b.nll > 0.0, f"{b.name}: NLL={b.nll}"

    def test_all_accuracies_in_range(self, baselines_result):
        for b in baselines_result.baselines:
            assert 0.0 <= b.accuracy <= 1.0, f"{b.name}: acc={b.accuracy}"

    def test_all_storage_positive(self, baselines_result):
        for b in baselines_result.baselines:
            assert b.storage_bits > 0, f"{b.name}: storage_bits={b.storage_bits}"

    def test_uq4_smaller_storage_than_uq8(self, baselines_result):
        """4-bit uses half the bits of 8-bit."""
        by_name = {b.name: b for b in baselines_result.baselines}
        assert by_name["UQ-4bit"].storage_bits < by_name["UQ-8bit"].storage_bits, (
            f"UQ-4bit={by_name['UQ-4bit'].storage_bits} should be < "
            f"UQ-8bit={by_name['UQ-8bit'].storage_bits}"
        )

    def test_h2o_more_eviction_less_storage(self, baselines_result):
        """75% eviction stores less than 25% eviction."""
        by_name = {b.name: b for b in baselines_result.baselines}
        assert by_name["H2O-75pct"].storage_bits < by_name["H2O-25pct"].storage_bits, (
            f"H2O-75pct={by_name['H2O-75pct'].storage_bits} should be < "
            f"H2O-25pct={by_name['H2O-25pct'].storage_bits}"
        )

    def test_storage_pct_matches_storage_bits(self, baselines_result):
        """storage_pct must be consistent with storage_bits relative to the clean baseline."""
        # UQ-8bit should be ~50% of fp16, UQ-4bit ~25%
        by_name = {b.name: b for b in baselines_result.baselines}
        uq8 = by_name["UQ-8bit"]
        uq4 = by_name["UQ-4bit"]
        assert abs(uq8.storage_pct - 0.5) < 0.01, f"UQ-8bit storage_pct={uq8.storage_pct:.3f} (expected ~0.5)"
        assert abs(uq4.storage_pct - 0.25) < 0.01, f"UQ-4bit storage_pct={uq4.storage_pct:.3f} (expected ~0.25)"


class TestDominanceLogic:
    def test_dominance_values_are_bool(self, baselines_result):
        for name, val in baselines_result.dominance.items():
            assert isinstance(val, bool), f"{name}: dominance value is {type(val)}"

    def test_dominance_count_matches_verdict(self, baselines_result):
        n_dominated = sum(baselines_result.dominance.values())
        n_total = len(baselines_result.baselines)
        if baselines_result.overall_verdict == "AEPK_DOMINATES_ALL":
            assert n_dominated == n_total
        elif baselines_result.overall_verdict == "AEPK_DOMINATES_SOME":
            assert 0 < n_dominated < n_total
        else:
            assert n_dominated == 0

    def test_aepk_points_present(self, baselines_result):
        """AEPK adaptive reference points must be non-empty (Phase 8.3 ran)."""
        assert len(baselines_result.aepk_points) > 0


class TestReport:
    def test_report_written(self, baselines_result):
        assert os.path.exists(baselines_result.report_path), (
            f"Report not at {baselines_result.report_path}"
        )

    def test_report_contains_verdict(self, baselines_result):
        with open(baselines_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "PHASE 8.4 DOMINANCE VERDICT:" in content

    def test_report_contains_all_baseline_names(self, baselines_result):
        with open(baselines_result.report_path, encoding="utf-8") as f:
            content = f.read()
        for name in EXPECTED_BASELINE_NAMES:
            assert name in content, f"Report missing baseline: {name}"

    def test_results_logged(self, baselines_result):
        print(f"\n{'='*65}")
        print(f"Phase 8.4 Baseline Comparison Results")
        print(f"{'='*65}")
        print(f"{'Method':<12} | {'NLL':>7} | {'Acc':>5} | {'Storage%':>9} | {'AEPK dom?':>10}")
        print("-" * 55)
        for b in baselines_result.baselines:
            dom = baselines_result.dominance[b.name]
            print(f"{b.name:<12} | {b.nll:>7.4f} | {b.accuracy:>5.3f} | "
                  f"{b.storage_pct*100:>8.1f}% | {'YES' if dom else 'no':>10}")
        print(f"\nAEPK reference points: {len(baselines_result.aepk_points)}")
        print(f"Overall verdict: {baselines_result.overall_verdict}")
        print(f"Report: {baselines_result.report_path}")
