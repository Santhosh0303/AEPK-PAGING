"""
Phase 8.3 acceptance test: adaptive per-layer precision sweep.

Gate:
  - adaptive_noise_levels respects budget constraint (mean == global_level)
  - adaptive_noise_levels assigns LESS noise to higher-attention_mass layers
  - adaptive frontier reported (even if narrower than uniform — honest)
  - comparison_verdict in ("ADAPTIVE_BETTER", "SAME", "ADAPTIVE_WORSE")
  - does NOT assert ADAPTIVE_BETTER (honesty spine — gate may FAIL)
"""

import os
import pytest
import numpy as np

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.harness.phase8_sweep import NOISE_LEVELS
from aepk_paging.harness.phase8_adaptive import (
    adaptive_noise_levels,
    run_adaptive_sweep,
    AdaptiveSweepResult,
)
from aepk_paging.real_model_adapter import dynamiccache_to_pages
from aepk_paging.kv_page import KVPage
import numpy as np

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Uniform frontier from Phase 8.2 (used for delta comparison)
UNIFORM_FRONTIER = [0.0, 0.05, 0.1, 0.2]


class TestAdaptiveNoiseLevels:
    """Unit tests — no model load needed."""

    def _make_pages(self, masses: list[float]) -> list[KVPage]:
        return [
            KVPage(
                page_id=i, layer=i, token_range=(0, 4),
                K=np.ones((4, 2, 8), dtype=np.float32),
                V=np.ones((4, 2, 8), dtype=np.float32),
                precision_tag="fp32", attention_mass=m,
            )
            for i, m in enumerate(masses)
        ]

    def test_budget_constraint_mean_equals_global(self):
        """mean(per_levels) must equal global_level."""
        pages = self._make_pages([1.0, 2.0, 3.0, 4.0, 5.0])
        for global_lvl in [0.05, 0.1, 0.3, 0.5]:
            levels = adaptive_noise_levels(pages, global_lvl)
            assert abs(np.mean(levels) - global_lvl) < 1e-6, (
                f"Budget violated: mean={np.mean(levels):.6f} != {global_lvl}"
            )

    def test_high_mass_gets_less_noise(self):
        """Layer with highest attention_mass must get the lowest noise level."""
        pages = self._make_pages([1.0, 2.0, 5.0, 3.0, 0.5])
        levels = adaptive_noise_levels(pages, 0.3)
        highest_mass_idx = int(np.argmax([p.attention_mass for p in pages]))
        lowest_noise_idx = int(np.argmin(levels))
        assert highest_mass_idx == lowest_noise_idx, (
            f"Highest mass at idx={highest_mass_idx} but lowest noise at idx={lowest_noise_idx}"
        )

    def test_low_mass_gets_more_noise(self):
        pages = self._make_pages([1.0, 2.0, 5.0, 3.0, 0.5])
        levels = adaptive_noise_levels(pages, 0.3)
        lowest_mass_idx = int(np.argmin([p.attention_mass for p in pages]))
        highest_noise_idx = int(np.argmax(levels))
        assert lowest_mass_idx == highest_noise_idx

    def test_uniform_masses_returns_uniform_levels(self):
        pages = self._make_pages([2.0, 2.0, 2.0])
        levels = adaptive_noise_levels(pages, 0.3)
        assert all(abs(l - 0.3) < 1e-6 for l in levels)

    def test_zero_global_level_returns_zeros(self):
        pages = self._make_pages([1.0, 3.0, 2.0])
        levels = adaptive_noise_levels(pages, 0.0)
        assert all(l == 0.0 for l in levels)

    def test_all_levels_nonnegative(self):
        pages = self._make_pages([0.01, 5.0, 2.0, 3.0, 1.0])
        levels = adaptive_noise_levels(pages, 0.3)
        assert all(l >= 0.0 for l in levels)


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
    model, tok = model_and_tok
    return run_adaptive_sweep(
        model, tok, DEVICE, DTYPE,
        uniform_frontier=UNIFORM_FRONTIER,
        noise_levels=NOISE_LEVELS,
    )


class TestAdaptiveSweep:
    def test_returns_adaptive_sweep_result(self, adaptive_result):
        assert isinstance(adaptive_result, AdaptiveSweepResult)

    def test_all_noise_levels_present(self, adaptive_result):
        levels = [sp.noise_level for sp in adaptive_result.adaptive_sweep_points]
        assert levels == NOISE_LEVELS

    def test_comparison_verdict_valid(self, adaptive_result):
        assert adaptive_result.comparison_verdict in (
            "ADAPTIVE_BETTER", "SAME", "ADAPTIVE_WORSE"
        ), f"Got: {adaptive_result.comparison_verdict!r}"

    def test_frontier_delta_definition(self, adaptive_result):
        expected = len(adaptive_result.adaptive_frontier) - len(adaptive_result.uniform_frontier)
        assert adaptive_result.frontier_delta == expected

    def test_adaptive_frontier_subset_of_levels(self, adaptive_result):
        for level in adaptive_result.adaptive_frontier:
            assert level in NOISE_LEVELS

    def test_report_written(self, adaptive_result):
        assert os.path.exists(adaptive_result.report_path)

    def test_report_contains_verdict(self, adaptive_result):
        with open(adaptive_result.report_path, encoding="utf-8") as f:
            content = f.read()
        assert "PHASE 8.3 COMPARISON VERDICT:" in content

    def test_pareto_definition_correct(self, adaptive_result):
        from aepk_paging.harness.phase8_sweep import NLL_THRESHOLD
        for sp in adaptive_result.adaptive_sweep_points:
            expected = (
                sp.nll_delta <= NLL_THRESHOLD and
                sp.b3_storage_bits < sp.b0_storage_bits
            )
            assert sp.on_pareto == expected

    def test_results_logged(self, adaptive_result):
        print(f"\nPhase 8.3 Adaptive Sweep Results")
        print(f"Uniform  frontier: {adaptive_result.uniform_frontier}")
        print(f"Adaptive frontier: {adaptive_result.adaptive_frontier}")
        print(f"Frontier delta: {adaptive_result.frontier_delta:+d}")
        print(f"Comparison verdict: {adaptive_result.comparison_verdict}")
        print(f"\nnoise | dNLL_unif | dNLL_adap | adap_acc | Pareto")
        from aepk_paging.harness.phase8_sweep import NOISE_LEVELS
        for sp in adaptive_result.adaptive_sweep_points:
            print(f"{sp.noise_level:.2f} | - | {sp.nll_delta:+.4f} | {sp.b3_accuracy:.3f} | {'YES' if sp.on_pareto else 'no'}")
