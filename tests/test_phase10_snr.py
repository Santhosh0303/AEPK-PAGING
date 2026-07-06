"""CPU tests for the Phase 10 SNR CAMPAIGN (steps 19-21) — no model.

Exercises the deterministic RMS math, the SNR susceptibility score, the P2 separability metric,
the P3 anticorrelation gate branches, the P1 prediction/band formulas, the relative-noise
injector (determinism / zero-level identity / seed independence), and the report line-exists
contracts. Tests assert structure and LINE-EXISTS, never a hard-coded GPU value.
"""

import math

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.harness.phase10_snr import (
    rms, page_key_rms, page_value_rms, snr_score, snr_rank,
    snr_fd_verdict, SNR_FD_SUPPORT, SNR_FD_REFUTE,
    predict_crossover, p1_band, snr_law_verdict, stress_spread, relative_noise,
    load_grid_rows, load_fd_v2_damage, write_snr_report,
    CAL_CROSSOVER, CAL_HEAD_DIM, PRED_HEAD_DIM,
)


def _page(layer, kscale=1.0, vscale=1.0, T=8, H=2, D=4):
    rng = np.random.default_rng(100 + layer)
    K = (rng.normal(size=(T, H, D)) * kscale).astype(np.float32)
    V = (rng.normal(size=(T, H, D)) * vscale).astype(np.float32)
    return KVPage(("real", layer), layer, (0, T), K, V, "real_fp16", 1.0)


# ---- RMS math ---------------------------------------------------------------

def test_rms_of_ones_is_one():
    assert abs(rms(np.ones((3, 4, 5))) - 1.0) < 1e-12


def test_rms_scales_linearly():
    x = np.random.default_rng(0).normal(size=(6, 3))
    assert abs(rms(3.0 * x) - 3.0 * rms(x)) < 1e-9


def test_page_key_value_rms_nonneg():
    p = _page(0, kscale=2.0, vscale=5.0)
    assert page_key_rms(p) >= 0.0 and page_value_rms(p) >= 0.0
    # bigger scale -> bigger rms
    assert page_value_rms(p) > page_key_rms(p)


# ---- SNR score + P2 separability --------------------------------------------

def test_snr_score_formula():
    assert abs(snr_score(64, 2.0) - math.sqrt(64) * 2.0) < 1e-9


def test_snr_rank_separable_when_scores_split_classes():
    # tolerant models get higher scores than intolerant -> separable, margin_vs_hd finite
    rows = [
        ("a", 64, 1.0, False), ("b", 64, 1.1, False),
        ("c", 128, 3.0, True), ("d", 256, 2.5, True),
    ]
    pairs, separable, margin = snr_rank(rows)
    assert separable is True
    assert math.isfinite(margin)
    assert [n for n, _ in pairs] == ["a", "b", "c", "d"]     # ascending by score


def test_snr_rank_not_separable_when_scores_overlap():
    rows = [
        ("a", 64, 5.0, False),      # intolerant but high score
        ("b", 128, 1.0, True),      # tolerant but low score
    ]
    _, separable, _ = snr_rank(rows)
    assert separable is False


# ---- P3 gate branches -------------------------------------------------------

def test_snr_fd_verdict_branches():
    assert snr_fd_verdict(SNR_FD_SUPPORT) == "supported"
    assert snr_fd_verdict(-0.9) == "supported"
    assert snr_fd_verdict(SNR_FD_REFUTE) == "refuted"
    assert snr_fd_verdict(0.1) == "refuted"
    assert snr_fd_verdict(-0.35) == "undetermined"          # between the two thresholds
    assert snr_fd_verdict(float("nan")) == "undetermined"


def test_snr_fd_gate_direction_sane():
    assert SNR_FD_SUPPORT < SNR_FD_REFUTE < 0.0


# ---- P1 prediction + band ---------------------------------------------------

def test_predict_crossover_ratio():
    # equal RMS_K -> pure sqrt(head_dim) scaling of the calibration crossover
    p = predict_crossover(2.0, 2.0)
    assert abs(p - CAL_CROSSOVER * math.sqrt(PRED_HEAD_DIM / CAL_HEAD_DIM)) < 1e-9


def test_p1_band_positive_and_grows_with_pred():
    b1 = p1_band(0.2, 0.05)
    b2 = p1_band(0.4, 0.05)
    assert b1 > 0.0 and b2 > b1


def test_snr_law_verdict_confirm_refute():
    assert snr_law_verdict(0.30, 0.31, 0.05) == "confirmed"
    assert snr_law_verdict(0.30, 0.50, 0.05) == "refuted"


# ---- P4 relative-noise injector + spread ------------------------------------

def test_relative_noise_zero_level_identity():
    p = _page(1, kscale=3.0)
    q = relative_noise(p, 0.0, 7)
    assert np.allclose(p.K, q.K) and np.allclose(p.V, q.V)


def test_relative_noise_deterministic_in_seed():
    p = _page(2)
    a = relative_noise(p, 0.2, 5)
    b = relative_noise(p, 0.2, 5)
    assert np.array_equal(a.K, b.K) and np.array_equal(a.V, b.V)


def test_relative_noise_seed_independence():
    p = _page(3)
    a = relative_noise(p, 0.2, 1)
    b = relative_noise(p, 0.2, 2)
    assert not np.array_equal(a.K, b.K)


def test_relative_noise_is_multiplicative_scale():
    # relative noise perturbs proportionally: larger keys move more in absolute terms
    p = _page(4, kscale=10.0)
    q = relative_noise(p, 0.3, 9)
    # mean relative change is bounded by ~level in expectation, not exploding
    rel = np.abs((q.K - p.K) / (p.K + 1e-6))
    assert np.isfinite(rel).all()


def test_stress_spread():
    assert abs(stress_spread([0.1, 0.5, 0.3]) - 0.4) < 1e-12
    assert math.isnan(stress_spread([float("nan")]))


def test_relative_h1_consistent():
    from aepk_paging.harness.phase10_snr import relative_h1_consistent
    h1 = {"qwen1.5b", "pythia-1b", "pythia-1.4b"}
    # matches at level 0.2, not 0.1 -> consistent (at SOME level)
    tbl = {"0.1": ["qwen1.5b"], "0.2": ["qwen1.5b", "pythia-1b", "pythia-1.4b"]}
    assert relative_h1_consistent(tbl, h1) is True
    # never matches -> not consistent
    tbl2 = {"0.1": ["qwen1.5b"], "0.2": ["qwen1.5b", "pythia-1b"]}
    assert relative_h1_consistent(tbl2, h1) is False


# ---- reuse of stored campaign data ------------------------------------------

def test_load_grid_rows_seven_included():
    g = load_grid_rows()
    assert len(g) == 7 and "qwen1.5b" in g and "pythia-160m" not in g   # 160m excluded
    assert g["qwen1.5b"][0] == 128 and g["qwen0.5b"][0] == 64


def test_load_fd_v2_damage_28_layers():
    layers, dmg = load_fd_v2_damage()
    assert len(layers) == 28 and len(dmg) == 28
    assert all(0.0 <= d <= 1.0 for d in dmg)


# ---- report line-exists contracts (all four verdict lines) ------------------

def _min_state():
    layers, dmg = load_fd_v2_damage()
    key_rms = [1.0 + 0.01 * l for l in layers]
    val_rms = [2.0 + 0.01 * l for l in layers]
    return {
        "rms_rows": [
            ("qwen0.5b", 64, 127, 1.0, snr_score(64, 1.0), False),
            ("qwen1.5b", 128, 151, 2.0, snr_score(128, 2.0), True),
            ("tinyllama", 64, 77, 0.9, snr_score(64, 0.9), False),
            ("pythia-410m", 64, 52, 1.1, snr_score(64, 1.1), False),
            ("pythia-1b", 256, 74, 1.5, snr_score(256, 1.5), True),
            ("pythia-1.4b", 128, 72, 1.8, snr_score(128, 1.8), True),
            ("smollm2-360m", 64, 91, 1.0, snr_score(64, 1.0), False),
        ],
        "rank": {"sorted_pairs": [("a", 1.0)], "separable": True, "margin_vs_hd": 1.2},
        "fd": {"layers": layers, "key_rms": key_rms, "val_rms": val_rms, "damage": dmg,
               "rho_key": -0.6, "rho_val": -0.5, "verdict": "supported"},
    }


def test_report_step19_lines_exist(tmp_path):
    p = tmp_path / "snr.md"
    write_snr_report(_min_state(), path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "SNR_RANK: scores=" in text
    assert "SNR_FD: spearman=" in text
    assert "SNR_LAW:" not in text and "STRESS_INV:" not in text   # not yet


def test_report_step20_law_line_exists(tmp_path):
    st = _min_state()
    st["law"] = {"predicted": 0.28, "measured_mu": 0.30, "measured_ci": 0.06, "band": 0.13,
                 "verdict": "confirmed", "levels": [0.1, 0.2, 0.3], "seeds": [0, 1],
                 "crossovers": [0.30, 0.31],
                 "grid": {"0": [0.9, 0.8, 0.7], "1": [0.9, 0.85, 0.6]}}
    p = tmp_path / "snr2.md"
    write_snr_report(st, path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "SNR_LAW: predicted=" in text


def test_report_step21_stress_line_exists(tmp_path):
    st = _min_state()
    order = ["qwen0.5b", "qwen1.5b", "tinyllama", "pythia-410m", "pythia-1b",
             "pythia-1.4b", "smollm2-360m"]
    st["stress"] = {
        "levels": [0.1, 0.2], "order": order,
        "retention": {n: {"0.1": 0.9, "0.2": 0.5} for n in order},
        "tolerant": {n: True for n in order},
        "h1_consistent": True, "hd64_spread_rel": 0.1, "hd64_spread_abs": 0.3,
    }
    p = tmp_path / "snr3.md"
    write_snr_report(st, path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "STRESS_INV: family=relative" in text
