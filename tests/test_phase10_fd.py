"""CPU tests for Phase 10 step (7) FENCED fluctuation-dissipation analogue (no model).

Exercises the deterministic clean-fluctuation statistics, the Spearman rank correlation, and
the supported/refuted/undetermined verdict logic. Tests assert structure/monotonicity, never a
hard-coded GPU rho.
"""

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.harness.phase10_fd import (
    key_norm_variance, value_norm_variance, spearman_rho, fd_verdict,
    RHO_SUPPORT, RHO_NULL,
    control_layer_ids, pick_control_level, write_fd_report_v2,
    CONTROL_LEVELS, CONTROL_DAMAGE_MIN,
)


def _page(layer, kscale, T=8, H=2, D=4):
    rng = np.random.default_rng(layer)
    K = (rng.normal(size=(T, H, D)) * kscale).astype(np.float32)
    V = rng.normal(size=(T, H, D)).astype(np.float32)
    return KVPage(("real", layer), layer, (0, T), K, V, "real_fp16", 1.0)


def test_key_norm_variance_scales_with_spread():
    # larger key scale -> larger per-token norm spread -> larger variance
    lo = key_norm_variance(_page(0, 1.0))
    hi = key_norm_variance(_page(0, 5.0))
    assert hi > lo >= 0.0


def test_value_norm_variance_nonnegative():
    assert value_norm_variance(_page(3, 2.0)) >= 0.0


def test_spearman_perfect_and_anti():
    assert abs(spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1.0) < 1e-9
    assert abs(spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) + 1.0) < 1e-9


def test_spearman_constant_is_nan():
    import math
    assert math.isnan(spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]))


def test_fd_verdict_thresholds():
    assert fd_verdict(RHO_SUPPORT) == "supported"
    assert fd_verdict(0.95) == "supported"
    assert fd_verdict(0.0) == "refuted"                    # |rho|<RHO_NULL -> null
    assert fd_verdict(RHO_NULL - 0.01) == "refuted"
    assert fd_verdict(0.45) == "undetermined"              # weak positive
    assert fd_verdict(-0.8) == "undetermined"              # wrong sign vs fixed direction
    assert fd_verdict(float("nan")) == "undetermined"


def test_constants_sane():
    assert 0.0 < RHO_NULL < RHO_SUPPORT <= 1.0
    assert tuple(CONTROL_LEVELS) == (0.5, 1.0, 2.0) and 0.0 < CONTROL_DAMAGE_MIN < 1.0


# --- PREREG v2: positive-control gate ----------------------------------------

def test_control_layer_ids_first_mid_last():
    assert control_layer_ids(28) == [0, 14, 27]
    assert control_layer_ids(3) == [0, 1, 2]


def test_pick_control_level_smallest_passing():
    rows = [(0.5, 0, 0.02), (0.5, 14, 0.05), (0.5, 27, 0.01),
            (1.0, 0, 0.20), (1.0, 14, 0.10), (1.0, 27, 0.05)]
    assert pick_control_level(rows) == 1.0          # 0.5 flat, 1.0 max=0.20 >= 0.15


def test_pick_control_level_none_when_flat():
    rows = [(lv, ly, 0.01) for lv in CONTROL_LEVELS for ly in (0, 14, 27)]
    assert pick_control_level(rows) is None


def test_fd_v2_report_no_response_verdict(tmp_path):
    res = dict(layers=list(range(4)), kvar=np.ones(4), vvar=np.ones(4), damage=None,
               n_cc=40, control_rows=[(lv, ly, 0.0) for lv in CONTROL_LEVELS
                                      for ly in (0, 2, 3)], chosen_level=None)
    p = tmp_path / "fd2.md"
    write_fd_report_v2(res, path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "FD: spearman=" in text                          # line exists
    assert "verdict=undetermined(no-response-regime)" in text


def test_fd_v2_report_with_sweep(tmp_path):
    dmg = np.array([0.1, 0.2, 0.3, 0.4])
    res = dict(layers=list(range(4)), kvar=np.array([1.0, 2.0, 3.0, 4.0]),
               vvar=np.ones(4), damage=dmg, n_cc=40,
               control_rows=[(0.5, 0, 0.20)], chosen_level=0.5)
    p = tmp_path / "fd2b.md"
    write_fd_report_v2(res, path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "FD: spearman=" in text and "chosen_level=0.5" in text
