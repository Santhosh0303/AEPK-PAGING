"""CPU-only tests for the Phase 9-CW confident-wrong primitives.

Proves the corrected fingerprints, calibration, structured corruptions, and
real-logit entropy behave correctly on real-model-shaped 3D pages — the exact
place detect.py's attention_mass detector is degenerate (FLAW A/B, see
phase9_cw docstring). No GPU / model needed here."""

from __future__ import annotations

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.harness.phase9_cw import (
    FINGERPRINTS, CORRUPTIONS, calibrate, physics_flags, any_physics_flag,
    fp_key_mass, fp_key_norm_mean, fp_norm_ratio, token_entropy,
    corrupt_k_scale, corrupt_v_scale, corrupt_v_bias,
)


def _page(seed, pid=0, heads=2, dim=128, T=16):
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(T, heads, dim)).astype(np.float32)
    V = rng.normal(size=(T, heads, dim)).astype(np.float32)
    mean_norm = float(np.linalg.norm(K.reshape(T, -1), axis=1).mean())
    return KVPage(("real", pid), 0, (0, T), K, V, "real_fp16", mean_norm)


def _clean_set(n=8):
    return [_page(i, pid=i) for i in range(n)]


def test_key_mass_is_a_proper_distribution_scalar():
    # CORRECT fingerprint: key_mass is a softmax mass in (0, 1], NOT a (T,H*D) matrix
    p = _page(0)
    m = fp_key_mass(p)
    assert 0.0 < m <= 1.0
    # top-half mass >= half-uniform (heaviest tokens carry >= their share)
    assert m >= 0.5 - 1e-6


def test_clean_vs_clean_never_flags():
    calib = calibrate(_clean_set())
    p = _page(0)
    flags = physics_flags(p, p, calib)          # identical page -> zero deviation
    assert not any(flags.values())


def test_calibration_thresholds_positive():
    calib = calibrate(_clean_set())
    for name in FINGERPRINTS:
        assert calib.tau[name] > 0.0


def test_k_scale_caught_by_key_fingerprints():
    calib = calibrate(_clean_set())
    p = _page(0)
    c = corrupt_k_scale(p, 2.0)
    flags = physics_flags(p, c, calib)
    # scaling K moves key-norm mean strongly -> key_norm_mean must flag
    assert flags["key_norm_mean"] is True
    assert any_physics_flag(p, c, calib)


def test_v_scale_caught_by_norm_ratio():
    calib = calibrate(_clean_set())
    p = _page(0)
    c = corrupt_v_scale(p, 0.5)
    flags = physics_flags(p, c, calib)
    # halving V raises ||K||/||V|| -> norm_ratio must flag; key fingerprints blind
    assert flags["norm_ratio"] is True
    assert flags["key_norm_mean"] is False


def test_v_bias_is_caught_only_by_directional_fingerprint():
    # AUDIT FINDING: a coherent (norm-preserving-ish) V-bias is a BLIND SPOT for
    # norm-based fingerprints; only the directional DC-offset fingerprint sees it.
    calib = calibrate(_clean_set())
    p = _page(0)
    c = corrupt_v_bias(p, 2.0, seed=7)
    flags = physics_flags(p, c, calib)
    assert fp_key_mass(c) == fp_key_mass(p)      # K untouched -> key_mass identical
    assert flags["norm_ratio"] is False          # BLIND to coherent bias (small |ΔV|)
    assert flags["v_mean_shift"] is True         # directional fingerprint catches it
    assert any_physics_flag(p, c, calib)


def test_corruptions_preserve_shape_and_finiteness():
    p = _page(0)
    for name, fn in CORRUPTIONS.items():
        c = fn(p, 123)
        assert c.K.shape == p.K.shape and c.V.shape == p.V.shape
        assert np.all(np.isfinite(c.K)) and np.all(np.isfinite(c.V))


def test_token_entropy_orders_correctly():
    peaked = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    flat = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert token_entropy(peaked) < token_entropy(flat)
    assert abs(token_entropy(flat) - np.log(4)) < 1e-6
