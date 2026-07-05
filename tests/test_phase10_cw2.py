"""CPU tests for Phase 10.2 CW-2 GPU-free portions: token-row needle corruption logic
and the CW2_BITFLIP verdict-line emission (assert LINE EXISTS, never its value)."""

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.harness.phase10_cw2 import (
    _corrupt_token_rows, write_cw2_bitflip_report, _apply_to_rows, write_cw2_needle_report,
)
from aepk_paging.harness.phase9_cw import calibrate


def _page():
    rng = np.random.default_rng(1)
    K = rng.normal(size=(6, 2, 8)).astype(np.float16).astype(np.float32)
    V = rng.normal(size=(6, 2, 8)).astype(np.float16).astype(np.float32)
    return KVPage(("real", 0), 0, (0, 6), K, V, "real_fp16", 1.0)


def test_needle_corrupts_only_target_rows():
    p = _page()
    out = _corrupt_token_rows(p, [2, 4], n_flips=3, region="exponent", seed=5, tensor="K")
    changed = np.any(out.K != p.K, axis=(1, 2))  # per-token: did any element change?
    assert set(np.nonzero(changed)[0].tolist()) <= {2, 4}   # only needle rows may change
    assert np.array_equal(out.V, p.V)                        # V untouched (tensor='K')
    # untouched rows are bit-exact
    for r in (0, 1, 3, 5):
        assert np.array_equal(out.K[r], p.K[r])


def test_needle_determinism_bits():
    p = _page()
    a = _corrupt_token_rows(p, [1, 3], 2, "mantissa", 9, "V")
    b = _corrupt_token_rows(p, [1, 3], 2, "mantissa", 9, "V")
    au = a.V.astype(np.float16).view(np.uint16)
    bu = b.V.astype(np.float16).view(np.uint16)
    assert np.array_equal(au, bu)


def test_cw2_verdict_line_exists(tmp_path):
    # synthetic rows (no GPU); the test asserts the harness EMITS the CW2_BITFLIP line,
    # never a specific value (honesty spine).
    rows = [("exponent", 1, 1, "K", -0.5, +3.0, False, 0.9, False, 0.0),
            ("mantissa", 5, 3, "V", +0.0, +0.0, True, 0.1, False, 0.0),
            ("exponent", 5, 3, "K", -0.9, -0.4, False, 1.0, False, 0.5)]  # nonfinite -> blind False
    calib = calibrate([_page(), _page()])
    path = tmp_path / "rep.md"
    verdict, n_cw = write_cw2_bitflip_report(0.9, 0.4, 0.2, calib, rows, path=str(path))
    text = path.read_text(encoding="utf-8")
    assert "CW2_BITFLIP: confident_wrong_cells=" in text
    assert f"of {len(rows)}" in text
    assert verdict in ("SHOWN", "NOT_SHOWN")


def test_apply_to_rows_only_touches_rows():
    p = _page()
    out = _apply_to_rows(p, [0, 5], lambda sub, s: sub, seed=0)  # identity corr -> no change
    assert np.array_equal(out.K, p.K) and np.array_equal(out.V, p.V)
    from aepk_paging.harness.fault_fp16 import bitflip_fp16
    out2 = _apply_to_rows(p, [2], lambda sub, s: bitflip_fp16(sub, 1, "sign", s, "K"), seed=3)
    changed = np.any(out2.K != p.K, axis=(1, 2))
    assert set(np.nonzero(changed)[0].tolist()) <= {2}


def test_cw2_needle_verdict_line_exists(tmp_path):
    rows = [("bitflip_exp_n1", -0.3, +2.0, False, 0.5, 0.5, False),
            ("quant_noise_0.3", -0.1, +0.5, False, 0.2, 0.0, False)]
    calib = calibrate([_page(), _page()])
    path = tmp_path / "n.md"
    verdict, n_cw = write_cw2_needle_report(0.9, 0.4, 0.2, 160, calib, rows, path=str(path))
    text = path.read_text(encoding="utf-8")
    assert "CW2_NEEDLE: confident_wrong_cells=" in text
    assert verdict in ("SHOWN", "NOT_SHOWN")
