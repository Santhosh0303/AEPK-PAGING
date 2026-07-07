"""CPU tests for Phase 10 step 22 persisted-cache store/heal demo (no model).

Exercises deterministic serialize/deserialize round-trip, on-disk corruption determinism, the
deployable stored-scalar detector, bit-exact erasure heal, and report line-exists. No GPU value
is hard-coded.
"""

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.coding import encode_rs_erasure_group, recover_rs_erasure
from aepk_paging.harness.phase9_cw import corrupt_k_scale, FINGERPRINTS, calibrate
from aepk_paging.harness.phase10_persist import (
    fingerprint_scalars, save_group, load_group, corrupt_stored_page_k,
    stored_scalar_flag, pages_byte_identical, write_persist_report, NUM_PARITY,
)


def _pages(n=4, T=8, H=2, D=4):
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        K = rng.normal(size=(T, H, D)).astype(np.float32)
        V = rng.normal(size=(T, H, D)).astype(np.float32)
        out.append(KVPage(i, i, (0, T), K, V, "real_fp16", 1.0))
    return out


def _saved(tmp_path, pages, target=0):
    grp = encode_rs_erasure_group(pages, NUM_PARITY)
    fps = fingerprint_scalars(pages[target])
    tau = calibrate(pages).tau
    p = str(tmp_path / "g.npz")
    save_group(p, pages, grp, fps, tau, target)
    return p, grp, fps, tau


def test_save_load_roundtrip_byte_identical(tmp_path):
    pages = _pages()
    p, grp, fps, tau = _saved(tmp_path, pages)
    lpages, lgroup, lfps, ltau, tidx = load_group(p)
    assert all(pages_byte_identical(a, b) for a, b in zip(pages, lpages))
    assert np.array_equal(grp.parity_bytes, lgroup.parity_bytes)
    assert tidx == 0 and lfps.keys() == fps.keys()


def test_corrupt_stored_page_determinism_and_locality(tmp_path):
    pages = _pages()
    p, *_ = _saved(tmp_path, pages)
    corrupt_stored_page_k(p, 0, 2.0)
    l1, *_ = load_group(p)
    # target K scaled x2; siblings untouched
    assert np.allclose(l1[0].K, pages[0].K * 2.0)
    assert all(np.array_equal(l1[i].K, pages[i].K) for i in range(1, len(pages)))
    # determinism: fresh save+corrupt yields identical bytes
    p2, *_ = _saved(tmp_path, pages)
    corrupt_stored_page_k(p2, 0, 2.0)
    l2, *_ = load_group(p2)
    assert np.array_equal(l1[0].K, l2[0].K)


def test_stored_scalar_flag_detects_and_clears(tmp_path):
    pages = _pages()
    tau = calibrate(pages).tau
    clean = pages[0]
    stored = fingerprint_scalars(clean)
    # a clean page matches its own stored scalars -> not flagged
    assert stored_scalar_flag(clean, stored, tau) is False
    # k_scale=2.0 moves key_norm_mean far beyond tau -> flagged
    assert stored_scalar_flag(corrupt_k_scale(clean, 2.0), stored, tau) is True


def test_heal_from_parity_bit_exact():
    pages = _pages()
    grp = encode_rs_erasure_group(pages, NUM_PARITY)
    rec = recover_rs_erasure(grp, [pages[0].page_id])
    healed = rec[pages[0].page_id]
    assert pages_byte_identical(healed, pages[0])          # RS erasure recovery is bit-exact


def test_fingerprint_scalars_match_fingerprint_set():
    fps = fingerprint_scalars(_pages()[0])
    assert set(fps.keys()) == set(FINGERPRINTS.keys())
    assert all(isinstance(v, float) for v in fps.values())


def test_persist_report_line_exists(tmp_path):
    res = {"n_cc": 24, "clean_acc": 1.0, "roundtrip_exact": True, "detected": True,
           "detection_rate": 1.0, "healed_exact": True, "baseline_acc": 0.2, "healed_acc": 1.0}
    p = tmp_path / "persist.md"
    write_persist_report(res, path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "PERSIST_HEAL: roundtrip_exact=" in text
