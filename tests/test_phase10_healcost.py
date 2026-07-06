"""CPU tests for Phase 10 step (18) — heal-cost microbenchmark math (no model, no GPU).

Assert the deterministic median/IQR/overhead accounting and the tolerance gate + report line
on SYNTHETIC timings and synthetic KV pages. The real Qwen timing run is separate (GPU).
"""

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.harness.phase10_healcost import (
    median_iqr, parity_overhead_bytes, overhead_pct, within_tolerance,
    write_healcost_report, GROUP_SIZE, NUM_PARITY, MEDIAN_TOL,
)


def _pages(n=GROUP_SIZE, shape=(4, 2, 8), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        K = rng.standard_normal(shape).astype(np.float16)
        V = rng.standard_normal(shape).astype(np.float16)
        out.append(KVPage(page_id=f"p{i}", layer=i, token_range=(0, shape[0]),
                          K=K, V=V, precision_tag="fp16", attention_mass=1.0))
    return out


def test_median_iqr_known():
    med, iqr = median_iqr([1, 2, 3, 4, 5])
    assert med == 3.0
    # P25=2, P75=4 -> IQR=2
    assert abs(iqr - 2.0) < 1e-9


def test_median_iqr_empty():
    med, iqr = median_iqr([])
    assert med != med and iqr != iqr          # nan, nan


def test_parity_overhead_exact_match():
    # measured parity bytes MUST equal the analytic num_parity * per-page row length.
    pages = _pages()
    measured, analytic = parity_overhead_bytes(pages, NUM_PARITY)
    assert measured == analytic
    # per-page row = K.nbytes + V.nbytes; num_parity=1 -> one parity row.
    row_bytes = pages[0].K.nbytes + pages[0].V.nbytes
    assert measured == NUM_PARITY * row_bytes


def test_overhead_pct_is_parity_over_group():
    # equal-size sibling pages -> overhead = num_parity/group_size * 100 (25% at 4/1).
    pages = _pages()
    pct = overhead_pct(pages, NUM_PARITY)
    assert abs(pct - 100.0 * NUM_PARITY / GROUP_SIZE) < 1e-9


def test_within_tolerance_boundary():
    assert within_tolerance(1.0, 1.20) is True          # exactly +20%
    assert within_tolerance(1.0, 1.2001) is False       # just over
    assert within_tolerance(1.0, 0.80) is True          # exactly -20%
    assert within_tolerance(1.0, 0.79) is False
    assert within_tolerance(0.0, 0.0) is True           # zero-median fallback
    assert within_tolerance(float("nan"), 1.0) is False # nan guard


def test_healcost_report_line_exists(tmp_path):
    # synthetic two-run dicts; run2 medians within tolerance of run1.
    def run(scale):
        return {
            "encode": (0.10 * scale, 0.01),
            "heal": (0.20 * scale, 0.02),
            "recompute": (5.00 * scale, 0.50),
            "fingerprint": (0.05 * scale, 0.005),
            "parity_bytes_measured": 256,
            "parity_bytes_analytic": 256,
            "parity_overhead_pct": 25.0,
            "group_page_shape": (4, 2, 8),
        }
    r1, r2 = run(1.0), run(1.1)                          # +10% -> within 20%
    p = tmp_path / "rep.md"
    heal_ms, rec_ms, ratio, ov, all_within, bytes_exact = write_healcost_report(
        r1, r2, path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "HEAL_COST: heal_ms=" in text and "parity_overhead_pct=" in text
    assert "MICROBENCHMARKS" in text                     # scope caveat present
    assert all_within is True and bytes_exact is True
    assert abs(ratio - rec_ms / heal_ms) < 1e-9
    assert ov == 25.0


def test_healcost_report_flags_tolerance_breach(tmp_path):
    base = {
        "encode": (0.10, 0.01), "heal": (0.20, 0.02),
        "recompute": (5.00, 0.5), "fingerprint": (0.05, 0.005),
        "parity_bytes_measured": 256, "parity_bytes_analytic": 256,
        "parity_overhead_pct": 25.0, "group_page_shape": (4, 2, 8),
    }
    breached = dict(base); breached["recompute"] = (10.0, 0.5)   # +100% -> out of tol
    p = tmp_path / "rep.md"
    *_, all_within, bytes_exact = write_healcost_report(base, breached, path=str(p))
    assert all_within is False        # the breach is reported, not hidden
    assert bytes_exact is True
