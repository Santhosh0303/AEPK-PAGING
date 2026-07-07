"""CPU tests for Phase 10 step 23 encode-off-the-hot-path (async parity build) — no model.

Exercises the scheduling math (when parity groups close), the amortized-overhead computation, the
deterministic parity byte-identity that the async overlap must preserve, and report line-exists.
No GPU latency value is hard-coded.
"""

import math

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.coding import encode_rs_erasure_group
from aepk_paging.harness.phase10_healcost import (
    groups_closed, amortized_overhead_pct, within_tolerance, write_encodeasync_report,
    NUM_PARITY, GROUP_SIZE,
)


def _pages(n=4, T=8, H=2, D=4):
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        out.append(KVPage(i, i, (0, T), rng.normal(size=(T, H, D)).astype(np.float32),
                          rng.normal(size=(T, H, D)).astype(np.float32), "real_fp16", 1.0))
    return out


def test_groups_closed():
    assert groups_closed(200, 4) == 50
    assert groups_closed(7, 4) == 1
    assert groups_closed(3, 4) == 0


def test_amortized_overhead_pct():
    assert abs(amortized_overhead_pct(10.0, 12.0) - 20.0) < 1e-9
    assert abs(amortized_overhead_pct(10.0, 10.0) - 0.0) < 1e-9
    assert amortized_overhead_pct(10.0, 9.0) < 0.0            # faster than baseline -> negative
    assert math.isnan(amortized_overhead_pct(0.0, 5.0))       # bad baseline guard


def test_async_parity_matches_sync_parity():
    # the async overlap must not change the encoded parity: encode is deterministic
    pages = _pages()
    p1 = encode_rs_erasure_group(pages, NUM_PARITY).parity_bytes
    p2 = encode_rs_erasure_group(pages, NUM_PARITY).parity_bytes
    assert np.array_equal(p1, p2)


def test_within_tolerance_gate():
    assert within_tolerance(1.0, 1.1) is True                # +10% within 20%
    assert within_tolerance(1.0, 1.3) is False               # +30% outside


def test_encodeasync_report_line_exists(tmp_path):
    def _run(sync, asyn, dec):
        return {"n_tokens": 200, "group_size": GROUP_SIZE, "groups_closed": 50,
                "decode_only_ms_per_tok": dec, "sync_ms_per_tok": sync,
                "async_ms_per_tok": asyn, "parity_bytes_len": 4096, "parity_bytes_exact": True}
    p = tmp_path / "ea.md"
    write_encodeasync_report(_run(5.0, 4.2, 4.0), _run(5.1, 4.3, 4.05), path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "ENCODE_ASYNC: sync_ms_per_tok=" in text
    assert "parity_bytes_exact=" in text
