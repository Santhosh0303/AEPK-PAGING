"""CPU tests for Phase 10.3 live-heal GPU-free parts: erasure-group heal wiring is
bit-exact, and the LIVE_HEAL verdict line is emitted (assert LINE EXISTS, never value)."""

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.coding import encode_rs_erasure_group, recover_rs_erasure
from aepk_paging.harness.phase10_liveheal import write_liveheal_report, NUM_PARITY


def _pages(n=4):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        K = rng.normal(size=(5, 2, 8)).astype(np.float16).astype(np.float32)
        V = rng.normal(size=(5, 2, 8)).astype(np.float16).astype(np.float32)
        out.append(KVPage(("real", i), i, (0, 5), K, V, "real_fp16", 1.0 + i))
    return out


def test_erasure_group_heal_bit_exact():
    pages = _pages()
    group = encode_rs_erasure_group(pages, NUM_PARITY)
    tgt = pages[0]
    rec = recover_rs_erasure(group, [tgt.page_id])
    healed = rec[tgt.page_id]
    assert np.array_equal(healed.K, tgt.K)      # bit-exact restore from parity + survivors
    assert np.array_equal(healed.V, tgt.V)


def test_liveheal_verdict_line_exists(tmp_path):
    rows = [(1.0, 1.0, 1.0, 0.0, True, "erasure"),     # control (identity)
            (2.0, 0.5, 1.0, 1.0, True, "erasure"),
            (4.0, 0.375, 1.0, 1.0, True, "erasure")]
    path = tmp_path / "lh.md"
    recovered = write_liveheal_report(1.0, rows, path=str(path))
    text = path.read_text(encoding="utf-8")
    assert "LIVE_HEAL: baseline_acc=" in text
    assert "decode_mode=erasure" in text
    assert "(CONTROL)" in text                       # explicit fault=0 control row present
    assert isinstance(recovered, bool)
    assert "HEAL_CONTROL:" not in text               # no control rows passed -> no line


def test_heal_control_line_exists(tmp_path):
    # PREREG v2 control arm: HEAL_CONTROL line emitted when control rows are passed.
    rows = [(1.0, 1.0, 1.0, 0.0, True, "erasure"),
            (2.0, 0.5, 1.0, 1.0, True, "erasure"),
            (4.0, 0.375, 1.0, 1.0, True, "erasure")]
    control_rows = [(2.0, 1.0), (4.0, 0.875)]
    path = tmp_path / "lh_ctl.md"
    write_liveheal_report(1.0, rows, path=str(path), control_rows=control_rows)
    text = path.read_text(encoding="utf-8")
    assert "HEAL_CONTROL: top_baseline=" in text     # line exists; value never asserted
    assert "selection_load_bearing=" in text
    assert "## Control arm (PREREG v2)" in text
