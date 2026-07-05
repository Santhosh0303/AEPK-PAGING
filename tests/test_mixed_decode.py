"""CPU tests for the mixed error/erasure RS decode (Phase 10.3 core).

Proves the erasure-conversion 2x on the real galois library at symbol level:
  RS(255,255-2t) corrects t BLIND errors, but 2t LOCATED erasures — a 2x capacity gain —
  and fails loud (never silent) beyond 2e+s<=2t.
"""

import numpy as np
import pytest

from aepk_paging.coding import ReedSolomonCode, UncorrectableError
from aepk_paging.harness.mixed_decode import rs_mixed_correct


def _one_block_vals(k):
    # exactly k message bytes -> a single codeword (no padding ambiguity)
    return (np.arange(k) % 251).astype(np.uint8)


def test_2t_located_erasures_corrected():
    t = 3
    code = ReedSolomonCode(t=t)            # 2t = 6 parity symbols
    vals = _one_block_vals(code.k)
    cw = code.encode_array(vals)
    # corrupt 2t = 6 symbols, ALL located (fed as erasures)
    locs = [2, 10, 50, 99, 150, 200]
    assert len(locs) == 2 * t
    bad = cw.codewords.copy()
    for i in locs:
        bad[0, i] ^= 0x5A
    corrupted = cw.__class__(bad, cw.original_len, cw.shape, cw.dtype)
    res = rs_mixed_correct(code, corrupted, located_symbols=locs)
    assert np.array_equal(res.recovered, vals)          # 2t located -> fully healed
    assert res.n_located_erasures == 2 * t


def test_blind_only_capped_at_t():
    t = 3
    code = ReedSolomonCode(t=t)
    vals = _one_block_vals(code.k)
    cw = code.encode_array(vals)
    # t = 3 blind errors, NO locations supplied -> still correctable (2e+s = 6 <= 6)
    locs = [5, 60, 120]
    bad = cw.codewords.copy()
    for i in locs:
        bad[0, i] ^= 0x33
    corrupted = cw.__class__(bad, cw.original_len, cw.shape, cw.dtype)
    res = rs_mixed_correct(code, corrupted, located_symbols=[])
    assert np.array_equal(res.recovered, vals)
    assert res.n_blind_errors == t


def test_mixed_e_plus_s_within_bound():
    t = 3                                    # 2t = 6
    code = ReedSolomonCode(t=t)
    vals = _one_block_vals(code.k)
    cw = code.encode_array(vals)
    # 1 blind error + 4 located erasures: 2*1 + 4 = 6 <= 6 -> correctable
    blind = [7]; located = [20, 40, 80, 160]
    bad = cw.codewords.copy()
    for i in blind + located:
        bad[0, i] ^= 0x77
    corrupted = cw.__class__(bad, cw.original_len, cw.shape, cw.dtype)
    res = rs_mixed_correct(code, corrupted, located_symbols=located)
    assert np.array_equal(res.recovered, vals)


def test_beyond_bound_fails_loud():
    t = 3                                    # 2t = 6
    code = ReedSolomonCode(t=t)
    vals = _one_block_vals(code.k)
    cw = code.encode_array(vals)
    # 4 blind + 0 located: 2*4 = 8 > 6 -> must raise (fail-loud), never silent wrong return
    blind = [1, 2, 3, 4]
    bad = cw.codewords.copy()
    for i in blind:
        bad[0, i] ^= 0x11
    corrupted = cw.__class__(bad, cw.original_len, cw.shape, cw.dtype)
    with pytest.raises(UncorrectableError):
        rs_mixed_correct(code, corrupted, located_symbols=[])
