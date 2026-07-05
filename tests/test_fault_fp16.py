"""CPU unit tests for the Phase 10.2 raw-fp16 bit-upset injector.

Verifies (before any GPU sweep): n_flips=0 bit-exact no-op; exact flip count; determinism;
region correctness (exponent flip only touches exponent bits); tensor selection; guards.
"""

import numpy as np
import pytest

from aepk_paging.kv_page import KVPage
from aepk_paging.harness.fault_fp16 import bitflip_fp16, _REGION_BITS


def _page():
    rng = np.random.default_rng(0)
    # fp16-representable clean baseline (mimics real-model KV that came from fp16)
    K = rng.normal(size=(4, 2, 8)).astype(np.float16).astype(np.float32)
    V = rng.normal(size=(4, 2, 8)).astype(np.float16).astype(np.float32)
    return KVPage("p0", 0, (0, 4), K, V, "fp16", 1.0)


def _uint16(arr):
    return np.asarray(arr, np.float32).astype(np.float16).reshape(-1).view(np.uint16)


def test_zero_flips_is_bit_exact_noop():
    p = _page()
    out = bitflip_fp16(p, 0, "exponent", seed=1, tensor="K")
    assert np.array_equal(out.K, p.K)
    assert np.array_equal(out.V, p.V)


def test_exact_flip_count_in_bits():
    p = _page()
    for n in (1, 3, 5):
        out = bitflip_fp16(p, n, "mantissa", seed=7, tensor="K")
        before = np.unpackbits(_uint16(p.K).view(np.uint8))
        after = np.unpackbits(_uint16(out.K).view(np.uint8))
        assert int((before != after).sum()) == n
        assert np.array_equal(out.V, p.V)  # V untouched when tensor='K'


def test_determinism_same_seed():
    # exponent flips can legitimately produce NaN/Inf (exponent -> all-ones), so compare
    # raw fp16 BIT PATTERNS, not float values (NaN != NaN would falsely flag differences).
    p = _page()
    a = bitflip_fp16(p, 3, "exponent", seed=42, tensor="V")
    b = bitflip_fp16(p, 3, "exponent", seed=42, tensor="V")
    assert np.array_equal(_uint16(a.V), _uint16(b.V))


def test_region_confinement_exponent():
    p = _page()
    out = bitflip_fp16(p, 5, "exponent", seed=3, tensor="K")
    before = _uint16(p.K)
    after = _uint16(out.K)
    diff = np.bitwise_xor(before, after)
    exp_mask = np.uint16(sum(1 << b for b in _REGION_BITS["exponent"]))
    # every changed bit lies within the exponent mask
    assert int(np.bitwise_and(diff, ~exp_mask & np.uint16(0xFFFF)).sum()) == 0
    assert int((diff != 0).sum()) == 5


def test_tensor_v_selection():
    p = _page()
    out = bitflip_fp16(p, 2, "sign", seed=9, tensor="V")
    assert np.array_equal(out.K, p.K)
    assert not np.array_equal(out.V, p.V)


def test_guards():
    p = _page()
    with pytest.raises(ValueError):
        bitflip_fp16(p, 1, "bogus", seed=0)
    with pytest.raises(ValueError):
        bitflip_fp16(p, -1, "sign", seed=0)
    with pytest.raises(ValueError):
        bitflip_fp16(p, 10_000, "sign", seed=0)  # exceeds element count
    with pytest.raises(ValueError):
        bitflip_fp16(p, 1, "sign", seed=0, tensor="Q")
