"""Pre-Phase-7 adversarial hardening: lock the vulnerabilities found by stress testing.

V1/V3 non-finite KV -> detected as corruption (not silently accepted).
V2 non-finite attention_mass -> rejected at construction (fail-fast).
V4 budget < min-recoverable-storage -> recoverability overrides budget, honestly reported.
Plus coding edge cases (k=1 group, oversized group, 1-element page).
"""

from __future__ import annotations

import numpy as np
import pytest

from aepk_paging.coding import (
    ReedSolomonCode,
    encode_rs_erasure_group,
    recover_rs_erasure,
)
from aepk_paging.detect import finiteness_detector
from aepk_paging.kv_page import KVPage, ResidencyTier
from aepk_paging.residency import ResidencyManager


def _page(pid: int, mass: float = 0.5) -> KVPage:
    return KVPage(
        page_id=pid,
        layer=0,
        token_range=(pid * 2, pid * 2 + 2),
        K=np.ones((2, 2), dtype=np.float32),
        V=np.ones((2, 2), dtype=np.float32),
        precision_tag="f32",
        attention_mass=mass,
    )


# --- V2: non-finite attention_mass rejected at construction ---
@pytest.mark.parametrize("bad_mass", [float("nan"), float("inf"), -1.0])
def test_kvpage_rejects_nonfinite_or_negative_mass(bad_mass: float) -> None:
    with pytest.raises(ValueError):
        _page(0, mass=bad_mass)


# --- V1/V3: non-finite KV detected as corruption ---
def test_finiteness_detector_flags_nan_and_inf() -> None:
    clean = _page(0)
    nan_k = KVPage(
        page_id=1, layer=0, token_range=(0, 2),
        K=np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32),
        V=np.ones((2, 2), dtype=np.float32), precision_tag="f32", attention_mass=1.0,
    )
    inf_v = KVPage(
        page_id=2, layer=0, token_range=(0, 2),
        K=np.ones((2, 2), dtype=np.float32),
        V=np.array([[np.inf, 1.0], [2.0, 3.0]], dtype=np.float32), precision_tag="f32", attention_mass=1.0,
    )

    assert not finiteness_detector(clean).flag
    assert finiteness_detector(nan_k).flag and finiteness_detector(nan_k).deviation == 1.0
    assert finiteness_detector(inf_v).flag and finiteness_detector(inf_v).deviation == 1.0


# --- V4: recoverability overrides budget, and storage is reported truthfully ---
def test_budget_zero_keeps_recoverable_and_reports_true_storage() -> None:
    pages = [_page(i) for i in range(4)]
    plan = ResidencyManager().plan(pages, budget_bits=0, erasure_recovery_bound=1)

    evicted = sum(1 for d in plan.decisions.values() if d.tier is ResidencyTier.EVICTED)
    assert evicted <= 1  # never evict beyond what 1 parity block can rebuild
    # budget was 0 but recoverability forced CODED pages; storage is honestly reported > budget
    assert plan.total_storage_bits > 0


# --- coding edge cases ---
def test_rs_erasure_single_page_group_recovers() -> None:
    group = encode_rs_erasure_group([_page(0)], num_parity=1)
    recovered = recover_rs_erasure(group, missing_page_ids=[0])
    assert np.array_equal(recovered[0].K, _page(0).K)


def test_rs_erasure_rejects_oversized_group() -> None:
    pages = [_page(i) for i in range(2)]
    with pytest.raises(ValueError):
        encode_rs_erasure_group(pages, num_parity=300)  # k + r > 255 over GF(2^8)


def test_rs_error_round_trips_single_element_page() -> None:
    code = ReedSolomonCode(t=1)
    values = np.array([[5]], dtype=np.int8)
    recovered, _ = code.correct_array(code.encode_array(values))
    assert np.array_equal(recovered, values)
