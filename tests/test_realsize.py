"""Improvement #7: exercise the full pipeline at realistic KV page shapes.

The earlier tests used toy pages (4x4 ... 16x8). A real KV page is
``(block_tokens, kv_heads * head_dim)`` per layer. These shapes (Qwen2.5-1.5B-ish:
2 GQA KV heads x 128 head_dim = 256 wide, 16-token blocks) prove the detection,
coding, and residency layers behave at scale, not just on toys.
"""

from __future__ import annotations

import numpy as np

from aepk_paging.coding import (
    ReedSolomonCode,
    ReedSolomonCodewords,
    UncorrectableError,
    encode_rs_erasure_group,
    recover_rs_erasure,
)
from aepk_paging.detect import (
    attention_mass,
    attention_mass_detector,
    norm_consistency_detector,
    norm_ratio,
)
from aepk_paging.kv_page import KVPage, PageTable, ResidencyTier
from aepk_paging.lossy_tier import quant_noise, quantize_page
from aepk_paging.residency import ResidencyManager

KV_HEADS = 2
HEAD_DIM = 128
BLOCK_TOKENS = 16
WIDTH = KV_HEADS * HEAD_DIM  # 256


def realistic_page(page_id: int, mass: float | None = None) -> KVPage:
    rng = np.random.default_rng(900 + page_id)
    K = rng.normal(0.0, 1.0, size=(BLOCK_TOKENS, WIDTH)).astype(np.float32)
    V = rng.normal(0.0, 1.0, size=(BLOCK_TOKENS, WIDTH)).astype(np.float32)
    base = KVPage(
        page_id=page_id,
        layer=0,
        token_range=(page_id * BLOCK_TOKENS, page_id * BLOCK_TOKENS + BLOCK_TOKENS),
        K=K,
        V=V,
        precision_tag="float32",
        attention_mass=0.0,
    )
    resolved = attention_mass(base) if mass is None else mass
    return KVPage(
        page_id=base.page_id,
        layer=base.layer,
        token_range=base.token_range,
        K=K,
        V=V,
        precision_tag="float32",
        attention_mass=resolved,
    )


def test_exact_round_trip_at_realistic_scale() -> None:
    page = realistic_page(0)
    table = PageTable()
    table.store(page, tier=ResidencyTier.RESIDENT)
    fetched = table.fetch(0)

    assert fetched.K.shape == (BLOCK_TOKENS, WIDTH)
    assert np.array_equal(fetched.K, page.K)
    assert np.array_equal(fetched.V, page.V)


def test_rs_erasure_recovers_realistic_pages_bit_exact() -> None:
    pages = [realistic_page(i) for i in range(4)]
    group = encode_rs_erasure_group(pages, num_parity=2)

    recovered = recover_rs_erasure(group, missing_page_ids=[1, 3])  # 2 = bound

    for pid in (1, 3):
        assert np.array_equal(recovered[pid].K, pages[pid].K)
        assert np.array_equal(recovered[pid].V, pages[pid].V)
    # beyond bound still fails loud at scale
    import pytest

    with pytest.raises(UncorrectableError):
        recover_rs_erasure(group, missing_page_ids=[0, 1, 2])


def test_rs_error_correction_on_realistic_page() -> None:
    values = quantize_page(realistic_page(0), bit_width=8).K.values  # (16, 256) int8
    code = ReedSolomonCode(t=4)
    enc = code.encode_array(values)
    cw = enc.codewords.copy()
    cw[0, [3, 77, 150, 222]] ^= np.array([0x11, 0x22, 0x44, 0x88], dtype=np.uint8)  # 4 <= t

    recovered, n_errors = code.correct_array(
        ReedSolomonCodewords(cw, enc.original_len, enc.shape, enc.dtype)
    )

    assert recovered.shape == values.shape
    assert np.array_equal(recovered, values)
    assert n_errors >= 1


def test_physics_detectors_flag_realistic_corruption() -> None:
    page = realistic_page(0)
    corrupted, _ = quant_noise(page, level=0.8, seed=17)

    mass = attention_mass_detector(corrupted, expected_mass=page.attention_mass, tolerance=0.01)
    norm = norm_consistency_detector(corrupted, expected_ratio=norm_ratio(page), tolerance=0.01)

    assert mass.flag and mass.deviation > mass.tolerance
    assert norm.flag and norm.deviation > norm.tolerance


def test_residency_plans_over_realistic_pages() -> None:
    pages = [realistic_page(i, mass=m) for i, m in enumerate([0.1, 0.4, 0.7, 0.95])]
    manager = ResidencyManager()
    budget = manager.cost_model.coded_bits(pages[0]) * 3
    plan = manager.plan(pages, budget_bits=budget)

    # highest-mass page is never at a lower tier than the lowest-mass page
    ranks = {ResidencyTier.EVICTED: 0, ResidencyTier.CODED: 1, ResidencyTier.RESIDENT: 2}
    assert ranks[plan.decisions[3].tier] >= ranks[plan.decisions[0].tier]
