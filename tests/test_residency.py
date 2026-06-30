import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from aepk_paging.kv_page import KVPage, ResidencyTier
from aepk_paging.residency import (
    LANDAUER_KT_LN2,
    ResidencyManager,
    TierCostModel,
    page_attention_entropy,
)


def page(page_id: str, mass: float) -> KVPage:
    K = np.full((4, 4), mass, dtype=np.float32)
    V = np.full((4, 4), mass / 2.0, dtype=np.float32)
    return KVPage(
        page_id=page_id,
        layer=0,
        token_range=(int(page_id[1:]) * 4, int(page_id[1:]) * 4 + 4),
        K=K,
        V=V,
        precision_tag="float32",
        attention_mass=mass,
    )


def phase5_pages() -> list[KVPage]:
    return [
        page("p0", 0.1),
        page("p1", 0.3),
        page("p2", 0.6),
        page("p3", 0.9),
    ]


def tier_rank(tier: ResidencyTier) -> int:
    return {
        ResidencyTier.EVICTED: 0,
        ResidencyTier.CODED: 1,
        ResidencyTier.RESIDENT: 2,
    }[tier]


def test_residency_policy_is_deterministic_and_monotonic_in_attention_mass() -> None:
    manager = ResidencyManager()
    budget = manager.cost_model.coded_bits(phase5_pages()[0]) * 3
    baseline = manager.plan(phase5_pages(), budget_bits=budget)
    raised_pages = [page("p0", 1.0), *phase5_pages()[1:]]
    raised = manager.plan(raised_pages, budget_bits=budget)
    repeat = manager.plan(raised_pages, budget_bits=budget)

    assert baseline.decisions["p0"].tier is ResidencyTier.EVICTED
    assert tier_rank(raised.decisions["p0"].tier) >= tier_rank(baseline.decisions["p0"].tier)
    assert [decision.tier for decision in raised.decisions.values()] == [
        decision.tier for decision in repeat.decisions.values()
    ]
    assert raised.total_free_energy == repeat.total_free_energy


def test_tightening_budget_demotes_in_attention_mass_order() -> None:
    manager = ResidencyManager()
    pages = phase5_pages()
    coded_bits = manager.cost_model.coded_bits(pages[0])
    resident_bits = manager.cost_model.resident_bits(pages[0])
    fullish = manager.plan(pages, budget_bits=coded_bits * 4 + (resident_bits - coded_bits) * 2)
    middle = manager.plan(pages, budget_bits=coded_bits * 4 + (resident_bits - coded_bits))
    tight = manager.plan(pages, budget_bits=coded_bits * 3)

    assert fullish.decisions["p3"].tier is ResidencyTier.RESIDENT
    assert fullish.decisions["p2"].tier is ResidencyTier.RESIDENT
    assert middle.decisions["p3"].tier is ResidencyTier.RESIDENT
    assert middle.decisions["p2"].tier is ResidencyTier.CODED
    assert tight.decisions["p3"].tier is ResidencyTier.CODED
    assert tight.decisions["p2"].tier is ResidencyTier.CODED
    assert tight.decisions["p1"].tier is ResidencyTier.CODED
    assert tight.decisions["p0"].tier is ResidencyTier.EVICTED


def test_landauer_cost_counts_evicted_pages_and_settling_energy_never_increases() -> None:
    manager = ResidencyManager(cost_model=TierCostModel(temperature_kt=1.0))
    pages = phase5_pages()
    budget = manager.cost_model.coded_bits(pages[0]) * 2
    plan = manager.plan(pages, budget_bits=budget)
    evicted_bits = sum(
        manager.cost_model.erased_bits(item)
        for item in pages
        if plan.decisions[item.page_id].tier is ResidencyTier.EVICTED
    )

    assert plan.eviction_count == 1
    assert plan.landauer_cost == evicted_bits * LANDAUER_KT_LN2
    assert all(
        later <= earlier
        for earlier, later in zip(plan.settling_free_energy, plan.settling_free_energy[1:])
    )


def test_attention_entropy_rises_when_distribution_flattens() -> None:
    peaked = KVPage(
        page_id="peaked",
        layer=0,
        token_range=(0, 4),
        K=np.array(
            [
                [8.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        V=np.ones((4, 4), dtype=np.float32),
        precision_tag="float32",
        attention_mass=0.9,
    )
    flattened = KVPage(
        page_id="flat",
        layer=0,
        token_range=(4, 8),
        K=np.zeros((4, 4), dtype=np.float32),
        V=np.ones((4, 4), dtype=np.float32),
        precision_tag="float32",
        attention_mass=0.9,
    )

    assert page_attention_entropy(flattened) > page_attention_entropy(peaked)


def test_tier_ordering_is_monotonic_across_utility_weight_sweep() -> None:
    pages = phase5_pages()
    for scale in (0.5, 1.0, 1.5):
        model = TierCostModel(
            resident_utility_weight=2400.0 * scale,
            coded_utility_weight=900.0 * scale,
            coded_distortion_weight=160.0 * scale,
            eviction_distortion_weight=1400.0 * scale,
        )
        manager = ResidencyManager(cost_model=model)
        budget = model.coded_bits(pages[0]) * 4 + (model.resident_bits(pages[0]) - model.coded_bits(pages[0]))
        plan = manager.plan(pages, budget_bits=budget)
        ranked_tiers = [tier_rank(plan.decisions[item.page_id].tier) for item in pages]

        assert ranked_tiers == sorted(ranked_tiers)


@given(
    masses=st.lists(
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    ),
    budget_bits=st.integers(min_value=0, max_value=32768),
    parity_group_size=st.integers(min_value=1, max_value=4),
)
def test_residency_plan_respects_erasure_capacity_per_parity_group(
    masses: list[float],
    budget_bits: int,
    parity_group_size: int,
) -> None:
    pages = [page(f"p{index}", mass) for index, mass in enumerate(masses)]
    manager = ResidencyManager()

    plan = manager.plan(
        pages,
        budget_bits=budget_bits,
        erasure_recovery_bound=1,
        parity_group_size=parity_group_size,
    )

    ordered_pages = sorted(pages, key=lambda item: (item.token_range, repr(item.page_id)))
    for start in range(0, len(ordered_pages), parity_group_size):
        group = ordered_pages[start : start + parity_group_size]
        evicted = sum(
            1 for item in group if plan.decisions[item.page_id].tier is ResidencyTier.EVICTED
        )
        assert evicted <= 1


def test_flagged_pages_are_kept_coded_before_unflagged_when_capacity_is_tight() -> None:
    pages = [page("p0", 0.1), page("p1", 0.2), page("p2", 0.3)]
    manager = ResidencyManager()

    plan = manager.plan(
        pages,
        budget_bits=0,
        erasure_recovery_bound=1,
        flagged_page_ids={"p0", "p1"},
    )

    assert plan.decisions["p2"].tier is ResidencyTier.EVICTED
    assert plan.decisions["p0"].tier is ResidencyTier.CODED
    assert plan.decisions["p1"].tier is ResidencyTier.CODED


def test_known_erasure_consumes_reconstruction_capacity() -> None:
    pages = [page("p0", 50.0), page("p1", 10.0), page("p2", 1.0), page("p3", 0.1)]
    manager = ResidencyManager()

    plan = manager.plan(
        pages,
        budget_bits=0,
        erasure_recovery_bound=1,
        known_erasure_ids={"p0"},
    )

    assert plan.decisions["p0"].tier is ResidencyTier.EVICTED
    assert sum(
        1 for item in pages if plan.decisions[item.page_id].tier is ResidencyTier.EVICTED
    ) == 1

    high_budget = manager.plan(
        pages,
        budget_bits=manager.cost_model.resident_bits(pages[0]) * len(pages),
        erasure_recovery_bound=1,
        known_erasure_ids={"p0"},
    )
    assert high_budget.decisions["p0"].tier is ResidencyTier.EVICTED
