import numpy as np

from aepk_paging.kv_page import KVPage, ResidencyTier
from aepk_paging.residency import LANDAUER_KT_LN2, ResidencyManager, TierCostModel


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

    assert plan.eviction_count == 2
    assert plan.landauer_cost == evicted_bits * LANDAUER_KT_LN2
    assert all(
        later <= earlier
        for earlier, later in zip(plan.settling_free_energy, plan.settling_free_energy[1:])
    )
