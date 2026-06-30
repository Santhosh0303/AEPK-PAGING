"""Thermodynamic residency controller for KV pages."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Mapping

from aepk_paging.kv_page import KVPage, ResidencyTier


LANDAUER_KT_LN2 = log(2.0)


@dataclass(frozen=True)
class TierEstimate:
    tier: ResidencyTier
    storage_bits: int
    free_energy: float


@dataclass(frozen=True)
class ResidencyDecision:
    page_id: object
    tier: ResidencyTier
    storage_bits: int
    free_energy: float


@dataclass(frozen=True)
class ResidencyPlan:
    decisions: Mapping[object, ResidencyDecision]
    total_storage_bits: int
    eviction_count: int
    landauer_cost: float
    total_free_energy: float
    settling_free_energy: tuple[float, ...]


@dataclass(frozen=True)
class TierCostModel:
    """Free-energy residency law.

    [Gibbs] role: attention mass is used as a Boltzmann/Gibbs-style utility
    weight, so high-mass pages get larger benefit from sharper residency.
    [Landauer] role: each EVICTED page accounts an irreversible erase floor of
    erased_bits * kT * ln2 through the configured `temperature_kt`.
    """

    coded_bit_width: int = 4
    temperature_kt: float = 1.0
    resident_utility_weight: float = 2400.0
    coded_utility_weight: float = 900.0
    coded_distortion_weight: float = 160.0
    eviction_distortion_weight: float = 1400.0

    @property
    def landauer_per_bit(self) -> float:
        return self.temperature_kt * LANDAUER_KT_LN2

    def resident_bits(self, page: KVPage) -> int:
        return int((page.K.nbytes + page.V.nbytes) * 8)

    def coded_bits(self, page: KVPage) -> int:
        return int((page.K.size + page.V.size) * self.coded_bit_width)

    def erased_bits(self, page: KVPage) -> int:
        return self.resident_bits(page)

    def tier_estimates(self, page: KVPage) -> Mapping[ResidencyTier, TierEstimate]:
        mass = float(page.attention_mass)
        resident_bits = self.resident_bits(page)
        coded_bits = self.coded_bits(page)
        landauer = self.erased_bits(page) * self.landauer_per_bit
        return {
            ResidencyTier.RESIDENT: TierEstimate(
                tier=ResidencyTier.RESIDENT,
                storage_bits=resident_bits,
                free_energy=float(resident_bits - self.resident_utility_weight * mass),
            ),
            ResidencyTier.CODED: TierEstimate(
                tier=ResidencyTier.CODED,
                storage_bits=coded_bits,
                free_energy=float(
                    coded_bits
                    - self.coded_utility_weight * mass
                    + self.coded_distortion_weight * mass
                ),
            ),
            ResidencyTier.EVICTED: TierEstimate(
                tier=ResidencyTier.EVICTED,
                storage_bits=0,
                free_energy=float(landauer + self.eviction_distortion_weight * mass),
            ),
        }

    def choose_tier(self, page: KVPage, budget_bits: int) -> TierEstimate:
        feasible = [
            estimate
            for estimate in self.tier_estimates(page).values()
            if estimate.storage_bits <= budget_bits
        ]
        if not feasible:
            raise ValueError("budget_bits must be non-negative")
        return min(feasible, key=lambda estimate: (estimate.free_energy, estimate.storage_bits))


class ResidencyManager:
    def __init__(self, cost_model: TierCostModel | None = None) -> None:
        self.cost_model = cost_model or TierCostModel()

    def plan(self, pages: Iterable[KVPage], budget_bits: int) -> ResidencyPlan:
        if budget_bits < 0:
            raise ValueError("budget_bits must be non-negative")
        ordered_pages = sorted(pages, key=lambda page: (page.layer, page.token_range, repr(page.page_id)))
        estimates = {page.page_id: self.cost_model.tier_estimates(page) for page in ordered_pages}
        current = {page.page_id: ResidencyTier.EVICTED for page in ordered_pages}
        total_bits = 0
        settling = [self._total_free_energy(current, estimates)]

        while True:
            best: tuple[float, int, float, str, ResidencyTier, object] | None = None
            for page in ordered_pages:
                next_tier = self._next_sharper_tier(current[page.page_id])
                if next_tier is None:
                    continue
                old = estimates[page.page_id][current[page.page_id]]
                new = estimates[page.page_id][next_tier]
                added_bits = new.storage_bits - old.storage_bits
                if total_bits + added_bits > budget_bits:
                    continue
                free_energy_drop = old.free_energy - new.free_energy
                if free_energy_drop <= 0.0:
                    continue
                rank = (
                    -free_energy_drop / max(1, added_bits),
                    -float(page.attention_mass),
                    new.free_energy,
                    repr(page.page_id),
                    next_tier,
                    page.page_id,
                )
                if best is None or rank < best:
                    best = rank
            if best is None:
                break
            next_tier = best[4]
            page_id = best[5]
            old = estimates[page_id][current[page_id]]
            new = estimates[page_id][next_tier]
            current[page_id] = next_tier
            total_bits += new.storage_bits - old.storage_bits
            settling.append(self._total_free_energy(current, estimates))

        decisions = {
            page_id: ResidencyDecision(
                page_id=page_id,
                tier=tier,
                storage_bits=estimates[page_id][tier].storage_bits,
                free_energy=estimates[page_id][tier].free_energy,
            )
            for page_id, tier in current.items()
        }
        eviction_count = sum(1 for decision in decisions.values() if decision.tier is ResidencyTier.EVICTED)
        erased_bits = sum(
            self.cost_model.erased_bits(page)
            for page in ordered_pages
            if decisions[page.page_id].tier is ResidencyTier.EVICTED
        )
        return ResidencyPlan(
            decisions=decisions,
            total_storage_bits=total_bits,
            eviction_count=eviction_count,
            landauer_cost=float(erased_bits * self.cost_model.landauer_per_bit),
            total_free_energy=settling[-1],
            settling_free_energy=tuple(settling),
        )

    @staticmethod
    def _next_sharper_tier(tier: ResidencyTier) -> ResidencyTier | None:
        if tier is ResidencyTier.EVICTED:
            return ResidencyTier.CODED
        if tier is ResidencyTier.CODED:
            return ResidencyTier.RESIDENT
        return None

    @staticmethod
    def _total_free_energy(
        current: Mapping[object, ResidencyTier],
        estimates: Mapping[object, Mapping[ResidencyTier, TierEstimate]],
    ) -> float:
        return float(sum(estimates[page_id][tier].free_energy for page_id, tier in current.items()))
