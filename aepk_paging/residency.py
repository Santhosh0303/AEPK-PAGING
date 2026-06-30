"""Thermodynamic residency controller for KV pages."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Hashable, Iterable, Mapping

import numpy as np

from aepk_paging.detect import attention_distribution
from aepk_paging.kv_page import KVPage, ResidencyTier


LANDAUER_KT_LN2 = log(2.0)


@dataclass(frozen=True)
class TierEstimate:
    tier: ResidencyTier
    storage_bits: int
    energy: float
    entropy_nats: float
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

    [Gibbs] role: softmax attention is treated as a Boltzmann distribution and
    contributes Shannon entropy S to F = E - kT*S; sharper tiers retain more S.
    [Landauer] role: each EVICTED page accounts an irreversible erase floor of
    erased_bits * kT * ln2 through the configured `temperature_kt`.
    """

    coded_bit_width: int = 4
    temperature_kt: float = 1.0
    resident_utility_weight: float = 2400.0
    coded_utility_weight: float = 900.0
    coded_distortion_weight: float = 160.0
    eviction_distortion_weight: float = 1400.0
    coded_entropy_retention: float = 0.5
    entropy_weight: float = 1.0  # isolates the Gibbs -T*S term for sensitivity study (#8); 1.0 = nominal

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
        entropy = page_attention_entropy(page)
        resident_energy = float(resident_bits - self.resident_utility_weight * mass)
        coded_energy = float(
            coded_bits - self.coded_utility_weight * mass + self.coded_distortion_weight * mass
        )
        evicted_energy = float(landauer + self.eviction_distortion_weight * mass)
        return {
            ResidencyTier.RESIDENT: TierEstimate(
                tier=ResidencyTier.RESIDENT,
                storage_bits=resident_bits,
                energy=resident_energy,
                entropy_nats=entropy,
                free_energy=self.free_energy(resident_energy, entropy),
            ),
            ResidencyTier.CODED: TierEstimate(
                tier=ResidencyTier.CODED,
                storage_bits=coded_bits,
                energy=coded_energy,
                entropy_nats=entropy * self.coded_entropy_retention,
                free_energy=self.free_energy(coded_energy, entropy * self.coded_entropy_retention),
            ),
            ResidencyTier.EVICTED: TierEstimate(
                tier=ResidencyTier.EVICTED,
                storage_bits=0,
                energy=evicted_energy,
                entropy_nats=0.0,
                free_energy=self.free_energy(evicted_energy, 0.0),
            ),
        }

    def free_energy(self, energy: float, entropy_nats: float) -> float:
        return float(energy - self.entropy_weight * self.temperature_kt * entropy_nats)

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

    def plan(
        self,
        pages: Iterable[KVPage],
        budget_bits: int,
        *,
        erasure_recovery_bound: int = 1,
        parity_group_size: int | None = None,
        flagged_page_ids: Iterable[Hashable] = (),
        known_erasure_ids: Iterable[Hashable] = (),
    ) -> ResidencyPlan:
        if budget_bits < 0:
            raise ValueError("budget_bits must be non-negative")
        if erasure_recovery_bound < 0:
            raise ValueError("erasure_recovery_bound must be non-negative")
        if parity_group_size is not None and parity_group_size <= 0:
            raise ValueError("parity_group_size must be positive")
        ordered_pages = sorted(pages, key=lambda page: (page.layer, page.token_range, repr(page.page_id)))
        estimates = {page.page_id: self.cost_model.tier_estimates(page) for page in ordered_pages}
        flagged_ids = set(flagged_page_ids)
        known_erasures = set(known_erasure_ids)
        current = self._initial_capacity_safe_tiers(
            ordered_pages,
            erasure_recovery_bound=erasure_recovery_bound,
            parity_group_size=parity_group_size,
            flagged_page_ids=flagged_ids,
            known_erasure_ids=known_erasures,
        )
        total_bits = sum(estimates[page_id][tier].storage_bits for page_id, tier in current.items())
        settling = [self._total_free_energy(current, estimates)]

        while True:
            best: tuple[float, int, float, str, ResidencyTier, object] | None = None
            for page in ordered_pages:
                if page.page_id in known_erasures and current[page.page_id] is ResidencyTier.EVICTED:
                    continue
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
    def _initial_capacity_safe_tiers(
        pages: list[KVPage],
        *,
        erasure_recovery_bound: int,
        parity_group_size: int | None,
        flagged_page_ids: set[Hashable],
        known_erasure_ids: set[Hashable],
    ) -> dict[Hashable, ResidencyTier]:
        current = {page.page_id: ResidencyTier.CODED for page in pages}
        for group in ResidencyManager._parity_groups(pages, parity_group_size):
            known_in_group = [page for page in group if page.page_id in known_erasure_ids]
            if len(known_in_group) > erasure_recovery_bound:
                raise ValueError("known erasures exceed erasure_recovery_bound")
            for page in known_in_group:
                current[page.page_id] = ResidencyTier.EVICTED
            remaining_capacity = erasure_recovery_bound - len(known_in_group)
            ordered_group = sorted(
                [page for page in group if page.page_id not in known_erasure_ids],
                key=lambda page: (
                    page.page_id in flagged_page_ids,
                    float(page.attention_mass),
                    page.layer,
                    page.token_range,
                    repr(page.page_id),
                ),
            )
            for page in ordered_group[:remaining_capacity]:
                current[page.page_id] = ResidencyTier.EVICTED
        return current

    @staticmethod
    def _parity_groups(pages: list[KVPage], parity_group_size: int | None) -> list[list[KVPage]]:
        by_layer: dict[int, list[KVPage]] = {}
        for page in pages:
            by_layer.setdefault(page.layer, []).append(page)
        groups: list[list[KVPage]] = []
        for layer_pages in by_layer.values():
            ordered = sorted(layer_pages, key=lambda page: (page.token_range, repr(page.page_id)))
            if parity_group_size is None:
                groups.append(ordered)
            else:
                for start in range(0, len(ordered), parity_group_size):
                    groups.append(ordered[start : start + parity_group_size])
        return groups

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


def page_attention_entropy(page: KVPage, *, temperature: float = 1.0) -> float:
    weights = attention_distribution(page, temperature=temperature).astype(np.float64)
    positive = weights[weights > 0.0]
    return float(-np.sum(positive * np.log(positive)))
