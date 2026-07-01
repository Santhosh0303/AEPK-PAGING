"""Exact KV page model and Page table."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Hashable

import numpy as np


class ResidencyTier(StrEnum):
    RESIDENT = "RESIDENT"
    CODED = "CODED"
    EVICTED = "EVICTED"


@dataclass(frozen=True)
class KVPage:
    page_id: Hashable
    layer: int
    token_range: tuple[int, int]
    K: np.ndarray
    V: np.ndarray
    precision_tag: str
    attention_mass: float

    def __post_init__(self) -> None:
        start, end = self.token_range
        if start < 0 or end <= start:
            raise ValueError("token_range must be a non-empty half-open range")
        if self.K.shape != self.V.shape:
            raise ValueError("K and V must have identical shapes")
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if not np.isfinite(self.attention_mass) or self.attention_mass < 0.0:
            raise ValueError("attention_mass must be a finite non-negative number")
        object.__setattr__(self, "K", np.array(self.K, copy=True))
        object.__setattr__(self, "V", np.array(self.V, copy=True))


@dataclass(frozen=True)
class PageEntry:
    logical_id: Hashable
    physical_id: int | None
    tier: ResidencyTier


class PageTable:
    def __init__(self) -> None:
        self._logical_to_entry: dict[Hashable, PageEntry] = {}
        self._physical_to_page: dict[int, KVPage] = {}
        self._next_physical_id = 0

    def store(self, page: KVPage, tier: ResidencyTier = ResidencyTier.RESIDENT) -> int:
        if not isinstance(tier, ResidencyTier):
            tier = ResidencyTier(tier)
        if tier is not ResidencyTier.RESIDENT:
            raise ValueError("Phase 1 store supports only RESIDENT pages")
        if page.page_id in self._logical_to_entry:
            raise ValueError("logical page already stored")
        self._ensure_no_overlap(page)
        physical_id = self._next_physical_id
        self._next_physical_id += 1
        self._logical_to_entry[page.page_id] = PageEntry(
            logical_id=page.page_id,
            physical_id=physical_id,
            tier=tier,
        )
        self._physical_to_page[physical_id] = page
        self.validate_invariants()
        return physical_id

    def fetch(self, logical_id: Hashable) -> KVPage:
        entry = self._logical_to_entry[logical_id]
        if entry.tier is not ResidencyTier.RESIDENT:
            raise KeyError("page is not RESIDENT")
        if entry.physical_id is None:
            raise KeyError("RESIDENT page has no physical mapping")
        page = self._physical_to_page[entry.physical_id]
        return KVPage(
            page_id=page.page_id,
            layer=page.layer,
            token_range=page.token_range,
            K=page.K,
            V=page.V,
            precision_tag=page.precision_tag,
            attention_mass=page.attention_mass,
        )

    def delete(self, logical_id: Hashable) -> None:
        entry = self._logical_to_entry.pop(logical_id)
        if entry.physical_id is not None:
            del self._physical_to_page[entry.physical_id]
        self.validate_invariants()

    def entry(self, logical_id: Hashable) -> PageEntry:
        return self._logical_to_entry[logical_id]

    def validate_invariants(self) -> None:
        physical_ids: list[int] = []
        seen_ranges: list[tuple[int, int, int]] = []
        for logical_id, entry in self._logical_to_entry.items():
            if logical_id != entry.logical_id:
                raise AssertionError("logical mapping key mismatch")
            if entry.tier is ResidencyTier.RESIDENT:
                if entry.physical_id is None:
                    raise AssertionError("RESIDENT page missing physical id")
                physical_ids.append(entry.physical_id)
                page = self._physical_to_page.get(entry.physical_id)
                if page is None:
                    raise AssertionError("logical mapping leaks missing physical page")
                if page.page_id != logical_id:
                    raise AssertionError("physical page id mismatch")
                start, end = page.token_range
                seen_ranges.append((page.layer, start, end))
            elif entry.physical_id is not None:
                raise AssertionError("non-RESIDENT page owns physical storage")
        if len(set(physical_ids)) != len(physical_ids):
            raise AssertionError("physical page aliased by multiple logical pages")
        if set(physical_ids) != set(self._physical_to_page):
            raise AssertionError("physical storage leak")
        for index, left in enumerate(seen_ranges):
            for right in seen_ranges[index + 1 :]:
                if left[0] == right[0] and max(left[1], right[1]) < min(left[2], right[2]):
                    raise AssertionError("overlapping token ranges in one layer")

    def _ensure_no_overlap(self, page: KVPage) -> None:
        start, end = page.token_range
        for stored in self._physical_to_page.values():
            stored_start, stored_end = stored.token_range
            if page.layer == stored.layer and max(start, stored_start) < min(end, stored_end):
                raise ValueError("page overlaps existing token range")
