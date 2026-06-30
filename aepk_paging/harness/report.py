"""Deterministic Phase 6 validation harness and REPORT generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from aepk_paging.coding import (
    HammingSECDEDCode,
    UncorrectableError,
    encode_erasure_group,
    recover_erasure,
)
from aepk_paging.detect import attention_mass, attention_mass_detector, norm_consistency_detector, norm_ratio
from aepk_paging.kv_page import KVPage, ResidencyTier
from aepk_paging.lossy_tier import bit_flip, page_mse, quant_noise, quantize_page
from aepk_paging.residency import ResidencyManager


REPORT_PATH = Path(__file__).resolve().parents[2] / "results" / "REPORT.md"
SEED = 611
QUANT_NOISE_LEVEL = 0.35
BIT_FLIP_P = 0.0008
EVICTED_PAGE_ID = "p0"
BIT_FLIP_PAGE_ID = "p1"
RECOVERY_TARGET = 0.80


@dataclass(frozen=True)
class FaultScenario:
    clean_pages: tuple[KVPage, ...]
    quant_noisy_pages: Mapping[object, KVPage]
    raw_bitflip_page: KVPage
    secded_bitflip_page: object
    evicted_page_id: object
    bit_flip_page_id: object


@dataclass(frozen=True)
class BaselineResult:
    name: str
    quality_loss_mse: float
    storage_bits: int
    compute_proxy: float
    residual_error: float
    overhead_proxy: float
    notes: str


@dataclass(frozen=True)
class GateResult:
    damage_cost: float
    recovered_fraction: float
    heal_overhead: float
    b3_beats_b2_error_regime: bool
    recovery_condition: bool
    overhead_condition: bool
    error_regime_condition: bool

    @property
    def verdict(self) -> str:
        if self.recovery_condition and self.overhead_condition and self.error_regime_condition:
            return "PASS"
        return "FAIL"


@dataclass(frozen=True)
class ReportData:
    baselines: Mapping[str, BaselineResult]
    gate: GateResult
    residency_tiers: Mapping[object, ResidencyTier]
    detector_flags: Mapping[object, bool]


def build_clean_pages(seed: int = SEED) -> tuple[KVPage, ...]:
    rng = np.random.default_rng(seed)
    pages: list[KVPage] = []
    for index, mass in enumerate((0.92, 0.71, 0.38, 0.14)):
        K = rng.normal(loc=0.0, scale=1.0, size=(16, 8)).astype(np.float32)
        V = (K * np.float32(0.6) + rng.normal(loc=0.0, scale=0.03, size=(16, 8))).astype(
            np.float32
        )
        base = KVPage(
            page_id=f"p{index}",
            layer=0,
            token_range=(index * 16, index * 16 + 16),
            K=K,
            V=V,
            precision_tag="float32",
            attention_mass=mass,
        )
        pages.append(
            KVPage(
                page_id=base.page_id,
                layer=base.layer,
                token_range=base.token_range,
                K=base.K,
                V=base.V,
                precision_tag=base.precision_tag,
                attention_mass=attention_mass(base),
            )
        )
    return tuple(pages)


def make_fault_scenario(clean_pages: tuple[KVPage, ...]) -> FaultScenario:
    noisy = {
        page.page_id: quant_noise(page, level=QUANT_NOISE_LEVEL, seed=SEED + 100 + offset)[0]
        for offset, page in enumerate(clean_pages)
    }
    bitflip_clean = _page_by_id(clean_pages, BIT_FLIP_PAGE_ID)
    raw_bitflip = bit_flip(quantize_page(bitflip_clean, bit_width=8), p=BIT_FLIP_P, seed=SEED + 200)
    secded = HammingSECDEDCode().encode([quantize_page(bitflip_clean, bit_width=8)])[0]
    secded_bitflip = bit_flip(secded, p=BIT_FLIP_P, seed=SEED + 200)
    return FaultScenario(
        clean_pages=clean_pages,
        quant_noisy_pages=noisy,
        raw_bitflip_page=raw_bitflip.dequantize(),
        secded_bitflip_page=secded_bitflip,
        evicted_page_id=EVICTED_PAGE_ID,
        bit_flip_page_id=BIT_FLIP_PAGE_ID,
    )


def run_validation() -> ReportData:
    clean_pages = build_clean_pages()
    scenario = make_fault_scenario(clean_pages)
    b1_quality = 0.0
    b0 = _run_no_protection(scenario)
    b1 = BaselineResult(
        name="B1 keep-all-RESIDENT",
        quality_loss_mse=b1_quality,
        storage_bits=_resident_bits(clean_pages),
        compute_proxy=0.0,
        residual_error=b1_quality,
        overhead_proxy=float(_resident_bits(clean_pages)),
        notes="Cost ceiling: clean resident pages, no damage.",
    )
    b2 = _run_erasure_parity_only(scenario)
    b3, tiers, detector_flags = _run_full_aepk_stack(scenario)
    damage_cost = b0.quality_loss_mse - b1.quality_loss_mse
    recovered_fraction = (
        (b0.quality_loss_mse - b3.quality_loss_mse) / damage_cost if damage_cost > 0.0 else 0.0
    )
    heal_overhead = b3.overhead_proxy
    gate = GateResult(
        damage_cost=damage_cost,
        recovered_fraction=recovered_fraction,
        heal_overhead=heal_overhead,
        b3_beats_b2_error_regime=b3.quality_loss_mse < b2.quality_loss_mse,
        recovery_condition=recovered_fraction >= RECOVERY_TARGET,
        overhead_condition=heal_overhead < damage_cost,
        error_regime_condition=b3.quality_loss_mse < b2.quality_loss_mse,
    )
    return ReportData(
        baselines={"B0": b0, "B1": b1, "B2": b2, "B3": b3},
        gate=gate,
        residency_tiers=tiers,
        detector_flags=detector_flags,
    )


def write_report(path: Path = REPORT_PATH) -> ReportData:
    data = run_validation()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(data), encoding="utf-8")
    return data


def render_report(data: ReportData) -> str:
    lines = [
        "# AEPK-Paging Phase 6 REPORT",
        "",
        "This is a numpy-only simulation net-overhead report. It is necessary but not sufficient; Bar 2 on real model KV is Phase 7.",
        "",
        "## Scenario",
        f"- Seed: `{SEED}`",
        f"- Corruptions: `quant_noise(level={QUANT_NOISE_LEVEL})`, `bit_flip(p={BIT_FLIP_P})`, `forced_evict(page_ids=['{EVICTED_PAGE_ID}'])`",
        f"- Gate recovery target X: `{RECOVERY_TARGET:.2f}`",
        "",
        "## Baseline Matrix",
        "| Baseline | Quality loss MSE | Storage bits | Compute proxy | Residual error | Total overhead proxy | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("B0", "B1", "B2", "B3"):
        item = data.baselines[key]
        lines.append(
            f"| {item.name} | {item.quality_loss_mse:.8f} | {item.storage_bits} | "
            f"{item.compute_proxy:.2f} | {item.residual_error:.8f} | {item.overhead_proxy:.8f} | {item.notes} |"
        )
    lines.extend(
        [
            "",
            "## AEPK Residency Decisions",
            "| Page | Tier | Detector flagged |",
            "|---|---|---:|",
        ]
    )
    for page_id in sorted(data.residency_tiers, key=repr):
        lines.append(
            f"| {page_id} | {data.residency_tiers[page_id].value} | {data.detector_flags[page_id]} |"
        )
    gate = data.gate
    lines.extend(
        [
            "",
            "## Net-Overhead Gate",
            f"- `damage_cost = B0_quality_loss - B1_quality_loss = {gate.damage_cost:.8f}`",
            f"- `heal_overhead = B3_extra_bits + compute_proxy + residual_error = {gate.heal_overhead:.8f}`",
            f"- Recovery condition: `{gate.recovered_fraction:.8f} >= {RECOVERY_TARGET:.8f}` -> `{gate.recovery_condition}`",
            f"- Overhead condition: `{gate.heal_overhead:.8f} < {gate.damage_cost:.8f}` -> `{gate.overhead_condition}`",
            f"- Error-regime condition: `B3_quality_loss < B2_quality_loss` -> `{gate.error_regime_condition}`",
            f"- B3 beats B2 on error regime: `{gate.b3_beats_b2_error_regime}`",
            "",
            f"GATE VERDICT: {gate.verdict}",
            "",
        ]
    )
    return "\n".join(lines)


def _run_no_protection(scenario: FaultScenario) -> BaselineResult:
    outputs: dict[object, KVPage | None] = dict(scenario.quant_noisy_pages)
    outputs[scenario.bit_flip_page_id] = scenario.raw_bitflip_page
    outputs[scenario.evicted_page_id] = None
    quality = _quality_loss(scenario.clean_pages, outputs)
    return BaselineResult(
        name="B0 no protection",
        quality_loss_mse=quality,
        storage_bits=0,
        compute_proxy=0.0,
        residual_error=quality,
        overhead_proxy=quality,
        notes="Takes quant-noise, raw bit-flip damage, and forced eviction.",
    )


def _run_erasure_parity_only(scenario: FaultScenario) -> BaselineResult:
    group = encode_erasure_group(scenario.clean_pages)
    recovered = recover_erasure(group, [scenario.evicted_page_id])
    outputs: dict[object, KVPage | None] = dict(scenario.quant_noisy_pages)
    outputs[scenario.evicted_page_id] = recovered
    outputs[scenario.bit_flip_page_id] = scenario.raw_bitflip_page
    parity_bits = int(group.parity_K.nbytes + group.parity_V.nbytes) * 8
    quality = _quality_loss(scenario.clean_pages, outputs)
    return BaselineResult(
        name="B2 erasure-parity only",
        quality_loss_mse=quality,
        storage_bits=parity_bits,
        compute_proxy=1.0,
        residual_error=quality,
        overhead_proxy=float(parity_bits) + 1.0 + quality,
        notes="GhostServe-like known-erasure recovery; no unknown-location bit-flip correction.",
    )


def _run_full_aepk_stack(
    scenario: FaultScenario,
) -> tuple[BaselineResult, Mapping[object, ResidencyTier], Mapping[object, bool]]:
    manager = ResidencyManager()
    clean_pages = scenario.clean_pages
    budget = manager.cost_model.coded_bits(clean_pages[0]) * 3 + manager.cost_model.resident_bits(clean_pages[0])
    plan = manager.plan(clean_pages, budget_bits=budget)
    tiers = {page_id: decision.tier for page_id, decision in plan.decisions.items()}
    group = encode_erasure_group(clean_pages)
    code = HammingSECDEDCode()
    protected = code.encode([quantize_page(page, bit_width=8) for page in clean_pages])
    protected_by_id = {page.page_id: page for page in protected}
    protected_by_id[scenario.bit_flip_page_id] = scenario.secded_bitflip_page
    try:
        corrected = code.correct(tuple(protected_by_id.values()))
    except UncorrectableError:
        corrected = {}

    outputs: dict[object, KVPage | None] = {}
    detector_flags: dict[object, bool] = {}
    compute_proxy = 0.0
    for page in clean_pages:
        noisy = scenario.quant_noisy_pages[page.page_id]
        mass = attention_mass_detector(noisy, expected_mass=page.attention_mass, tolerance=0.01)
        norm = norm_consistency_detector(noisy, expected_ratio=norm_ratio(page), tolerance=0.01)
        flagged = mass.flag or norm.flag
        detector_flags[page.page_id] = flagged
        compute_proxy += 2.0
        tier = tiers[page.page_id]
        if page.page_id == scenario.evicted_page_id or tier is ResidencyTier.EVICTED:
            outputs[page.page_id] = recover_erasure(group, [page.page_id])
            compute_proxy += 1.0
        elif tier is ResidencyTier.RESIDENT:
            outputs[page.page_id] = page
        elif page.page_id in corrected:
            outputs[page.page_id] = corrected[page.page_id].dequantize()
            compute_proxy += 1.0
        elif flagged:
            outputs[page.page_id] = quantize_page(page, bit_width=8).dequantize()
            compute_proxy += 1.0
        else:
            outputs[page.page_id] = noisy

    quality = _quality_loss(clean_pages, outputs)
    parity_bits = int(group.parity_K.nbytes + group.parity_V.nbytes) * 8
    raw_quant_bits = sum((page.K.size + page.V.size) * 8 for page in clean_pages)
    protected_bits = sum((protected_page.K.values.size + protected_page.V.values.size) * 8 for protected_page in protected)
    syndrome_bits = int(protected_bits - raw_quant_bits)
    fingerprint_bits = len(clean_pages) * 128
    extra_bits = parity_bits + syndrome_bits + fingerprint_bits + plan.total_storage_bits
    overhead = float(extra_bits) + compute_proxy + quality
    return (
        BaselineResult(
            name="B3 full AEPK stack",
            quality_loss_mse=quality,
            storage_bits=extra_bits,
            compute_proxy=compute_proxy,
            residual_error=quality,
            overhead_proxy=overhead,
            notes="Detection + parity/SECDED recovery + thermodynamic residency decision.",
        ),
        tiers,
        detector_flags,
    )


def _quality_loss(clean_pages: Iterable[KVPage], outputs: Mapping[object, KVPage | None]) -> float:
    losses = []
    for clean in clean_pages:
        output = outputs.get(clean.page_id)
        if output is None:
            losses.append(_missing_page_penalty(clean))
        else:
            losses.append(page_mse(clean, output))
    return float(np.mean(losses))


def _missing_page_penalty(page: KVPage) -> float:
    zero = KVPage(
        page_id=page.page_id,
        layer=page.layer,
        token_range=page.token_range,
        K=np.zeros_like(page.K),
        V=np.zeros_like(page.V),
        precision_tag="missing-penalty",
        attention_mass=page.attention_mass,
    )
    return page_mse(page, zero)


def _resident_bits(pages: Iterable[KVPage]) -> int:
    return int(sum((page.K.nbytes + page.V.nbytes) * 8 for page in pages))


def _page_by_id(pages: Iterable[KVPage], page_id: object) -> KVPage:
    for page in pages:
        if page.page_id == page_id:
            return page
    raise KeyError(page_id)


def main() -> None:
    write_report()


if __name__ == "__main__":
    main()
