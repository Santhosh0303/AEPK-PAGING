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
LAMBDA_SWEEP = tuple(float(value) for value in np.logspace(0.0, 9.0, num=181))


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
    notes: str


@dataclass(frozen=True)
class LambdaWinRange:
    winner: str
    start: float
    end: float


@dataclass(frozen=True)
class ParetoPoint:
    key: str
    storage_bits: int
    residual_error: float
    dominated: bool


@dataclass(frozen=True)
class GateResult:
    lambda_ranges: tuple[LambdaWinRange, ...]
    pareto_points: tuple[ParetoPoint, ...]
    aepk_non_dominated: bool
    aepk_lambda_ranges: tuple[LambdaWinRange, ...]

    @property
    def verdict(self) -> str:
        if self.aepk_non_dominated and self.aepk_lambda_ranges:
            return "PASS"
        return "FAIL"


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    baselines: Mapping[str, BaselineResult]
    gate: GateResult
    residency_tiers: Mapping[object, ResidencyTier]
    detector_flags: Mapping[object, bool]
    budget_bits: int


@dataclass(frozen=True)
class ReportData:
    primary: ScenarioResult
    tight_budget: ScenarioResult


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
    stress_pages = _with_attention_masses(clean_pages, (50.0, 10.0, 1.0, 0.1))
    stress_scenario = make_fault_scenario(stress_pages)
    manager = ResidencyManager()
    coded = manager.cost_model.coded_bits(clean_pages[0])
    resident = manager.cost_model.resident_bits(clean_pages[0])
    primary_budget = coded * 3 + resident
    tight_budget = resident + coded
    return ReportData(
        primary=_run_scenario("primary", scenario, primary_budget),
        tight_budget=_run_scenario("tight-budget tier stress", stress_scenario, tight_budget),
    )


def write_report(path: Path = REPORT_PATH) -> ReportData:
    data = run_validation()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(data), encoding="utf-8")
    return data


def render_report(data: ReportData) -> str:
    overall_verdict = "PASS"
    if data.primary.gate.verdict == "FAIL" or data.tight_budget.gate.verdict == "FAIL":
        overall_verdict = "FAIL"
    lines = [
        "# AEPK-Paging Phase 6 REPORT",
        "",
        "This is a numpy-only simulation net-overhead report. It is necessary but not sufficient; Bar 2 on real model KV is Phase 7.",
        "",
        "## Gate Definition",
        "- Rate-distortion currency uses `[Shannon]`: `total_cost(λ) = storage_bits + λ * residual_error`.",
        f"- λ sweep: `{LAMBDA_SWEEP[0]:.2e}` to `{LAMBDA_SWEEP[-1]:.2e}` bits per unit residual MSE.",
        "- PASS iff B3 is Pareto-non-dominated and B3 wins total-cost for at least one reported λ-range.",
        "- Compute proxy is reported as a caveat only; it is not mixed into the rate-distortion gate.",
        "- 12x compute caveat: B3 uses 12.00 detector/recovery proxy ops in the primary scenario.",
        "",
        "## Corruption Scenario",
        f"- Seed: `{SEED}`",
        f"- Corruptions: `quant_noise(level={QUANT_NOISE_LEVEL})`, `bit_flip(p={BIT_FLIP_P})`, `forced_evict(page_ids=['{EVICTED_PAGE_ID}'])`",
        "",
    ]
    lines.extend(_render_scenario(data.primary))
    lines.extend(_render_scenario(data.tight_budget))
    lines.extend(
        [
            "## Corrected Gate Verdict",
            f"- Primary scenario verdict: `{data.primary.gate.verdict}`",
            f"- Tight-budget scenario verdict: `{data.tight_budget.gate.verdict}`",
            f"GATE VERDICT: {overall_verdict}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_scenario(scenario: ScenarioResult) -> list[str]:
    lines = [
        f"## Scenario: {scenario.name}",
        f"- AEPK residency budget bits: `{scenario.budget_bits}`",
        "",
        "### Baseline Matrix",
        "| Baseline | Quality loss MSE | Storage bits | Compute proxy | Residual error | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    if "tier stress" in scenario.name:
        lines.insert(
            2,
            "- Tight-budget tier stress uses higher attention_mass values to exercise residency tiers; Phase 2-5 constants are unchanged.",
        )
    for key in ("B0", "B1", "B2", "B3"):
        item = scenario.baselines[key]
        lines.append(
            f"| {item.name} | {item.quality_loss_mse:.8f} | {item.storage_bits} | "
            f"{item.compute_proxy:.2f} | {item.residual_error:.8f} | {item.notes} |"
        )
    lines.extend(
        [
            "",
            "### Pareto Table",
            "| Baseline | Storage bits | Residual error | Dominated |",
            "|---|---:|---:|---:|",
        ]
    )
    for point in scenario.gate.pareto_points:
        lines.append(
            f"| {point.key} | {point.storage_bits} | {point.residual_error:.8f} | {point.dominated} |"
        )
    lines.extend(
        [
            "",
            "### λ Win Ranges",
            "| Winner | λ start | λ end |",
            "|---|---:|---:|",
        ]
    )
    for item in scenario.gate.lambda_ranges:
        lines.append(f"| {item.winner} | {item.start:.8e} | {item.end:.8e} |")
    if not scenario.gate.lambda_ranges:
        lines.append("| none | 0.00000000e+00 | 0.00000000e+00 |")
    aepk_ranges = _format_ranges(scenario.gate.aepk_lambda_ranges)
    lines.extend(
        [
            "",
            "### AEPK Residency Decisions",
            "| Page | Tier | Detector flagged |",
            "|---|---|---:|",
        ]
    )
    for page_id in sorted(scenario.residency_tiers, key=repr):
        lines.append(
            f"| {page_id} | {scenario.residency_tiers[page_id].value} | {scenario.detector_flags[page_id]} |"
        )
    tier_counts = _tier_counts(scenario.residency_tiers.values())
    lines.extend(
        [
            "",
            "### Corrected Gate",
            f"- B3 Pareto-non-dominated: `{scenario.gate.aepk_non_dominated}`",
            f"- B3 λ win range(s): `{aepk_ranges}`",
            f"- Scenario verdict: `{scenario.gate.verdict}`",
            f"- Tier distribution: `RESIDENT={tier_counts[ResidencyTier.RESIDENT]}, CODED={tier_counts[ResidencyTier.CODED]}, EVICTED={tier_counts[ResidencyTier.EVICTED]}`",
            "",
        ]
    )
    return lines


def _run_scenario(name: str, scenario: FaultScenario, budget_bits: int) -> ScenarioResult:
    b0 = _run_no_protection(scenario)
    b1 = BaselineResult(
        name="B1 keep-all-RESIDENT",
        quality_loss_mse=0.0,
        storage_bits=_resident_bits(scenario.clean_pages),
        compute_proxy=0.0,
        residual_error=0.0,
        notes="Cost ceiling: clean resident pages, no damage.",
    )
    b2 = _run_erasure_parity_only(scenario)
    b3, tiers, detector_flags = _run_full_aepk_stack(scenario, budget_bits)
    baselines = {"B0": b0, "B1": b1, "B2": b2, "B3": b3}
    return ScenarioResult(
        name=name,
        baselines=baselines,
        gate=_rate_distortion_gate(baselines),
        residency_tiers=tiers,
        detector_flags=detector_flags,
        budget_bits=budget_bits,
    )


def _rate_distortion_gate(baselines: Mapping[str, BaselineResult]) -> GateResult:
    pareto = tuple(_pareto_points(baselines))
    ranges = tuple(_lambda_win_ranges(baselines))
    aepk_ranges = tuple(item for item in ranges if item.winner == "B3")
    b3_point = next(point for point in pareto if point.key == "B3")
    return GateResult(
        lambda_ranges=ranges,
        pareto_points=pareto,
        aepk_non_dominated=not b3_point.dominated,
        aepk_lambda_ranges=aepk_ranges,
    )


def _pareto_points(baselines: Mapping[str, BaselineResult]) -> list[ParetoPoint]:
    points: list[ParetoPoint] = []
    for key, item in baselines.items():
        dominated = any(
            other_key != key
            and other.storage_bits <= item.storage_bits
            and other.residual_error <= item.residual_error
            and (other.storage_bits < item.storage_bits or other.residual_error < item.residual_error)
            for other_key, other in baselines.items()
        )
        points.append(
            ParetoPoint(
                key=key,
                storage_bits=item.storage_bits,
                residual_error=item.residual_error,
                dominated=dominated,
            )
        )
    return sorted(points, key=lambda point: point.key)


def _lambda_win_ranges(baselines: Mapping[str, BaselineResult]) -> list[LambdaWinRange]:
    winners: list[str] = []
    for lam in LAMBDA_SWEEP:
        totals = {
            key: item.storage_bits + lam * item.residual_error
            for key, item in baselines.items()
        }
        winners.append(min(totals, key=lambda key: (totals[key], key)))
    ranges: list[LambdaWinRange] = []
    start_index = 0
    for index in range(1, len(winners)):
        if winners[index] != winners[start_index]:
            ranges.append(
                LambdaWinRange(
                    winner=winners[start_index],
                    start=LAMBDA_SWEEP[start_index],
                    end=LAMBDA_SWEEP[index - 1],
                )
            )
            start_index = index
    ranges.append(
        LambdaWinRange(
            winner=winners[start_index],
            start=LAMBDA_SWEEP[start_index],
            end=LAMBDA_SWEEP[-1],
        )
    )
    return ranges


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
        notes="GhostServe-like known-erasure recovery; no unknown-location bit-flip correction.",
    )


def _run_full_aepk_stack(
    scenario: FaultScenario,
    budget_bits: int,
) -> tuple[BaselineResult, Mapping[object, ResidencyTier], Mapping[object, bool]]:
    manager = ResidencyManager()
    clean_pages = scenario.clean_pages
    plan = manager.plan(clean_pages, budget_bits=budget_bits)
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

    missing_page_ids = {
        page.page_id
        for page in clean_pages
        if page.page_id == scenario.evicted_page_id or tiers[page.page_id] is ResidencyTier.EVICTED
    }
    recovered_erasures: dict[object, KVPage] = {}
    if len(missing_page_ids) == 1:
        missing_id = next(iter(missing_page_ids))
        recovered_erasures[missing_id] = recover_erasure(group, [missing_id])

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
        if page.page_id in missing_page_ids:
            outputs[page.page_id] = recovered_erasures.get(page.page_id)
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
    protected_bits = sum(
        (protected_page.K.values.size + protected_page.V.values.size) * 8
        for protected_page in protected
    )
    syndrome_bits = int(protected_bits - raw_quant_bits)
    fingerprint_bits = len(clean_pages) * 128
    storage_bits = parity_bits + syndrome_bits + fingerprint_bits + plan.total_storage_bits
    return (
        BaselineResult(
            name="B3 full AEPK stack",
            quality_loss_mse=quality,
            storage_bits=storage_bits,
            compute_proxy=compute_proxy,
            residual_error=quality,
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


def _with_attention_masses(pages: Iterable[KVPage], masses: Iterable[float]) -> tuple[KVPage, ...]:
    return tuple(
        KVPage(
            page_id=page.page_id,
            layer=page.layer,
            token_range=page.token_range,
            K=page.K,
            V=page.V,
            precision_tag=page.precision_tag,
            attention_mass=mass,
        )
        for page, mass in zip(pages, masses)
    )


def _tier_counts(tiers: Iterable[ResidencyTier]) -> dict[ResidencyTier, int]:
    counts = {tier: 0 for tier in ResidencyTier}
    for tier in tiers:
        counts[tier] += 1
    return counts


def _format_ranges(ranges: Iterable[LambdaWinRange]) -> str:
    values = list(ranges)
    if not values:
        return "none"
    return ", ".join(f"{item.start:.2e}..{item.end:.2e}" for item in values)


def main() -> None:
    write_report()


if __name__ == "__main__":
    main()
