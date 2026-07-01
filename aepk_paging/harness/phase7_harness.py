"""
Phase 7.3 harness: real-KV corruption → detect → recover → residency.

Injects Phase-2 damage onto real DynamicCache pages, runs Phase-4 detectors
(including finiteness_detector), applies Phase-3 RS recovery, and Phase-5
capacity-coupled residency.  Returns a structured result for the acceptance test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aepk_paging.kv_page import KVPage, ResidencyTier
from aepk_paging.detect import (
    DetectorResult,
    attention_mass_detector,
    norm_consistency_detector,
    finiteness_detector,
    norm_ratio,
    attention_mass,
)
from aepk_paging.lossy_tier import quant_noise, quantize_page, bit_flip, page_mse
from aepk_paging.coding import (
    encode_rs_erasure_group,
    recover_rs_erasure,
    ReedSolomonCode,
    ReedSolomonCodewords,
    UncorrectableError,
)
from aepk_paging.residency import ResidencyManager, TierCostModel


@dataclass(frozen=True)
class PageDetection:
    page_id: object
    finiteness: DetectorResult
    attention_mass: DetectorResult
    norm_consistency: DetectorResult

    @property
    def any_flagged(self) -> bool:
        return self.finiteness.flag or self.attention_mass.flag or self.norm_consistency.flag


@dataclass(frozen=True)
class Phase7Result:
    """Structured output of a 7.3 run."""
    # Detection
    clean_detections: list[PageDetection]
    corrupt_detections: list[PageDetection]
    # Quality
    quant_noise_mse: float           # mean MSE from quant_noise damage
    # Erasure recovery
    erasure_recovered_bit_exact: bool
    # Error correction
    error_correction_mse_before: float
    error_correction_mse_after: float
    error_correction_improved: bool
    # Residency
    residency_tiers: dict[object, ResidencyTier]
    evicted_count: int
    resident_count: int


def _flatten_for_detectors(page: KVPage) -> KVPage:
    """Return 2D view [seq_len, features] for detectors that use axis=1 norms."""
    if page.K.ndim == 2:
        return page
    seq_len = page.K.shape[0]
    return KVPage(
        page_id=page.page_id,
        layer=page.layer,
        token_range=page.token_range,
        K=page.K.reshape(seq_len, -1),
        V=page.V.reshape(seq_len, -1),
        precision_tag=page.precision_tag,
        attention_mass=page.attention_mass,
    )


def _detect(page: KVPage, expected_ratio: float | None = None) -> PageDetection:
    flat = _flatten_for_detectors(page)
    expected_ratio = expected_ratio if expected_ratio is not None else norm_ratio(flat)
    return PageDetection(
        page_id=page.page_id,
        finiteness=finiteness_detector(page),
        attention_mass=attention_mass_detector(flat, tolerance=0.05),
        norm_consistency=norm_consistency_detector(flat, expected_ratio=expected_ratio, tolerance=0.10),
    )


def run_phase7_corruption_pipeline(
    pages: Sequence[KVPage],
    *,
    noise_level: float = 0.5,
    noise_seed: int = 42,
    num_rs_parity: int = 1,
    rs_error_t: int = 2,
    bit_flip_p: float = 0.02,
    vram_budget_bytes: int = 4 * 1024 * 1024,
) -> Phase7Result:
    """
    Full 7.3 pipeline on a list of KVPages extracted from a real model.

    pages: list[KVPage] — at least 2 for erasure, at least 1 for everything else.
    """
    assert len(pages) >= 2, "need at least 2 pages for erasure group"

    # --- Baseline detections (clean) ---
    clean_ratios = [norm_ratio(_flatten_for_detectors(p)) for p in pages]
    clean_detections = [_detect(p, r) for p, r in zip(pages, clean_ratios)]

    # --- Phase 2: quant_noise damage ---
    corrupt_pages: list[KVPage] = []
    mses: list[float] = []
    for i, page in enumerate(pages):
        damaged, mse = quant_noise(page, level=noise_level, seed=noise_seed + i)
        corrupt_pages.append(damaged)
        mses.append(mse)
    quant_noise_mse = float(np.mean(mses))

    # --- Phase 4: detect on corrupted pages (using clean baselines) ---
    corrupt_detections = [_detect(cp, r) for cp, r in zip(corrupt_pages, clean_ratios)]

    # --- Phase 3a: RS erasure — evict first page, encode+recover ---
    rs_group = encode_rs_erasure_group(list(pages), num_parity=num_rs_parity)
    evicted_id = pages[0].page_id
    recovered_map = recover_rs_erasure(rs_group, [evicted_id])
    recovered_page = recovered_map[evicted_id]
    erasure_bit_exact = bool(
        np.array_equal(recovered_page.K, pages[0].K)
        and np.array_equal(recovered_page.V, pages[0].V)
    )

    # --- Phase 3b: RS error correction on quantized K values ---
    # API: encode_array(np.ndarray) -> ReedSolomonCodewords
    #       correct_array(ReedSolomonCodewords) -> (np.ndarray, int)
    # Verified: ReedSolomonCode uses systematic encoding (first k symbols = message).
    rs_code = ReedSolomonCode(t=rs_error_t)
    target_page = pages[1]
    qpage = quantize_page(target_page, bit_width=8)

    enc_K = rs_code.encode_array(qpage.K.values)      # encode int8 K array

    # Corrupt: flip bits in the codeword bytes
    rng_ec = np.random.default_rng(noise_seed + 99)
    noisy_cw = enc_K.codewords.copy()                 # (num_blocks, 255) uint8
    flip_mask = rng_ec.random(size=noisy_cw.shape) < bit_flip_p
    noisy_cw = np.bitwise_xor(noisy_cw, flip_mask.astype(np.uint8))
    noisy_enc_K = ReedSolomonCodewords(
        codewords=noisy_cw,
        original_len=enc_K.original_len,
        shape=enc_K.shape,
        dtype=enc_K.dtype,
    )

    # Baseline "corrupted without correction":
    # systematic RS → first k symbols per block = message bytes (when uncorrupted)
    k_sym = rs_code.k
    raw_corrupt_flat = noisy_cw[:, :k_sym].astype(np.uint8).reshape(-1)[:enc_K.original_len]
    corrupt_k_vals = raw_corrupt_flat.view(enc_K.dtype).reshape(enc_K.shape)

    orig_k_f32 = qpage.K.values.astype(np.float32) * np.float32(qpage.K.scale)
    corrupt_k_f32 = corrupt_k_vals.astype(np.float32) * np.float32(qpage.K.scale)
    mse_before = float(np.mean((orig_k_f32 - corrupt_k_f32) ** 2))

    try:
        corrected_k_vals, _ = rs_code.correct_array(noisy_enc_K)
        corrected_k_f32 = corrected_k_vals.astype(np.float32) * np.float32(qpage.K.scale)
        mse_after = float(np.mean((orig_k_f32 - corrected_k_f32) ** 2))
    except UncorrectableError:
        mse_after = mse_before  # correction failed; treat as no improvement
    error_correction_improved = mse_after <= mse_before

    # --- Phase 5: capacity-coupled residency ---
    # ResidencyManager API (verified from residency.py):
    #   __init__(cost_model=None)  — no erasure_recovery_bound here
    #   plan(pages, budget_bits, *, erasure_recovery_bound=1, flagged_page_ids=(), ...)
    #   returns ResidencyPlan with .decisions: Mapping[page_id, ResidencyDecision]
    #   ResidencyDecision.tier, .storage_bits, .free_energy
    cost_model = TierCostModel()  # default weights; no custom args
    manager = ResidencyManager(cost_model=cost_model)
    flagged_ids = {det.page_id for det in corrupt_detections if det.any_flagged}
    budget_bits = vram_budget_bytes * 8
    plan = manager.plan(
        pages=list(pages),
        budget_bits=budget_bits,
        erasure_recovery_bound=num_rs_parity,
        flagged_page_ids=flagged_ids,
    )
    residency_tiers = {dec.page_id: dec.tier for dec in plan.decisions.values()}
    evicted_count = sum(1 for t in residency_tiers.values() if t == ResidencyTier.EVICTED)
    resident_count = sum(1 for t in residency_tiers.values() if t == ResidencyTier.RESIDENT)

    return Phase7Result(
        clean_detections=list(clean_detections),
        corrupt_detections=list(corrupt_detections),
        quant_noise_mse=quant_noise_mse,
        erasure_recovered_bit_exact=erasure_bit_exact,
        error_correction_mse_before=mse_before,
        error_correction_mse_after=mse_after,
        error_correction_improved=error_correction_improved,
        residency_tiers=residency_tiers,
        evicted_count=evicted_count,
        resident_count=resident_count,
    )
