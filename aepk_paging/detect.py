"""Content-agnostic detection invariants for uncoded KV corruption."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aepk_paging.kv_page import KVPage


@dataclass(frozen=True)
class DetectorResult:
    flag: bool
    deviation: float
    tolerance: float


def attention_mass(page: KVPage, *, temperature: float = 1.0, top_fraction: float = 0.5) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    scores = np.linalg.norm(page.K.astype(np.float32), axis=1) / np.float32(temperature)
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    keep = max(1, int(np.ceil(weights.shape[0] * top_fraction)))
    return float(np.sum(weights[:keep]))


def attention_mass_detector(
    page: KVPage,
    *,
    expected_mass: float | None = None,
    tolerance: float = 0.05,
    temperature: float = 1.0,
    top_fraction: float = 0.5,
) -> DetectorResult:
    baseline = page.attention_mass if expected_mass is None else expected_mass
    current = attention_mass(page, temperature=temperature, top_fraction=top_fraction)
    deviation = abs(current - baseline)
    return DetectorResult(flag=deviation > tolerance, deviation=deviation, tolerance=tolerance)


def norm_ratio(page: KVPage) -> float:
    k_norm = np.linalg.norm(page.K.astype(np.float32))
    v_norm = np.linalg.norm(page.V.astype(np.float32))
    return float(k_norm / (v_norm + np.float32(1e-12)))


def norm_consistency_detector(
    page: KVPage,
    *,
    expected_ratio: float,
    tolerance: float = 0.05,
) -> DetectorResult:
    current = norm_ratio(page)
    deviation = abs(current - expected_ratio)
    return DetectorResult(flag=deviation > tolerance, deviation=deviation, tolerance=tolerance)


def confidence_proxy(logits: np.ndarray, *, surprise_threshold: float = 0.25) -> DetectorResult:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    probabilities = weights / np.sum(weights)
    surprise = float(-np.log(np.max(probabilities)))
    return DetectorResult(
        flag=surprise > surprise_threshold,
        deviation=surprise,
        tolerance=surprise_threshold,
    )
