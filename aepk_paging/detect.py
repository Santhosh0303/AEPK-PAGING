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
    """Return the Gibbs-fingerprint mass over the leading page tokens."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    weights = attention_distribution(page, temperature=temperature)
    keep = max(1, int(np.ceil(weights.shape[0] * top_fraction)))
    return float(np.sum(weights[:keep]))


def attention_distribution(page: KVPage, *, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scores = np.linalg.norm(page.K.astype(np.float32), axis=1) / np.float32(temperature)
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def attention_mass_detector(
    page: KVPage,
    *,
    expected_mass: float | None = None,
    tolerance: float = 0.05,
    temperature: float = 1.0,
    top_fraction: float = 0.5,
) -> DetectorResult:
    """Gibbs-fingerprint drift detector vs stored clean baseline."""
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


def fixed_kv_readout_logits(
    page: KVPage,
    *,
    seed: int,
    num_logits: int = 3,
    head_scale: float = 3.0,
) -> np.ndarray:
    if num_logits < 2:
        raise ValueError("num_logits must be at least 2")
    K = page.K.astype(np.float32)
    V = page.V.astype(np.float32)
    d = K.shape[1]
    rng = np.random.default_rng(seed)
    q = rng.normal(loc=0.0, scale=1.0, size=d).astype(np.float32)
    head = (
        rng.normal(loc=0.0, scale=1.0, size=(d, num_logits)).astype(np.float32)
        * np.float32(head_scale)
    )
    scores = (K @ q) / np.float32(d**0.5)
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    output = weights @ V
    return output @ head


def finiteness_detector(page: KVPage) -> DetectorResult:
    """Content-agnostic corruption check: valid KV is finite. Any NaN/inf in K or V
    (e.g. an exponent bit-flip) is corruption -> flag. deviation = count of non-finite cells."""
    nonfinite = int(
        np.count_nonzero(~np.isfinite(page.K)) + np.count_nonzero(~np.isfinite(page.V))
    )
    return DetectorResult(flag=nonfinite > 0, deviation=float(nonfinite), tolerance=0.0)
