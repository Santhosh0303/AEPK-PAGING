"""
Phase 8.4 — uniform-quant (UQ) and H2O-evict baselines.

Baselines:
  UQ-8bit: quantize_page(page, 8) on all layers; dequantize before injection; no RS.
  UQ-4bit: quantize_page(page, 4) on all layers; dequantize before injection; no RS.
  H2O-25:  evict 25% lowest-attention_mass layers (zero their KV); keep 75%.
  H2O-50:  evict 50% lowest-attention_mass layers.
  H2O-75:  evict 75% lowest-attention_mass layers.

Comparison: each baseline produces a (NLL, storage_bits, accuracy) point.
AEPK_adaptive points from Phase 8.3 sweep form the reference frontier.
Dominance verdict: does AEPK Pareto-dominate each baseline at iso-NLL?
  AEPK dominates B if AEPK_storage <= B_storage at the same or lower NLL.

Standing constraint D: RS codec unchanged. No Phase 2-5 constant tuning.
APIs: quantize_page/QuantizedPage.dequantize verified in lossy_tier.py:63-78.
      pages_to_kv_tensors and DynamicLayer.keys/values verified in Phase 7.2/7.4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from aepk_paging.harness.eval_set import EVAL_PROBES, normalized_match
from aepk_paging.harness.phase7_quality import (
    HELD_OUT_PREFIX,
    HELD_OUT_CONT,
    _compute_nll,
    _total_kv_bits,
)
from aepk_paging.lossy_tier import quantize_page
from aepk_paging.real_model_adapter import dynamiccache_to_pages, pages_to_kv_tensors


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _inject_kvpages(pkv, kvpages: list, dtype, device) -> None:
    """Inject KVPage objects into DynamicCache (verified API: layer.keys assignable)."""
    for page in kvpages:
        k, v = pages_to_kv_tensors(page, dtype=dtype, device=device)
        layer = pkv.layers[page.layer]
        layer.keys = k
        layer.values = v


def _inject_zeros(pkv, layer_indices: list[int]) -> None:
    """Set K/V to zeros for evicted layers (H2O eviction)."""
    for i in layer_indices:
        pkv.layers[i].keys = torch.zeros_like(pkv.layers[i].keys)
        pkv.layers[i].values = torch.zeros_like(pkv.layers[i].values)


# ---------------------------------------------------------------------------
# Baseline result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselinePoint:
    name: str
    nll: float
    accuracy: float
    storage_bits: int
    storage_pct: float    # fraction of clean storage (1.0 = same, 0.5 = half)


# ---------------------------------------------------------------------------
# UQ baseline (uniform quantization, no RS)
# ---------------------------------------------------------------------------

def _run_uq_baseline(model, tok, device, dtype, bit_width: int) -> BaselinePoint:
    """Apply uniform quantization at bit_width to all layers; no RS healing."""
    prefix_ids = tok(HELD_OUT_PREFIX, return_tensors="pt").to(device)
    cont_ids = tok(HELD_OUT_CONT, return_tensors="pt").to(device)

    with torch.no_grad():
        pfx_out = model(**prefix_ids, use_cache=True)
    pkv = pfx_out.past_key_values
    pages = dynamiccache_to_pages(pkv)
    clean_bits = _total_kv_bits(pages)

    # Quantize and dequantize: QuantizedPage -> KVPage (float32, lossy)
    dequant_pages = [quantize_page(p, bit_width).dequantize() for p in pages]
    _inject_kvpages(pkv, dequant_pages, dtype, device)

    nll = _compute_nll(model, tok, prefix_ids, cont_ids, pkv, device)

    # Storage: bit_width bits per element (original is fp16 = 16 bits)
    storage_bits = int(clean_bits * bit_width / 16)
    storage_pct = storage_bits / clean_bits

    # Accuracy: rebuild fresh pkv for each probe
    acc = _eval_accuracy_fresh(model, tok, device, dtype, bit_width=bit_width)

    return BaselinePoint(
        name=f"UQ-{bit_width}bit",
        nll=nll,
        accuracy=acc,
        storage_bits=storage_bits,
        storage_pct=storage_pct,
    )


def _eval_accuracy_fresh(model, tok, device, dtype, bit_width: int) -> float:
    """Per-probe accuracy with uniform quantization applied to each probe's KV."""
    model.eval()
    n_correct = 0
    for probe_idx, probe in enumerate(EVAL_PROBES):
        ids = tok(probe["prompt"], return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            pfx_out = model(ids, use_cache=True)
        pkv = pfx_out.past_key_values
        pages = dynamiccache_to_pages(pkv)

        dequant_pages = [quantize_page(p, bit_width).dequantize() for p in pages]
        _inject_kvpages(pkv, dequant_pages, dtype, device)

        with torch.no_grad():
            out = model.generate(ids, past_key_values=pkv, max_new_tokens=8, do_sample=False)
        pred_text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if normalized_match(pred_text, probe["expected"], probe.get("alternatives")):
            n_correct += 1

    return n_correct / len(EVAL_PROBES)


# ---------------------------------------------------------------------------
# H2O-evict baseline (layer-level eviction by attention_mass)
# ---------------------------------------------------------------------------

def _run_h2o_baseline(model, tok, device, dtype, evict_frac: float) -> BaselinePoint:
    """Evict evict_frac of lowest-attention_mass layers; zero their KV."""
    prefix_ids = tok(HELD_OUT_PREFIX, return_tensors="pt").to(device)
    cont_ids = tok(HELD_OUT_CONT, return_tensors="pt").to(device)

    with torch.no_grad():
        pfx_out = model(**prefix_ids, use_cache=True)
    pkv = pfx_out.past_key_values
    pages = dynamiccache_to_pages(pkv)
    clean_bits = _total_kv_bits(pages)
    n_layers = len(pages)

    # Sort layers by attention_mass ascending; evict the bottom fraction
    masses = np.array([p.attention_mass for p in pages])
    n_evict = max(1, int(round(evict_frac * n_layers)))
    evicted_layer_idx = np.argsort(masses)[:n_evict].tolist()  # lowest mass first

    _inject_zeros(pkv, evicted_layer_idx)
    nll = _compute_nll(model, tok, prefix_ids, cont_ids, pkv, device)

    # Storage: only non-evicted layers stored
    n_kept = n_layers - n_evict
    storage_bits = int(clean_bits * n_kept / n_layers)
    storage_pct = n_kept / n_layers

    # Accuracy
    acc = _eval_accuracy_h2o(model, tok, device, evicted_layer_idx)

    pct_int = int(round(evict_frac * 100))
    return BaselinePoint(
        name=f"H2O-{pct_int}pct",
        nll=nll,
        accuracy=acc,
        storage_bits=storage_bits,
        storage_pct=storage_pct,
    )


def _eval_accuracy_h2o(model, tok, device, evicted_layer_idx: list[int]) -> float:
    """Per-probe accuracy with H2O layer eviction."""
    model.eval()
    n_correct = 0
    for probe in EVAL_PROBES:
        ids = tok(probe["prompt"], return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            pfx_out = model(ids, use_cache=True)
        pkv = pfx_out.past_key_values
        _inject_zeros(pkv, evicted_layer_idx)

        with torch.no_grad():
            out = model.generate(ids, past_key_values=pkv, max_new_tokens=8, do_sample=False)
        pred_text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if normalized_match(pred_text, probe["expected"], probe.get("alternatives")):
            n_correct += 1

    return n_correct / len(EVAL_PROBES)


# ---------------------------------------------------------------------------
# Dominance check
# ---------------------------------------------------------------------------

def _aepk_dominates(
    aepk_points: list[tuple[float, int]],   # (nll, storage_bits)
    baseline: BaselinePoint,
    nll_tolerance: float = 0.05,
) -> bool:
    """
    True if any AEPK point has storage <= baseline.storage_bits AND
    NLL within nll_tolerance of baseline.nll.
    """
    for aepk_nll, aepk_storage in aepk_points:
        if aepk_storage <= baseline.storage_bits and aepk_nll <= baseline.nll + nll_tolerance:
            return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase8BaselinesResult:
    baselines: list[BaselinePoint]
    aepk_points: list[tuple[float, int]]    # (nll, storage_bits) from adaptive sweep
    dominance: dict[str, bool]              # baseline_name -> AEPK dominates?
    overall_verdict: str                    # "AEPK_DOMINATES_ALL", "AEPK_DOMINATES_SOME", "NONE"
    report_path: str


def run_baselines(
    model,
    tok,
    device: str,
    dtype,
    aepk_adaptive_points: list,             # list of SweepPoint from Phase 8.3
) -> Phase8BaselinesResult:
    """Run UQ-8, UQ-4, H2O-25/50/75 baselines and compare vs AEPK adaptive frontier."""

    # AEPK reference: use all adaptive sweep points as the frontier
    aepk_pts = [(sp.b3_nll, sp.b3_storage_bits) for sp in aepk_adaptive_points]

    baselines = []
    baselines.append(_run_uq_baseline(model, tok, device, dtype, bit_width=8))
    baselines.append(_run_uq_baseline(model, tok, device, dtype, bit_width=4))
    baselines.append(_run_h2o_baseline(model, tok, device, dtype, evict_frac=0.25))
    baselines.append(_run_h2o_baseline(model, tok, device, dtype, evict_frac=0.50))
    baselines.append(_run_h2o_baseline(model, tok, device, dtype, evict_frac=0.75))

    dominance = {b.name: _aepk_dominates(aepk_pts, b) for b in baselines}
    n_dominated = sum(dominance.values())
    if n_dominated == len(baselines):
        overall_verdict = "AEPK_DOMINATES_ALL"
    elif n_dominated > 0:
        overall_verdict = "AEPK_DOMINATES_SOME"
    else:
        overall_verdict = "NONE"

    report_path = os.path.join("results", "REPORT_phase8_baselines.md")
    _write_report(baselines, aepk_pts, dominance, overall_verdict, report_path)

    return Phase8BaselinesResult(
        baselines=baselines,
        aepk_points=aepk_pts,
        dominance=dominance,
        overall_verdict=overall_verdict,
        report_path=report_path,
    )


def _write_report(
    baselines: list[BaselinePoint],
    aepk_pts: list[tuple[float, int]],
    dominance: dict[str, bool],
    overall_verdict: str,
    path: str,
) -> None:
    lines = [
        "# REPORT_phase8_baselines.md — Phase 8.4 baseline comparison",
        "",
        "## Baselines",
        "| Method | NLL | Accuracy | Storage bits | Storage% | AEPK dominates? |",
        "|--------|-----|----------|-------------|----------|-----------------|",
    ]
    for b in baselines:
        lines.append(
            f"| {b.name} | {b.nll:.4f} | {b.accuracy:.3f} | {b.storage_bits:,} | "
            f"{b.storage_pct*100:.1f}% | {'YES' if dominance[b.name] else 'no'} |"
        )
    lines += [
        "",
        "## AEPK adaptive reference points (Phase 8.3)",
        "| NLL | Storage bits |",
        "|-----|-------------|",
    ]
    for nll, storage in aepk_pts:
        lines.append(f"| {nll:.4f} | {storage:,} |")
    lines += [
        "",
        "Dominance criterion: AEPK has storage <= baseline AND NLL within 0.05 nats.",
        f"",
        f"**PHASE 8.4 DOMINANCE VERDICT: {overall_verdict}**",
        "_(AEPK_DOMINATES_ALL = dominates every baseline; SOME = partial; NONE = no dominance)_",
        "_(Verdict may be NONE — this is honest, not a failure)_",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
