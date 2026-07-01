"""
Phase 8.2 — quant_noise sweep: NLL + task_accuracy + storage for B0 and B3
across noise_levels {0.0, 0.05, 0.1, 0.2, 0.3, 0.5}.

Emits results/REPORT_phase8_sweep.md with:
  - per-level table: NLL, task_accuracy, storage, ΔNLL, Δacc, savings_pct
  - harness-computed Pareto frontier: levels where ΔNLL ≤ 0.5 AND B3 saves storage
  - crossover_level: max noise_level on the frontier (None if AEPK never worthwhile)
  - overall verdict: PASS if any crossover exists, FAIL otherwise

Standing constraint D (per HITL): do NOT strengthen RS; floor is lossy source-coding.
No Phase 2-5 constant tuning. Gate allowed to FAIL.

APIs: all verified in Phase 7.4 and Phase 8.1. No new external API introduced.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from aepk_paging.coding import encode_rs_erasure_group, recover_rs_erasure
from aepk_paging.harness.eval_set import run_task_eval_b0, run_task_eval_b3
from aepk_paging.harness.phase7_quality import (
    HELD_OUT_PREFIX,
    HELD_OUT_CONT,
    _compute_nll,
    _inject_pages,
    _total_kv_bits,
)
from aepk_paging.kv_page import ResidencyTier
from aepk_paging.lossy_tier import quant_noise, page_mse
from aepk_paging.real_model_adapter import dynamiccache_to_pages, pages_to_kv_tensors
from aepk_paging.residency import ResidencyManager, TierCostModel

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

# R-D gate threshold (same as Phase 6 / Phase 7.4 — not tuned)
NLL_THRESHOLD = 0.5


@dataclass(frozen=True)
class SweepPoint:
    noise_level: float
    b0_nll: float
    b0_accuracy: float
    b0_storage_bits: int
    b3_nll: float
    b3_accuracy: float
    b3_storage_bits: int
    b3_residual_mse: float
    # Derived fields (harness-computed from above)
    nll_delta: float         # b3_nll - b0_nll
    acc_delta: float         # b3_accuracy - b0_accuracy (negative = degraded)
    storage_savings_pct: float   # (b0_bits - b3_bits) / b0_bits * 100
    on_pareto: bool          # nll_delta <= NLL_THRESHOLD AND b3 saves storage


@dataclass(frozen=True)
class SweepResult:
    sweep_points: list[SweepPoint]
    pareto_frontier: list[float]     # noise_levels on Pareto frontier
    crossover_level: float | None    # max noise_level in pareto_frontier
    overall_verdict: str             # "PASS" or "FAIL"
    report_path: str


def _b3_nll_at_level(model, tok, device, dtype, noise_level: float) -> tuple[float, int, float]:
    """Return (b3_nll, b3_storage_bits, b3_residual_mse) at the given noise_level."""
    prefix_ids = tok(HELD_OUT_PREFIX, return_tensors="pt").to(device)
    cont_ids = tok(HELD_OUT_CONT, return_tensors="pt").to(device)

    with torch.no_grad():
        pfx3 = model(**prefix_ids, use_cache=True)
    pkv_b3 = pfx3.past_key_values
    pages_b3 = dynamiccache_to_pages(pkv_b3)
    clean_bits = _total_kv_bits(pages_b3)

    rs_group_b3 = encode_rs_erasure_group(pages_b3, num_parity=2)

    damaged_pages = []
    mses = []
    for i, page in enumerate(pages_b3):
        if noise_level == 0.0:
            damaged_pages.append(page)
            mses.append(0.0)
        else:
            dam, mse = quant_noise(page, level=noise_level, seed=1234 + i)
            damaged_pages.append(dam)
            mses.append(float(mse))

    if noise_level > 0.0:
        worst_2_ids = [pages_b3[i].page_id for i in np.argsort(mses)[-2:]]
        try:
            recovered = recover_rs_erasure(rs_group_b3, worst_2_ids)
            for pid, rpage in recovered.items():
                idx = next(j for j, p in enumerate(damaged_pages) if p.page_id == pid)
                damaged_pages[idx] = rpage
        except Exception:
            pass

    recovered_mse = float(np.mean([
        page_mse(orig, dam) for orig, dam in zip(pages_b3, damaged_pages)
    ]))

    cost_model = TierCostModel()
    manager = ResidencyManager(cost_model=cost_model)
    plan = manager.plan(
        pages=damaged_pages,
        budget_bits=clean_bits,
        erasure_recovery_bound=2,
    )
    parity_bits = int(2 * (pages_b3[0].K.nbytes + pages_b3[0].V.nbytes) * 8)
    storage_b3 = plan.total_storage_bits + parity_bits

    _inject_pages(pkv_b3, damaged_pages, dtype, device)
    nll_b3 = _compute_nll(model, tok, prefix_ids, cont_ids, pkv_b3, device)
    return nll_b3, storage_b3, recovered_mse


def run_sweep(model, tok, device: str, dtype, noise_levels: list[float] | None = None) -> SweepResult:
    """
    Run the quant_noise sweep. B0 runs once; B3 runs at each noise level.
    Writes REPORT_phase8_sweep.md. Returns SweepResult.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    # -- B0 baseline (runs once — independent of noise_level) --
    prefix_ids = tok(HELD_OUT_PREFIX, return_tensors="pt").to(device)
    cont_ids = tok(HELD_OUT_CONT, return_tensors="pt").to(device)

    with torch.no_grad():
        pfx0 = model(**prefix_ids, use_cache=True)
    pkv0 = pfx0.past_key_values
    pages0 = dynamiccache_to_pages(pkv0)
    b0_storage_bits = _total_kv_bits(pages0)
    b0_nll = _compute_nll(model, tok, prefix_ids, cont_ids, pkv0, device)

    b0_eval = run_task_eval_b0(model, tok, device)
    b0_accuracy = b0_eval.accuracy

    # -- B3 at each noise level --
    sweep_points: list[SweepPoint] = []

    for level in noise_levels:
        b3_nll, b3_storage_bits, b3_mse = _b3_nll_at_level(model, tok, device, dtype, level)
        b3_eval = run_task_eval_b3(model, tok, device, dtype, noise_level=level)
        b3_accuracy = b3_eval.accuracy

        nll_delta = b3_nll - b0_nll
        acc_delta = b3_accuracy - b0_accuracy
        savings_pct = (b0_storage_bits - b3_storage_bits) / b0_storage_bits * 100.0
        on_pareto = (nll_delta <= NLL_THRESHOLD) and (b3_storage_bits < b0_storage_bits)

        sweep_points.append(SweepPoint(
            noise_level=level,
            b0_nll=b0_nll,
            b0_accuracy=b0_accuracy,
            b0_storage_bits=b0_storage_bits,
            b3_nll=b3_nll,
            b3_accuracy=b3_accuracy,
            b3_storage_bits=b3_storage_bits,
            b3_residual_mse=b3_mse,
            nll_delta=nll_delta,
            acc_delta=acc_delta,
            storage_savings_pct=savings_pct,
            on_pareto=on_pareto,
        ))

    # -- Pareto frontier and crossover (harness-computed) --
    pareto_frontier = [sp.noise_level for sp in sweep_points if sp.on_pareto]
    crossover_level = float(max(pareto_frontier)) if pareto_frontier else None
    overall_verdict = "PASS" if crossover_level is not None else "FAIL"

    # -- Write report --
    report_path = os.path.join("results", "REPORT_phase8_sweep.md")
    _write_report(sweep_points, pareto_frontier, crossover_level, overall_verdict, report_path)

    return SweepResult(
        sweep_points=sweep_points,
        pareto_frontier=pareto_frontier,
        crossover_level=crossover_level,
        overall_verdict=overall_verdict,
        report_path=report_path,
    )


def _write_report(
    sweep_points: list[SweepPoint],
    pareto_frontier: list[float],
    crossover_level: float | None,
    overall_verdict: str,
    path: str,
) -> None:
    lines = [
        "# REPORT_phase8_sweep.md — Phase 8.2 quant_noise sweep",
        "",
        "Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)",
        f"Noise levels swept: {[sp.noise_level for sp in sweep_points]}",
        f"NLL threshold (unchanged from Phase 6 / 7): {NLL_THRESHOLD}",
        "",
        "## Per-level results",
        "",
        "| noise | B0_NLL | B3_NLL | ΔNLL | B0_acc | B3_acc | Δacc | B3_storage_bits | savings_pct | Pareto |",
        "|-------|--------|--------|------|--------|--------|------|-----------------|-------------|--------|",
    ]
    for sp in sweep_points:
        lines.append(
            f"| {sp.noise_level:.2f} | {sp.b0_nll:.4f} | {sp.b3_nll:.4f} | "
            f"{sp.nll_delta:+.4f} | {sp.b0_accuracy:.3f} | {sp.b3_accuracy:.3f} | "
            f"{sp.acc_delta:+.3f} | {sp.b3_storage_bits:,} | {sp.storage_savings_pct:+.1f}% | "
            f"{'YES' if sp.on_pareto else 'no'} |"
        )
    lines += [
        "",
        "## Pareto frontier",
        f"Noise levels where ΔNLL ≤ {NLL_THRESHOLD} AND B3 saves storage vs B0:",
        f"  {pareto_frontier}",
        "",
        f"Crossover level (max Pareto noise): {crossover_level}",
        "",
        "Interpretation: below the crossover level, AEPK's storage savings come at",
        "acceptable NLL cost (≤0.5 nats). Above it, the damage exceeds the threshold.",
        "",
        "COMPUTE CAVEAT: RS encode/decode CPU time not measured in sweep (same caveat as Phase 7.4).",
        "",
        f"**PHASE 8 SWEEP VERDICT: {overall_verdict}**",
        f"_(PASS = crossover exists; FAIL = AEPK never within NLL threshold at any noise level)_",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
