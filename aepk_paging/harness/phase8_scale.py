"""
Phase 8.5 — scale check: repeat crossover at 0.5B model and 2 context lengths.

Experiment matrix (4 cells):
  model_id × ctx_label:
    Qwen2.5-1.5B-Instruct × short   (~10 tokens)
    Qwen2.5-1.5B-Instruct × long    (~150 tokens)
    Qwen2.5-0.5B-Instruct × short
    Qwen2.5-0.5B-Instruct × long

Each cell: run B3 at SCALE_NOISE_LEVEL=0.2 (uniform crossover from Phase 8.2).
  on_pareto = (ΔNLL ≤ NLL_THRESHOLD) AND (b3_storage_bits < b0_storage_bits)

generalizes_verdict:
  GENERALIZES_ALL   — all 4 cells on_pareto
  GENERALIZES_SOME  — 1-3 cells on_pareto
  NONE              — 0 cells on_pareto

Honesty: verdict NOT asserted. NONE is allowed.

APIs: same as Phases 7-8.2. 0.5B uses same transformers DynamicCache — verified by
  checking pkv.layers post-load in run_scale_check().

Standing constraint D: RS codec unchanged; no Phase 2-5 constant changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from aepk_paging.coding import encode_rs_erasure_group, recover_rs_erasure
from aepk_paging.harness.phase7_quality import (
    HELD_OUT_CONT,
    HELD_OUT_PREFIX,
    _compute_nll,
    _inject_pages,
    _total_kv_bits,
)
from aepk_paging.harness.phase8_sweep import NLL_THRESHOLD
from aepk_paging.lossy_tier import quant_noise, page_mse
from aepk_paging.real_model_adapter import dynamiccache_to_pages, pages_to_kv_tensors
from aepk_paging.residency import ResidencyManager, TierCostModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALE_NOISE_LEVEL = 0.2   # uniform crossover from Phase 8.2

SHORT_PREFIX = HELD_OUT_PREFIX   # ~10 tokens
LONG_PREFIX = (
    "Large language models process text by computing attention over all previous tokens "
    "at each layer. The key-value cache stores these intermediate representations to avoid "
    "redundant computation during autoregressive generation. As sequence length grows, "
    "cache size scales linearly with the number of layers and tokens while quadratic "
    "cost is amortized across the batch. Modern systems therefore require adaptive eviction "
    "policies to manage the cache under memory constraints. High-frequency tokens and "
    "attention sinks retain their values in GPU memory while infrequent tokens migrate "
    "to compressed storage. A thermodynamic criterion based on Gibbs free energy determines "
    "which cache pages remain resident and which are quantized or evicted. Reed-Solomon "
    "erasure coding enables exact recovery of lost pages when GPU memory is reclaimed "
    "for computation. Artificial intelligence systems must handle memory"
)  # ~150 tokens (same ending as SHORT_PREFIX so CONT is grammatically consistent)

CONT = HELD_OUT_CONT

MODEL_IDS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
CTX_CONFIGS = [
    ("short", SHORT_PREFIX),
    ("long",  LONG_PREFIX),
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalePoint:
    model_id: str
    ctx_label: str           # "short" or "long"
    prefix_token_count: int
    b0_nll: float
    b0_storage_bits: int
    b3_nll: float
    b3_storage_bits: int
    nll_delta: float
    storage_savings_pct: float
    on_pareto: bool


@dataclass(frozen=True)
class ScaleCheckResult:
    scale_points: list[ScalePoint]
    generalizes_verdict: str   # GENERALIZES_ALL / GENERALIZES_SOME / NONE
    report_path: str


# ---------------------------------------------------------------------------
# Single-cell B3 runner
# ---------------------------------------------------------------------------

def _run_scale_cell(
    model,
    tok,
    device: str,
    dtype,
    prefix_text: str,
    cont_text: str,
    model_id: str,
    ctx_label: str,
) -> ScalePoint:
    """Run B0 and B3 at SCALE_NOISE_LEVEL for one (model, context) cell."""
    prefix_ids = tok(prefix_text, return_tensors="pt").to(device)
    cont_ids   = tok(cont_text,   return_tensors="pt").to(device)
    prefix_token_count = int(prefix_ids.input_ids.shape[1])

    # -- B0 (clean) --
    with torch.no_grad():
        pfx0 = model(**prefix_ids, use_cache=True)
    pkv0   = pfx0.past_key_values
    pages0 = dynamiccache_to_pages(pkv0)
    b0_storage = _total_kv_bits(pages0)
    b0_nll     = _compute_nll(model, tok, prefix_ids, cont_ids, pkv0, device)

    # -- B3 (noise → RS recover → residency) --
    with torch.no_grad():
        pfx3 = model(**prefix_ids, use_cache=True)
    pkv3   = pfx3.past_key_values
    pages3 = dynamiccache_to_pages(pkv3)

    rs_group = encode_rs_erasure_group(pages3, num_parity=2)

    damaged: list = []
    mses: list[float] = []
    for i, page in enumerate(pages3):
        if SCALE_NOISE_LEVEL == 0.0:
            damaged.append(page)
            mses.append(0.0)
        else:
            dam, mse = quant_noise(page, level=SCALE_NOISE_LEVEL, seed=3000 + i)
            damaged.append(dam)
            mses.append(float(mse))

    # RS recover worst 2 pages
    worst_2_ids = [pages3[i].page_id for i in np.argsort(mses)[-2:]]
    try:
        recovered = recover_rs_erasure(rs_group, worst_2_ids)
        for pid, rpage in recovered.items():
            idx = next(j for j, p in enumerate(damaged) if p.page_id == pid)
            damaged[idx] = rpage
    except Exception:
        pass

    cost_model = TierCostModel()
    manager    = ResidencyManager(cost_model=cost_model)
    plan       = manager.plan(pages=damaged, budget_bits=b0_storage, erasure_recovery_bound=2)
    parity_bits = int(2 * (pages3[0].K.nbytes + pages3[0].V.nbytes) * 8)
    b3_storage  = plan.total_storage_bits + parity_bits

    _inject_pages(pkv3, damaged, dtype, device)
    b3_nll = _compute_nll(model, tok, prefix_ids, cont_ids, pkv3, device)

    nll_delta     = b3_nll - b0_nll
    savings_pct   = (b0_storage - b3_storage) / b0_storage * 100.0
    on_pareto     = (nll_delta <= NLL_THRESHOLD) and (b3_storage < b0_storage)

    return ScalePoint(
        model_id=model_id,
        ctx_label=ctx_label,
        prefix_token_count=prefix_token_count,
        b0_nll=b0_nll,
        b0_storage_bits=b0_storage,
        b3_nll=b3_nll,
        b3_storage_bits=b3_storage,
        nll_delta=nll_delta,
        storage_savings_pct=savings_pct,
        on_pareto=on_pareto,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_scale_check(device: str, dtype) -> ScaleCheckResult:
    """
    Load each model once, run short+long context, unload, repeat for next model.
    Returns ScaleCheckResult with 4 ScalePoints.
    """
    scale_points: list[ScalePoint] = []

    for model_id in MODEL_IDS:
        tok   = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
        model.eval()

        # Verify DynamicCache API consistent for this model
        _verify_dynamic_cache(model, tok, device)

        for ctx_label, prefix_text in CTX_CONFIGS:
            pt = _run_scale_cell(
                model, tok, device, dtype,
                prefix_text, CONT,
                model_id=model_id,
                ctx_label=ctx_label,
            )
            scale_points.append(pt)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    n_pareto = sum(pt.on_pareto for pt in scale_points)
    if n_pareto == len(scale_points):
        generalizes_verdict = "GENERALIZES_ALL"
    elif n_pareto > 0:
        generalizes_verdict = "GENERALIZES_SOME"
    else:
        generalizes_verdict = "NONE"

    report_path = os.path.join("results", "REPORT_phase8_scale.md")
    _write_report(scale_points, generalizes_verdict, report_path)

    return ScaleCheckResult(
        scale_points=scale_points,
        generalizes_verdict=generalizes_verdict,
        report_path=report_path,
    )


def _verify_dynamic_cache(model, tok, device) -> None:
    """Confirm pkv.layers accessible for this model (same DynamicCache API)."""
    probe_ids = tok("hello", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(probe_ids, use_cache=True)
    pkv = out.past_key_values
    assert hasattr(pkv, "layers"), (
        f"Model {model.config.name_or_path}: past_key_values has no .layers attribute — "
        f"unexpected cache type {type(pkv)}"
    )
    assert len(pkv.layers) == model.config.num_hidden_layers, (
        f"layers count mismatch: {len(pkv.layers)} vs config {model.config.num_hidden_layers}"
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(
    scale_points: list[ScalePoint],
    generalizes_verdict: str,
    path: str,
) -> None:
    lines = [
        "# REPORT_phase8_scale.md — Phase 8.5 scale generalization check",
        "",
        f"Noise level: {SCALE_NOISE_LEVEL} (uniform crossover from Phase 8.2)",
        f"NLL threshold: {NLL_THRESHOLD}",
        f"Pareto criterion: ΔNLL ≤ threshold AND b3_storage < b0_storage",
        "",
        "## Results",
        "| Model | Ctx | Prefix toks | B0_NLL | B3_NLL | ΔNLL | savings% | Pareto |",
        "|-------|-----|-------------|--------|--------|------|----------|--------|",
    ]
    for pt in scale_points:
        model_short = pt.model_id.split("/")[-1]
        lines.append(
            f"| {model_short} | {pt.ctx_label} | {pt.prefix_token_count} "
            f"| {pt.b0_nll:.4f} | {pt.b3_nll:.4f} | {pt.nll_delta:+.4f} "
            f"| {pt.storage_savings_pct:+.1f}% | {'YES' if pt.on_pareto else 'no'} |"
        )

    n_pareto = sum(pt.on_pareto for pt in scale_points)
    lines += [
        "",
        f"Cells on Pareto: {n_pareto}/{len(scale_points)}",
        "",
        f"**PHASE 8.5 GENERALIZATION VERDICT: {generalizes_verdict}**",
        "_(GENERALIZES_ALL = crossover exists at every scale; SOME = partial; NONE = no crossover)_",
        "_(NONE is honest — not a build failure)_",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
