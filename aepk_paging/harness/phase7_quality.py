"""
Phase 7.4 — real quality metric harness.

Measures perplexity/NLL on fixed held-out text with baselines B0-B3 on real KV,
runs a task-accuracy probe, applies the same rate-distortion gate as the simulator
([Shannon]: total_cost(lambda) = storage_bits + lambda * residual_error), and
reports a harness-computed verdict.

Compute caveat: RS encode/decode CPU time is recorded and reported but NOT
mixed into the R-D gate (the gate is storage_bits vs residual_error only).

Verified APIs (2026-07-01):
  - DynamicLayer.keys / .values: directly assignable (layer.keys = new_tensor).
  - NLL: computed over continuation tokens given cached prefix KV.
  - transformers 5.12.1, torch 2.5.1+cu121.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from aepk_paging.real_model_adapter import dynamiccache_to_pages, pages_to_kv_tensors
from aepk_paging.lossy_tier import quant_noise, quantize_page, page_mse
from aepk_paging.coding import encode_rs_erasure_group, recover_rs_erasure
from aepk_paging.residency import ResidencyManager, TierCostModel
from aepk_paging.kv_page import ResidencyTier


# ---------------------------------------------------------------------------
# Fixed held-out text (deterministic; never changed)
# ---------------------------------------------------------------------------
HELD_OUT_TEXT = (
    "Artificial intelligence systems must handle memory efficiently "
    "to sustain long context reasoning under hardware constraints."
)
TASK_PROBE_PROMPT = "What is the capital of France? Answer in one word:"
TASK_PROBE_EXPECTED = "Paris"

# Prefix / continuation split (first ~half of HELD_OUT_TEXT)
HELD_OUT_PREFIX = "Artificial intelligence systems must handle memory"
HELD_OUT_CONT = " efficiently to sustain long context reasoning under hardware constraints."


@dataclass(frozen=True)
class BaselineResult:
    name: str
    nll: float             # mean NLL per token on continuation
    storage_bits: int      # KV storage after pipeline
    residual_mse: float    # MSE vs original KV
    compute_sec: float     # RS encode/decode time (caveat; not in RD gate)


@dataclass(frozen=True)
class Phase7QualityResult:
    baselines: list[BaselineResult]
    task_probe_correct_b0: bool
    task_probe_correct_b3: bool
    # R-D gate (same as simulator Phase 6)
    gate_verdict: str      # "PASS" or "FAIL"
    gate_lambda_win_range: tuple[float, float] | None
    report_lines: list[str]


def _compute_nll(model, tok, prefix_ids, cont_ids, pkv, device) -> float:
    """Run continuation with given prefix cache; return mean NLL per token.
    cont_ids: BatchEncoding; extract .input_ids for model forward.
    """
    ids = cont_ids.input_ids if hasattr(cont_ids, "input_ids") else cont_ids
    with torch.no_grad():
        out = model(ids, past_key_values=pkv)
    logits = out.logits[0]  # [n_cont, vocab]
    lp = torch.nn.functional.log_softmax(logits, dim=-1)
    n = ids.shape[1]
    if n <= 1:
        return 0.0
    nll = 0.0
    for i in range(n - 1):
        nll += -lp[i, ids[0, i + 1]].item()
    return nll / (n - 1)


def _inject_pages(pkv, pages: list, dtype, device, batch_idx: int = 0) -> None:
    """Inject KVPage K/V back into DynamicCache layers in-place."""
    for page in pages:
        k, v = pages_to_kv_tensors(page, dtype=dtype, device=device)
        layer = pkv.layers[page.layer]
        layer.keys = k    # [1, num_kv_heads, seq_len, head_dim]
        layer.values = v


def _clone_pkv_pages(pkv) -> list:
    """Extract KVPages from a DynamicCache (all layers)."""
    return dynamiccache_to_pages(pkv)


def _total_kv_bits(pages: list) -> int:
    return sum(int((p.K.nbytes + p.V.nbytes) * 8) for p in pages)


def run_phase7_quality(model, tok, device, dtype) -> Phase7QualityResult:
    """
    Compute B0-B3 NLL + task probe + R-D gate on real Qwen2.5-1.5B-Instruct KV.
    """
    prefix_ids = tok(HELD_OUT_PREFIX, return_tensors="pt").to(device)
    cont_ids = tok(HELD_OUT_CONT, return_tensors="pt").to(device)

    # -----------------------------------------------------------------------
    # Get clean prefix cache (re-used by all baselines)
    # -----------------------------------------------------------------------
    with torch.no_grad():
        prefix_out = model(**prefix_ids, use_cache=True)
    pkv_clean = prefix_out.past_key_values
    clean_pages = dynamiccache_to_pages(pkv_clean)
    clean_bits = _total_kv_bits(clean_pages)

    # -----------------------------------------------------------------------
    # B0 — no protection; clean KV; reference NLL
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    nll_b0 = _compute_nll(model, tok, prefix_ids, cont_ids, pkv_clean, device)
    compute_b0 = time.perf_counter() - t0
    b0 = BaselineResult("B0_no_protection", nll_b0, clean_bits, 0.0, compute_b0)

    # -----------------------------------------------------------------------
    # B1 — keep all RESIDENT; same clean KV; same NLL as B0
    # -----------------------------------------------------------------------
    b1 = BaselineResult("B1_all_resident", nll_b0, clean_bits, 0.0, compute_b0)

    # -----------------------------------------------------------------------
    # B2 — erasure parity only; evict 1 layer; recover bit-exact; measure NLL
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    with torch.no_grad():
        pfx2 = model(**prefix_ids, use_cache=True)
    pkv_b2 = pfx2.past_key_values
    pages_b2 = dynamiccache_to_pages(pkv_b2)
    # Erasure: encode all layers, evict first, recover
    rs_group = encode_rs_erasure_group(pages_b2, num_parity=1)
    evict_id = pages_b2[0].page_id
    recovered = recover_rs_erasure(rs_group, [evict_id])
    # Replace evicted page with recovered page
    pages_b2[0] = recovered[evict_id]
    _inject_pages(pkv_b2, pages_b2, dtype, device)
    compute_b2 = time.perf_counter() - t0

    nll_b2 = _compute_nll(model, tok, prefix_ids, cont_ids, pkv_b2, device)
    # MSE vs original: recovered page = bit-exact → 0
    mse_b2 = 0.0
    parity_bits = int(pages_b2[0].K.nbytes + pages_b2[0].V.nbytes) * 8  # 1 parity page
    storage_b2 = clean_bits + parity_bits
    b2 = BaselineResult("B2_erasure_parity", nll_b2, storage_b2, mse_b2, compute_b2)

    # -----------------------------------------------------------------------
    # B3 — full AEPK: quant_noise damage → detect → RS erasure → residency
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    with torch.no_grad():
        pfx3 = model(**prefix_ids, use_cache=True)
    pkv_b3 = pfx3.past_key_values
    pages_b3 = dynamiccache_to_pages(pkv_b3)

    # Encode erasure group BEFORE damage (parity computed on clean pages)
    rs_group_b3 = encode_rs_erasure_group(pages_b3, num_parity=2)

    # Phase 2: quant_noise damage on all pages
    damaged_pages = []
    mses = []
    for i, page in enumerate(pages_b3):
        damaged, mse = quant_noise(page, level=0.3, seed=1234 + i)
        damaged_pages.append(damaged)
        mses.append(mse)
    mean_damaged_mse = float(np.mean(mses))

    # Phase 3: RS erasure — recover the 2 most damaged pages
    worst_2_ids = [pages_b3[i].page_id for i in np.argsort(mses)[-2:]]
    try:
        recovered_b3 = recover_rs_erasure(rs_group_b3, worst_2_ids)
        for pid, rpage in recovered_b3.items():
            idx = next(j for j, p in enumerate(damaged_pages) if p.page_id == pid)
            damaged_pages[idx] = rpage
        recovered_mse = float(np.mean([
            page_mse(orig, dam) for orig, dam in zip(pages_b3, damaged_pages)
        ]))
    except Exception:
        recovered_mse = mean_damaged_mse

    # Phase 5: residency plan (capacity-coupled)
    cost_model = TierCostModel()
    manager = ResidencyManager(cost_model=cost_model)
    plan = manager.plan(
        pages=damaged_pages,
        budget_bits=clean_bits,
        erasure_recovery_bound=2,
    )
    storage_b3 = plan.total_storage_bits + int(
        2 * (pages_b3[0].K.nbytes + pages_b3[0].V.nbytes) * 8  # 2 parity pages
    )
    compute_b3 = time.perf_counter() - t0

    _inject_pages(pkv_b3, damaged_pages, dtype, device)
    nll_b3 = _compute_nll(model, tok, prefix_ids, cont_ids, pkv_b3, device)
    b3 = BaselineResult("B3_full_AEPK", nll_b3, storage_b3, recovered_mse, compute_b3)

    # -----------------------------------------------------------------------
    # Task-accuracy probe
    # -----------------------------------------------------------------------
    probe_ids = tok(TASK_PROBE_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_b0 = model.generate(probe_ids.input_ids, max_new_tokens=5, do_sample=False)
        gen_b0_text = tok.decode(gen_b0[0][probe_ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

    task_probe_b0 = TASK_PROBE_EXPECTED.lower() in gen_b0_text.lower()

    # B3 task probe: damage prefix cache, generate
    with torch.no_grad():
        pfx_tp = model(**probe_ids, use_cache=True)
    pkv_tp = pfx_tp.past_key_values
    pages_tp = dynamiccache_to_pages(pkv_tp)
    rs_tp = encode_rs_erasure_group(pages_tp, num_parity=1)
    dam_tp = []
    for i, page in enumerate(pages_tp):
        dam, _ = quant_noise(page, level=0.3, seed=9999 + i)
        dam_tp.append(dam)
    try:
        rec_tp = recover_rs_erasure(rs_tp, [pages_tp[0].page_id])
        dam_tp[0] = rec_tp[pages_tp[0].page_id]
    except Exception:
        pass
    _inject_pages(pkv_tp, dam_tp, dtype, device)
    with torch.no_grad():
        gen_b3 = model.generate(
            probe_ids.input_ids,
            past_key_values=pkv_tp,
            max_new_tokens=5,
            do_sample=False,
        )
    gen_b3_text = tok.decode(gen_b3[0][probe_ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    task_probe_b3 = TASK_PROBE_EXPECTED.lower() in gen_b3_text.lower()

    # -----------------------------------------------------------------------
    # Rate-distortion gate (same [Shannon] gate as Phase 6 simulator)
    # -----------------------------------------------------------------------
    lambdas = np.logspace(0, 9, 50)
    b3_wins = 0
    win_lambdas = []
    for lam in lambdas:
        cost_b2 = storage_b2 + lam * mse_b2
        cost_b3 = storage_b3 + lam * recovered_mse
        if cost_b3 < cost_b2 and nll_b3 <= nll_b2 + 0.5:
            b3_wins += 1
            win_lambdas.append(lam)

    gate_verdict = "PASS" if b3_wins > 0 else "FAIL"
    gate_lambda_range = (float(min(win_lambdas)), float(max(win_lambdas))) if win_lambdas else None

    # -----------------------------------------------------------------------
    # Report lines
    # -----------------------------------------------------------------------
    lines = [
        "## Real-model validation (Phase 7.4)",
        f"Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)",
        f"Held-out text: '{HELD_OUT_PREFIX}...'",
        f"",
        "| Baseline | NLL | Storage bits | Residual MSE | Compute (s) |",
        "|----------|-----|-------------|--------------|-------------|",
    ]
    for b in [b0, b1, b2, b3]:
        lines.append(
            f"| {b.name} | {b.nll:.4f} | {b.storage_bits:,} | {b.residual_mse:.6f} | {b.compute_sec:.3f} |"
        )
    lines += [
        f"",
        f"Task probe: '{TASK_PROBE_PROMPT}'",
        f"  B0 answer: '{gen_b0_text}' → {'CORRECT' if task_probe_b0 else 'WRONG'}",
        f"  B3 answer: '{gen_b3_text}' → {'CORRECT' if task_probe_b3 else 'WRONG'}",
        f"",
        f"Rate-distortion gate: B3 wins {b3_wins}/50 lambda points",
        f"Lambda win range: {gate_lambda_range}",
        f"COMPUTE CAVEAT: RS encode/decode CPU time reported above; NOT mixed into RD gate.",
        f"",
        f"**REAL-MODEL GATE VERDICT: {gate_verdict}**",
    ]

    return Phase7QualityResult(
        baselines=[b0, b1, b2, b3],
        task_probe_correct_b0=task_probe_b0,
        task_probe_correct_b3=task_probe_b3,
        gate_verdict=gate_verdict,
        gate_lambda_win_range=gate_lambda_range,
        report_lines=lines,
    )
