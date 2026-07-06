"""Phase 10 SNR CAMPAIGN (steps 19-21) — the mechanism day.

Derives the corruption-tolerance behaviour of the compression-floor law from a signal-to-noise
argument on the attention logit, and tests four predictions (P1-P4).

THEORY (HITL-derived from lossy_tier.py:90-105; recorded verbatim in PREREG_phase10_snr.md):
`quant_noise` adds ABSOLUTE Gaussian noise, sigma = level, per K/V component. Attention logit
l = q.k/sqrt(d). The SIGNAL logit is coherent over d dims: scale = RMS_q*RMS_K*cos(theta)*sqrt(d)
(grows with sqrt(head_dim)). The NOISE logit q.eps/sqrt(d) is incoherent: std = RMS_q*level
(flat in d). SNR = sqrt(d)*RMS_K*cos/level -> retention collapses when noise reaches the logit
gap -> CRITICAL LEVEL LAW: L_c = C * sqrt(head_dim) * RMS_K (C global, absorbs alignment +
softmax scale).

Predictions:
  P1 out-of-sample crossover ratio (step 20 gate): qwen0.5b crossover predicted from qwen1.5b
     calibration by the L_c law.
  P2 SNR score s = sqrt(head_dim)*RMS_K ranks the 7 grid models (EXPLORATORY — retentions seen).
  P3 per-layer damage anticorrelates with per-layer MEAN key RMS (magnitude is the susceptibility
     variable; the FD moonshot asked VARIANCE).
  P4 under RELATIVE (multiplicative) noise RMS_K cancels -> tolerance follows pure sqrt(d)
     (step 21).

Honesty spine S9: zero edits to Phase 2-5 source. New harness file only. Reuses quant_noise,
dynamiccache_to_pages, _inject_pages, _decode_under_cache, normalized_match, get_combined_probes,
get_large_probes, phase10_grid.normalize_answer, phase10_fd.spearman_rho, phase10_stats.run_stats.
Deterministic clean-stat measurement (no RNG) -> byte-identical. Runtime verdict f-strings; tests
assert line-exists never values; ALLOWED-to-FAIL (a refuted derived law is a publishable
falsification, reported at equal prominence).
"""

from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from aepk_paging.harness.phase10_fd import spearman_rho

# ---- calibration constants (all FIXED before any GPU measurement) -----------
CAL_CROSSOVER = 0.398       # qwen1.5b LARGE-pool crossover mu (PREREG v3, REPORT_phase10_stats.md)
CAL_CI = 0.105              # its 95% CI half-width (PREREG v3)
CAL_HEAD_DIM = 128          # qwen1.5b head_dim
PRED_HEAD_DIM = 64          # qwen0.5b head_dim
CAL_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PRED_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# P3 directional gate (fixed BEFORE computing rho): anticorrelation predicted.
SNR_FD_SUPPORT = -0.5       # rho <= this => supported (strong anticorrelation)
SNR_FD_REFUTE = -0.2        # rho >= this => refuted (no anticorrelation)

# P1 band tolerance scale — the calibration CI scales proportionally with the prediction.
P1_TOL_SCALE = 0.105        # == CAL_CI; proportional band coefficient (kept explicit for the test)

GRID_ROWS_PATH = "results/grid_v2_run1.json"
FD_V2_REPORT_PATH = "results/REPORT_phase10_fd_v2.md"


# ============================================================================
# deterministic, CPU-testable math (no model)
# ============================================================================

def rms(x) -> float:
    """Root-mean-square over ALL elements of an array (float64 accumulation)."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(a * a)))


def page_key_rms(page) -> float:
    return rms(page.K)


def page_value_rms(page) -> float:
    return rms(page.V)


def snr_score(head_dim: int, rms_k: float) -> float:
    """SNR susceptibility score s = sqrt(head_dim) * RMS_K (the L_c law up to the global C)."""
    return float(math.sqrt(head_dim) * rms_k)


def _rel_gap(values, tol_flags) -> float:
    """Relative separating-band width for a single-threshold split of `values` by `tol_flags`:
    (min value among tolerant - max value among intolerant) / (range of all values). Positive iff
    one threshold cleanly separates tolerant (high) from intolerant (low); <=0 otherwise."""
    v = np.asarray(values, dtype=np.float64)
    tol = np.asarray(tol_flags, dtype=bool)
    if tol.all() or (~tol).all():
        return float("nan")                 # degenerate: only one class
    lo_of_tol = float(v[tol].min())
    hi_of_intol = float(v[~tol].max())
    rng = float(v.max() - v.min())
    if rng == 0.0:
        return float("nan")
    return (lo_of_tol - hi_of_intol) / rng


def snr_rank(rows):
    """rows: list of (name, head_dim, rms_k, tolerant). Returns
    (sorted_pairs, separable, margin_vs_hd).
      sorted_pairs : [(name, score), ...] ascending by score.
      separable    : does ONE threshold on s split tolerant (high) from intolerant (low)?
      margin_vs_hd : (relative s-gap) / (relative head_dim-gap) — >1 means the continuous SNR
                     score separates with a relatively wider band than the binary head_dim split;
                     EXPLORATORY only (retentions already seen, no gate)."""
    names = [r[0] for r in rows]
    hds = [r[1] for r in rows]
    scores = [snr_score(r[1], r[2]) for r in rows]
    tol = [bool(r[3]) for r in rows]
    s_gap = _rel_gap(scores, tol)
    hd_gap = _rel_gap(hds, tol)
    separable = bool(np.isfinite(s_gap) and s_gap > 0.0)
    margin_vs_hd = float(s_gap / hd_gap) if (np.isfinite(s_gap) and np.isfinite(hd_gap)
                                             and hd_gap != 0.0) else float("nan")
    order = sorted(range(len(names)), key=lambda i: scores[i])
    sorted_pairs = [(names[i], round(scores[i], 4)) for i in order]
    return sorted_pairs, separable, margin_vs_hd


def snr_fd_verdict(rho: float) -> str:
    """P3 gate (direction FIXED before measurement): supported iff rho<=SNR_FD_SUPPORT; refuted
    iff rho>=SNR_FD_REFUTE; else undetermined (weak or wrong-sign anticorrelation)."""
    if not np.isfinite(rho):
        return "undetermined"
    if rho <= SNR_FD_SUPPORT:
        return "supported"
    if rho >= SNR_FD_REFUTE:
        return "refuted"
    return "undetermined"


def predict_crossover(rms_k_pred: float, rms_k_cal: float,
                      cal_crossover: float = CAL_CROSSOVER,
                      pred_head_dim: int = PRED_HEAD_DIM,
                      cal_head_dim: int = CAL_HEAD_DIM) -> float:
    """P1: L_c(pred) = cal_crossover * sqrt(pred_hd/cal_hd) * (RMS_K_pred / RMS_K_cal). The global
    C cancels in the ratio; only sqrt(head_dim) and RMS_K enter."""
    return float(cal_crossover * math.sqrt(pred_head_dim / cal_head_dim)
                 * (rms_k_pred / rms_k_cal))


def p1_band(pred: float, measured_ci: float, cal_crossover: float = CAL_CROSSOVER,
            tol_scale: float = P1_TOL_SCALE) -> float:
    """Success band half-width (formula fixed in the step-19 addendum):
    tol_scale * (pred / cal_crossover) + measured_ci."""
    return float(tol_scale * (pred / cal_crossover) + measured_ci)


def snr_law_verdict(pred: float, measured_mu: float, band: float) -> str:
    """confirmed iff |pred - measured_mu| <= band; else refuted. A clean refutation of a derived
    law is a publishable falsification — reported at equal prominence (ALLOWED-to-FAIL)."""
    return "confirmed" if abs(pred - measured_mu) <= band else "refuted"


def stress_spread(retentions) -> float:
    """Spread = max - min retention among a set of models (used for hd=64 spread, P4)."""
    v = [r for r in retentions if r == r]        # drop nan
    if not v:
        return float("nan")
    return float(max(v) - min(v))


def relative_h1_consistent(tolerant_by_level, h1_set) -> bool:
    """P4: is the relative-noise tolerant set equal to the pure head_dim>=128 set (h1_set) at SOME
    level? tolerant_by_level: {level: set-or-list of tolerant model names}."""
    target = set(h1_set)
    return any(set(names) == target for names in tolerant_by_level.values())


# ---- step 21 relative-noise injector (phase10-local; NOT an edit to lossy_tier) ----

def relative_noise(page, level: float, seed: int):
    """Multiplicative (relative) analogue of quant_noise: K,V *= (1 + level*N(0,1)) elementwise.
    Under relative noise RMS_K cancels in the SNR ratio -> P4 tolerance follows pure sqrt(d).
    Phase10-local — the frozen lossy_tier.quant_noise (absolute noise) is untouched. Deterministic
    in (level, seed): same seed -> same multipliers; level=0 -> identity."""
    from aepk_paging.kv_page import KVPage
    if level < 0.0:
        raise ValueError("level must be non-negative")
    rng = np.random.default_rng(seed)
    K = page.K.astype(np.float32)
    V = page.V.astype(np.float32)
    Kn = (K * (np.float32(1.0) + np.float32(level) * rng.normal(size=K.shape).astype(np.float32)))
    Vn = (V * (np.float32(1.0) + np.float32(level) * rng.normal(size=V.shape).astype(np.float32)))
    return KVPage(page_id=page.page_id, layer=page.layer, token_range=page.token_range,
                  K=Kn, V=Vn, precision_tag=f"{page.precision_tag}+relative_noise",
                  attention_mass=page.attention_mass)


# ---- reuse of stored campaign data (single source of truth) -----------------

def load_grid_rows(path: str = GRID_ROWS_PATH):
    """The 7 INCLUDED grid_v2 models: {name: (head_dim, n_kv, n_cc, retention, tolerant)}. Rows
    are (name, family, head_dim, n_kv, n_cc, retention, tolerant, status)."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for name, fam, hd, nkv, ncc, ret, tol, status in rows:
        if status == "included":
            out[name] = (int(hd), int(nkv), int(ncc), float(ret), bool(tol))
    return out


def load_fd_v2_damage(path: str = FD_V2_REPORT_PATH):
    """Parse the stored per-layer retention_damage column (28 layers, level=1.0) from the
    'Full per-layer sweep' table of REPORT_phase10_fd_v2.md. Reused for P3 (no GPU rerun)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    i = text.find("## Full per-layer sweep")
    if i == -1:
        raise ValueError("fd_v2 report has no full per-layer sweep (gate did not pass)")
    seg = text[i:]
    dmg = {}
    for m in re.finditer(r"^\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*([-\d.]+)\s*\|", seg, re.M):
        dmg[int(m.group(1))] = float(m.group(2))
    layers = sorted(dmg)
    return layers, [dmg[l] for l in layers]


# ============================================================================
# GPU measurement (deterministic clean prefill -> byte-identical)
# ============================================================================

def _prefix_pages(model, tok, prompt, device):
    """Clean prefill -> (ids, past_key_values). Deterministic (no sampling, no RNG)."""
    import torch
    ids = tok(prompt, return_tensors="pt").to(device).input_ids
    with torch.no_grad():
        out = model(ids[:, :-1], use_cache=True)
    return ids, out.past_key_values


def _clean_correct_pages(model, tok, device, dtype, probes):
    """Return list of clean page-lists for the probes the model answers correctly on the CLEAN
    cache (same clean-correct conditioning as the grid: normalize_answer + normalized_match)."""
    import torch
    from aepk_paging.harness.phase9_cw import _decode_under_cache
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    from aepk_paging.harness.phase10_grid import normalize_answer
    subset = []
    for pr in probes:
        ids, pkv = _prefix_pages(model, tok, pr["prompt"], device)
        pg = dynamiccache_to_pages(pkv)
        _inject_pages(pkv, pg, dtype, device)
        t, _ = _decode_under_cache(model, tok, ids, pkv, device)
        if normalized_match(normalize_answer(t), pr["expected"], pr.get("alternatives")):
            subset.append(pg)
    return subset


def measure_model_rms_k(model, tok, device, dtype, probes):
    """(n_cc, rms_k): RMS over ALL clean K elements of ALL layer-pages across the clean-correct
    subset. Deterministic -> byte-identical across repeats."""
    subset = _clean_correct_pages(model, tok, device, dtype, probes)
    ss = 0.0
    cnt = 0
    for pg in subset:
        for page in pg:
            K = np.asarray(page.K, dtype=np.float64)
            ss += float(np.sum(K * K))
            cnt += K.size
    rms_k = float(np.sqrt(ss / cnt)) if cnt else float("nan")
    return len(subset), rms_k


def measure_layer_rms(model, tok, device, dtype, probes):
    """(layers, key_rms_per_layer, val_rms_per_layer): per-LAYER mean key/value RMS averaged over
    the clean-correct subset (for P3, qwen1.5b). Deterministic."""
    subset = _clean_correct_pages(model, tok, device, dtype, probes)
    n = len(subset)
    n_layers = len(subset[0]) if subset else 0
    key_rms = np.zeros(n_layers, dtype=np.float64)
    val_rms = np.zeros(n_layers, dtype=np.float64)
    for pg in subset:
        for L, page in enumerate(pg):
            key_rms[L] += page_key_rms(page)
            val_rms[L] += page_value_rms(page)
    if n:
        key_rms /= n
        val_rms /= n
    return list(range(n_layers)), key_rms.tolist(), val_rms.tolist()


def run_relative_grid_model(model, tok, device, dtype, *, probes, levels, seeds=(0, 1, 2),
                            floor: float = 0.70):
    """Step 21 (P4): per-model retention under RELATIVE noise. Same clean-correct conditioning and
    seed derivation (sd*1000+layer) as the absolute-noise grid, but the injector is relative_noise
    (multiplicative). Returns (n_cc, {level: retention}). retention on the clean-correct subset."""
    import torch
    from aepk_paging.harness.phase9_cw import _decode_under_cache
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    from aepk_paging.harness.phase10_grid import normalize_answer

    def prefix(prompt):
        ids = tok(prompt, return_tensors="pt").to(device).input_ids
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=True)
        return ids, out.past_key_values

    def correct(pr, level=None, sd=None):
        ids, pkv = prefix(pr["prompt"])
        pg = dynamiccache_to_pages(pkv)
        if level is None:
            _inject_pages(pkv, pg, dtype, device)
        else:
            noisy = [relative_noise(p, level, sd * 1000 + p.layer) for p in pg]
            _inject_pages(pkv, noisy, dtype, device)
        t, _ = _decode_under_cache(model, tok, ids, pkv, device)
        return int(normalized_match(normalize_answer(t), pr["expected"], pr.get("alternatives")))

    clean_correct = [pr for pr in probes if correct(pr)]
    n_cc = len(clean_correct)
    retention = {}
    for lv in levels:
        if n_cc == 0:
            retention[lv] = float("nan")
            continue
        per_seed = [sum(correct(pr, level=lv, sd=sd) for pr in clean_correct) / n_cc
                    for sd in seeds]
        retention[lv] = float(np.mean(per_seed))
    return n_cc, retention


# ============================================================================
# report (regenerated from the persisted campaign state each step)
# ============================================================================

def write_snr_report(state, path="results/REPORT_phase10_snr.md"):
    """Regenerate REPORT_phase10_snr.md from the persisted state dict. Sections present as their
    data lands: rms/rank/fd (step 19), law (step 20), stress (step 21). All verdict lines are
    runtime f-strings."""
    grid = load_grid_rows()
    rms_rows = state["rms_rows"]                     # [(name, head_dim, n_cc, rms_k, score, tol)]
    rank_pairs = state["rank"]["sorted_pairs"]
    separable = state["rank"]["separable"]
    margin = state["rank"]["margin_vs_hd"]
    fd = state["fd"]

    L = [
        "# REPORT_phase10_snr.md — Phase 10 SNR CAMPAIGN (mechanism day, steps 19-21)",
        "",
        "Derived law (see PREREG_phase10_snr.md for the verbatim theory): "
        "L_c = C * sqrt(head_dim) * RMS_K. SNR susceptibility score s = sqrt(head_dim)*RMS_K.",
        "",
        "## P2 — SNR score vs the 7 included grid_v2 models (EXPLORATORY; retentions already seen)",
        "",
        "| model | head_dim | N_cc | RMS_K | SNR score s | tolerant |",
        "|-------|----------|------|-------|-------------|----------|",
    ]
    for name, hd, ncc, rms_k, score, tol in rms_rows:
        L.append(f"| {name} | {hd} | {ncc} | {rms_k:.5f} | {score:.4f} | {tol} |")
    L += [
        "",
        f"Scores sorted (ascending): {rank_pairs}. `separable` asks whether ONE threshold on s "
        "splits the 3 tolerant (high) from the 4 intolerant (low). `margin_vs_hd` = relative "
        "s-separation band / relative head_dim-separation band (>1 => the continuous SNR score "
        "separates with a wider relative margin than the binary head_dim split). Exploratory — "
        "no gate.",
        "",
        f"SNR_RANK: scores={rank_pairs} separable={separable} margin_vs_hd={margin:.4f}",
        "",
        "## P3 — per-layer damage vs per-layer MEAN key RMS (qwen1.5b; damage reused from fd_v2)",
        "",
        "Magnitude (not variance) is the predicted susceptibility variable: higher clean key RMS "
        "=> MORE per-layer damage is the FD sign, but the theory predicts the ANTICORRELATION "
        "(a layer whose keys are large has a large signal logit gap -> the fixed absolute noise "
        "is relatively weaker -> LESS damage). Gate fixed pre-measurement: supported iff "
        f"rho<={SNR_FD_SUPPORT}; refuted iff rho>={SNR_FD_REFUTE}. Per-layer damage reused "
        "verbatim from REPORT_phase10_fd_v2.md (level=1.0 sweep, n_cc=50).",
        "",
        "| layer | mean key RMS (clean) | retention_damage (fd_v2) |",
        "|-------|----------------------|--------------------------|",
    ]
    for ly, kr, dm in zip(fd["layers"], fd["key_rms"], fd["damage"]):
        L.append(f"| {ly} | {kr:.5f} | {dm:.4f} |")
    L += [
        "",
        f"Spearman(mean key RMS, damage) = {fd['rho_key']:.4f} (primary, P3). "
        f"Spearman(mean value RMS, damage) = {fd['rho_val']:.4f} (secondary — the -0.4888 "
        f"value-norm thread; reported as-is, no gate). n_layers={len(fd['layers'])}.",
        "",
        f"SNR_FD: spearman={fd['rho_key']:.4f} n_layers={len(fd['layers'])} verdict={fd['verdict']}",
    ]

    # ---- step 20: SNR_LAW (out-of-sample crossover) --------------------------
    if state.get("law") is not None:
        law = state["law"]
        L += [
            "",
            "## P1 — out-of-sample crossover prediction (qwen0.5b), step-20 GATE",
            "",
            f"Predicted from the step-19 LOCKED addendum: qwen0.5b crossover = {CAL_CROSSOVER} * "
            f"sqrt({PRED_HEAD_DIM}/{CAL_HEAD_DIM}) * (RMS_K_q0.5b/RMS_K_q1.5b) = "
            f"{law['predicted']:.4f} (prediction written down BEFORE the sweep launched). Success "
            f"band half-width = {P1_TOL_SCALE}*(pred/{CAL_CROSSOVER}) + measured_CI = "
            f"{law['band']:.4f}. Same pool as calibration (LARGE_PROBES n=105), same "
            "levels/seeds/FLOOR/conditioning as PREREG v3. A clean refutation is a publishable "
            "falsification, reported at equal prominence (ALLOWED-to-FAIL).",
            "",
            "| seed | " + " | ".join(f"L={lv}" for lv in law["levels"]) + " | crossover |",
            "|------|" + "|".join(["------"] * len(law["levels"])) + "|-----------|",
        ]
        for sd, xo in zip(law["seeds"], law["crossovers"]):
            rets = law["grid"][str(sd)] if str(sd) in law["grid"] else law["grid"][sd]
            L.append(f"| {sd} | " + " | ".join(f"{r:.3f}" for r in rets) + f" | {xo:.3f} |")
        L += [
            "",
            f"Predicted={law['predicted']:.4f}; measured crossover mu={law['measured_mu']:.4f} "
            f"+/-{law['measured_ci']:.4f} (n={len(law['seeds'])}); |pred-measured|="
            f"{abs(law['predicted']-law['measured_mu']):.4f} vs band {law['band']:.4f}.",
            "",
            f"SNR_LAW: predicted={law['predicted']:.4f} measured={law['measured_mu']:.4f}"
            f"±{law['measured_ci']:.4f} verdict={law['verdict']}",
        ]

    # ---- step 21: STRESS_INV (relative-noise grid) ---------------------------
    if state.get("stress") is not None:
        st = state["stress"]
        L += [
            "",
            "## P4 — stress-family invariance: relative (multiplicative) noise grid",
            "",
            "Under relative noise K,V *= (1+level*N(0,1)) the RMS_K factor cancels in the SNR "
            "ratio, so tolerance should follow the PURE sqrt(head_dim) split (H1 set) and the "
            "RMS_K-driven gradations AMONG the hd=64 models should COMPRESS vs the absolute-noise "
            "grid. h1_consistent = does the relative-noise tolerant set equal the head_dim>=128 "
            "set at some level. spread = max-min retention among hd=64 models; P4 predicts "
            "rel < abs. ALLOWED to fail.",
            "",
            "| model | head_dim | " + " | ".join(f"ret@{lv}" for lv in st["levels"])
            + " | tolerant(any) |",
            "|-------|----------|" + "|".join(["------"] * len(st["levels"])) + "|------------|",
        ]
        for name in st["order"]:
            hd = grid[name][0]
            rets = st["retention"][name]
            tolany = st["tolerant"][name]
            L.append(f"| {name} | {hd} | "
                     + " | ".join(f"{rets[str(lv)] if str(lv) in rets else rets[lv]:.3f}"
                                   for lv in st["levels"])
                     + f" | {tolany} |")
        L += [
            "",
            f"hd=64 spread under relative noise = {st['hd64_spread_rel']:.4f}; under absolute "
            f"noise (grid_v2 retentions) = {st['hd64_spread_abs']:.4f}. P4 predicts rel < abs "
            f"({'holds' if st['hd64_spread_rel'] < st['hd64_spread_abs'] else 'does NOT hold'}).",
            "",
            f"STRESS_INV: family=relative levels={st['levels']} "
            f"h1_consistent={st['h1_consistent']} hd64_spread_rel={st['hd64_spread_rel']:.4f} "
            f"hd64_spread_abs={st['hd64_spread_abs']:.4f}",
        ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path
