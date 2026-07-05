"""Phase 10 step (5) — factorial model grid: the redundancy-floor law test + transition sharpness.

Crosses head_dim x KV-width x family over 8 VRAM-verified models (<=3.4GB fp16). Per model we
condition on the model's CLEAN-CORRECT subset (probes it answers right on the clean cache), then
measure retention under quant_noise(LEVEL) on that subset over seeds. tolerant <=> retention >=
FLOOR. Two predictors (H1 head_dim>=128, H2 n_kv*head_dim>=256) are compared to the observed
tolerant-set (FLOOR_LAW_GRID verdict). Retention-vs-redundancy is then fit with a logistic
(sharp) and a linear (gradual) curve, compared by AIC (TRANSITION verdict).

Pre-registered in PREREG_phase10_grid.md. Honesty spine S9: zero edits to Phase 2-5 source.
Reuses quant_noise, dynamiccache_to_pages, _inject_pages, _decode_under_cache, normalized_match,
LARGE_PROBES, predict_head_dim/predict_kv_width. Deterministic. Runtime verdict f-strings.
"""

from __future__ import annotations

import math

import numpy as np

from aepk_paging.harness.phase10_floor_law import predict_head_dim, predict_kv_width

LEVEL = 0.20
FLOOR = 0.70
SEEDS = (0, 1, 2)
MIN_CLEAN_CORRECT = 30          # inclusion: need >=30 clean-correct probes (PREREG grid)
AIC_MARGIN = 2.0                # decisive-model margin for the transition verdict

# (name, model_id, family)  — head_dim/n_kv are READ from config at load, not trusted from here.
GRID = [
    ("qwen0.5b",     "Qwen/Qwen2.5-0.5B-Instruct",          "qwen2"),
    ("qwen1.5b",     "Qwen/Qwen2.5-1.5B-Instruct",          "qwen2"),
    ("tinyllama",    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   "llama"),
    ("pythia-160m",  "EleutherAI/pythia-160m",               "gpt_neox"),
    ("pythia-410m",  "EleutherAI/pythia-410m",               "gpt_neox"),
    ("pythia-1b",    "EleutherAI/pythia-1b",                 "gpt_neox"),
    ("pythia-1.4b",  "EleutherAI/pythia-1.4b",               "gpt_neox"),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct",  "llama"),
]


def arch(config) -> tuple[int, int]:
    """(head_dim, n_kv) from a HF config, robust across qwen2/llama/gpt_neox."""
    hd = getattr(config, "head_dim", None)
    if hd is None:
        hd = config.hidden_size // config.num_attention_heads
    nkv = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    return int(hd), int(nkv)


def normalize_answer(text: str) -> str:
    """PREREG grid: strip the raw-prompt chat-continuation artifact by truncating at the first
    'Human:'/'Assistant:' marker before matching. Nothing else altered."""
    for marker in ("Human:", "Assistant:"):
        i = text.find(marker)
        if i != -1:
            text = text[:i]
    return text


# ---- deterministic verdict math (CPU-testable, no model) --------------------

def floor_law_grid_verdict(rows) -> tuple[list, list, list, str]:
    """rows: included (name, head_dim, n_kv, retention, tolerant). Returns
    (predicted_H1, predicted_H2, observed, verdict). verdict in
    {H1, H2, neither, indistinguishable} — indistinguishable when both laws
    predict the same set AND the observation matches it (no discriminating
    model survived inclusion, so H1 vs H2 cannot be separated)."""
    pH1 = sorted(n for n, hd, _, _, _ in rows if predict_head_dim(hd))
    pH2 = sorted(n for n, hd, nkv, _, _ in rows if predict_kv_width(nkv, hd))
    obs = sorted(n for n, _, _, _, tol in rows if tol)
    if obs == pH1 == pH2:
        v = "indistinguishable"
    elif obs == pH1 and obs != pH2:
        v = "H1"
    elif obs == pH2 and obs != pH1:
        v = "H2"
    else:
        v = "neither"
    return pH1, pH2, obs, v


def _logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k * (x - x0)))


def _aic(n, sse, k_params) -> float:
    """AIC for least-squares: 2k + n ln(SSE/n). Guards SSE=0."""
    sse = max(float(sse), 1e-12)
    return 2 * k_params + n * math.log(sse / n)


def transition_verdict(xs, ys, margin: float = AIC_MARGIN) -> tuple[str, dict]:
    """Fit logistic (sharp, 3 params) and linear (gradual, 2 params) to y(x); compare by AIC.
    Returns (form, detail). form in {sharp, gradual, undetermined}."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = x.size
    detail = {"n": int(n)}
    if n < 5:
        detail["reason"] = "fewer than 5 included models"
        return "undetermined", detail
    # linear
    a, b = np.polyfit(x, y, 1)
    sse_lin = float(np.sum((y - (a * x + b)) ** 2))
    aic_lin = _aic(n, sse_lin, 2)
    # logistic
    try:
        from scipy.optimize import curve_fit
        p0 = [max(y.max(), 1e-3), 1.0 / (x.std() or 1.0), float(np.median(x))]
        popt, _ = curve_fit(_logistic, x, y, p0=p0, maxfev=20000)
        sse_log = float(np.sum((y - _logistic(x, *popt)) ** 2))
        aic_log = _aic(n, sse_log, 3)
    except Exception as e:                        # noqa: BLE001 — any fit failure -> undetermined
        detail["reason"] = f"logistic fit failed: {type(e).__name__}"
        detail.update(aic_linear=round(aic_lin, 3))
        return "undetermined", detail
    detail.update(aic_linear=round(aic_lin, 3), aic_logistic=round(aic_log, 3),
                  sse_linear=round(sse_lin, 5), sse_logistic=round(sse_log, 5))
    if aic_log + margin < aic_lin:
        return "sharp", detail
    if aic_lin + margin < aic_log:
        return "gradual", detail
    detail["reason"] = "AIC difference within margin"
    return "undetermined", detail


# ---- GPU per-model measurement ----------------------------------------------

def run_grid_model(model, tok, device, dtype, *, probes, level=LEVEL, seeds=SEEDS):
    """Return (n_clean_correct, retention, tolerant). Retention is measured on the model's
    clean-correct subset (clean_acc=1.0 there by construction), seed derivation sd*1000+p.layer."""
    import torch
    from aepk_paging.harness.phase9_cw import _decode_under_cache
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    from aepk_paging.lossy_tier import quant_noise

    def prefix(prompt):
        ids = tok(prompt, return_tensors="pt").to(device).input_ids
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=True)
        return ids, out.past_key_values

    def correct(pr, noisy_seed=None):
        ids, pkv = prefix(pr["prompt"])
        pg = dynamiccache_to_pages(pkv)
        if noisy_seed is None:
            _inject_pages(pkv, pg, dtype, device)
        else:
            noisy = [quant_noise(p, level, noisy_seed * 1000 + p.layer)[0] for p in pg]
            _inject_pages(pkv, noisy, dtype, device)
        t, _ = _decode_under_cache(model, tok, ids, pkv, device)
        return int(normalized_match(normalize_answer(t), pr["expected"], pr.get("alternatives")))

    clean_correct = [pr for pr in probes if correct(pr)]
    n_cc = len(clean_correct)
    if n_cc == 0:
        return 0, float("nan"), False
    per_seed = []
    for sd in seeds:
        ok = sum(correct(pr, noisy_seed=sd) for pr in clean_correct)
        per_seed.append(ok / n_cc)
    ret = float(np.mean(per_seed))
    return n_cc, ret, bool(ret >= FLOOR)


def write_grid_report(rows, path="results/REPORT_phase10_grid.md"):
    """rows: (name, family, head_dim, n_kv, n_cc, retention, tolerant, status). status is
    'included' or an exclusion reason string."""
    import os
    incl = [(n, hd, nkv, ret, tol) for n, fam, hd, nkv, ncc, ret, tol, st in rows
            if st == "included"]
    pH1, pH2, obs, verdict = floor_law_grid_verdict(incl)

    # transition: retention vs redundancy (KV-width primary; head_dim secondary)
    xs_kv = [hd * nkv for n, hd, nkv, ret, tol in incl]
    xs_hd = [hd for n, hd, nkv, ret, tol in incl]
    ys = [ret for n, hd, nkv, ret, tol in incl]
    form_kv, det_kv = transition_verdict(xs_kv, ys)
    form_hd, det_hd = transition_verdict(xs_hd, ys)

    L = [
        "# REPORT_phase10_grid.md — Phase 10 step (5) factorial grid: floor-law + transition",
        "",
        f"Stress: quant_noise LEVEL={LEVEL} on every KV page, SEEDS={list(SEEDS)} "
        f"(seed derivation sd*1000+layer). Inclusion: clean-correct conditioning, a model enters "
        f"iff N_clean_correct >= {MIN_CLEAN_CORRECT}; retention measured on that subset "
        f"(clean_acc=1.0 there). tolerant <=> retention >= FLOOR={FLOOR}. H1: head_dim>=128; "
        "H2: n_kv*head_dim>=256. See PREREG_phase10_grid.md.",
        "",
        "| model | family | head_dim | KV-width | N_clean_correct | retention | tolerant | "
        "H1_pred | H2_pred | status |",
        "|-------|--------|----------|----------|-----------------|-----------|----------|"
        "---------|---------|--------|",
    ]
    for n, fam, hd, nkv, ncc, ret, tol, st in rows:
        rets = f"{ret:.3f}" if ret == ret else "nan"       # nan-safe
        L.append(f"| {n} | {fam} | {hd} | {hd*nkv} | {ncc} | {rets} | {tol} | "
                 f"{predict_head_dim(hd)} | {predict_kv_width(nkv, hd)} | {st} |")
    excluded = [(n, st) for n, fam, hd, nkv, ncc, ret, tol, st in rows if st != "included"]

    # verdict=indistinguishable: name the failure mode + exploratory pattern (runtime-derived).
    indist_lines: list[str] = []
    if verdict == "indistinguishable":
        allr = [(n, hd, nkv, tol) for n, fam, hd, nkv, ncc, ret, tol, st in rows]
        h1_ok = sum(tol == predict_head_dim(hd) for n, hd, nkv, tol in allr)
        h2_ok = sum(tol == predict_kv_width(nkv, hd) for n, hd, nkv, tol in allr)
        h2_contra = sorted(f"{n}(width-{hd*nkv})" for n, hd, nkv, tol in allr
                           if tol != predict_kv_width(nkv, hd))
        indist_lines = [
            "",
            f"verdict=indistinguishable: every model where H1 and H2 disagree was excluded by "
            f"the N_clean_correct >= {MIN_CLEAN_CORRECT} gate, so both laws predict the SAME "
            f"tolerant-set on the included models and the data cannot separate them there. "
            f"EXPLORATORY ONLY (all grid rows, inclusion gate ignored, under-powered — not a "
            f"pre-registered comparison): H1 consistent on {h1_ok}/{len(allr)} rows; H2 "
            f"consistent on {h2_ok}/{len(allr)} rows, contradicted by {h2_contra}.",
        ]

    L += [
        "",
        "## Interpretation",
        f"Included models (N_clean_correct >= {MIN_CLEAN_CORRECT}): {sorted(x[0] for x in incl)}. "
        f"Excluded: {excluded} (reason per entry). The FLOOR_LAW_GRID verdict compares the "
        "observed tolerant-set against H1 (head_dim) and H2 (KV-width) predictions over the "
        "included models; reported as-is even if neither law holds. The TRANSITION verdict fits "
        "retention vs redundancy with a logistic (sharp threshold) and a linear (gradual) curve "
        "and compares AIC — sharp = phase-transition-style critical threshold, gradual = smooth "
        "law; both are laws, reported as-is.",
        "",
        f"transition-by-KV-width detail: {det_kv}",
        f"transition-by-head_dim detail: {det_hd}",
        *indist_lines,
        "",
        f"FLOOR_LAW_GRID: predicted_H1={pH1} predicted_H2={pH2} observed={obs} verdict={verdict}",
        f"TRANSITION: form={form_kv} (by KV-width; by head_dim={form_hd})",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return pH1, pH2, obs, verdict, form_kv, form_hd


if __name__ == "__main__":
    import torch, gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from aepk_paging.harness.eval_set_large import LARGE_PROBES

    rows = []
    for name, mid, fam in GRID:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.float16, device_map="cuda").eval()
            hd, nkv = arch(model.config)
            n_cc, ret, tol = run_grid_model(model, tok, "cuda", torch.float16, probes=LARGE_PROBES)
            status = "included" if n_cc >= MIN_CLEAN_CORRECT else f"excluded(N_cc={n_cc}<{MIN_CLEAN_CORRECT})"
            rows.append((name, fam, hd, nkv, n_cc, ret, tol, status))
            print(f"  {name}: hd={hd} kvw={hd*nkv} N_cc={n_cc} retention={ret:.3f} "
                  f"tolerant={tol} [{status}]")
            del model; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:                    # noqa: BLE001 — model-level failure is recorded
            rows.append((name, fam, -1, -1, 0, float("nan"), False, f"excluded(error:{type(e).__name__})"))
            print(f"  {name}: EXCLUDED error {type(e).__name__}: {str(e)[:120]}")
            gc.collect(); torch.cuda.empty_cache()
    pH1, pH2, obs, verdict, fkv, fhd = write_grid_report(rows)
    print(f"FLOOR_LAW_GRID: predicted_H1={pH1} predicted_H2={pH2} observed={obs} verdict={verdict}")
    print(f"TRANSITION: form={fkv} (by KV-width; by head_dim={fhd})")
