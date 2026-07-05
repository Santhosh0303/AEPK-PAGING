"""Phase 10 step (6) / 9.4 — statistics on the final config.

>=5 seeds/cell on Qwen2.5-1.5B (the step-5 tolerant config). Sweeps quant_noise level, computes
per-seed retention curves, finds the FLOOR-crossover level per seed by linear interpolation, and
reports crossover mu +/- 95% CI across seeds. No cherry-pick — all seeds enter mu.

Honesty spine S9: zero edits to Phase 2-5 source. Reuses quant_noise, dynamiccache_to_pages,
_inject_pages, _decode_under_cache, normalized_match, CW_PROBES. Deterministic. Runtime verdict.
"""

from __future__ import annotations

import numpy as np

FLOOR = 0.70
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SEEDS = (0, 1, 2, 3, 4)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# t(0.975, df) for small df (two-sided 95% CI)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def crossover_level(levels, retentions, floor: float = FLOOR) -> float:
    """Level where retention crosses `floor` going down, linear-interpolated. Censoring:
    left-censored (already below at level[0]) -> level[0]; right-censored (never below) ->
    level[-1]."""
    lv = list(levels); rt = list(retentions)
    if rt[0] < floor:
        return float(lv[0])                       # left-censored
    for i in range(1, len(lv)):
        if rt[i] < floor:                          # bracket [i-1, i]
            hi, lo = rt[i - 1], rt[i]
            frac = (hi - floor) / (hi - lo) if hi != lo else 0.0
            return float(lv[i - 1] + frac * (lv[i] - lv[i - 1]))
    return float(lv[-1])                            # right-censored


def ci95(values) -> tuple[float, float]:
    """(mean, 95% CI half-width via t). ci=0 for n<=1."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    mu = float(v.mean())
    if n <= 1:
        return mu, 0.0
    sd = float(v.std(ddof=1))
    t = _T95.get(n - 1, 1.96)
    return mu, float(t * sd / np.sqrt(n))


def run_stats(model, tok, device, dtype, *, probes=None, levels=LEVELS, seeds=SEEDS):
    """Return (clean_acc, n_cc, per_seed_crossovers, retention_grid). retention_grid[seed][level].

    PREREG v3 (grid-consistent methodology): clean-correct conditioning — the noise sweep runs
    only on the probes the model answers correctly on the CLEAN cache (clean_acc=1.0 there by
    construction), so retention = corrupt_acc on that subset and can never exceed 1. Seed
    derivation sd*1000 + p.layer — non-overlapping across seeds (same fix as grid/fd)."""
    import torch
    from aepk_paging.harness.phase9_cw import _decode_under_cache, CW_PROBES
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    from aepk_paging.lossy_tier import quant_noise
    probes = probes or CW_PROBES

    def prefix(prompt):
        enc = tok(prompt, return_tensors="pt").to(device)
        ids = enc.input_ids
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=True)
        return ids, out.past_key_values

    clean_correct = []
    for pr in probes:
        ids, pkv = prefix(pr["prompt"])
        pg = dynamiccache_to_pages(pkv); _inject_pages(pkv, pg, dtype, device)
        t, _ = _decode_under_cache(model, tok, ids, pkv, device)
        if normalized_match(t, pr["expected"], pr.get("alternatives")):
            clean_correct.append(pr)
    clean_acc = len(clean_correct) / len(probes)
    n_cc = len(clean_correct)

    grid = {}          # seed -> list of retention per level (retention <= 1 by construction)
    crossovers = []
    for sd in seeds:
        rets = []
        for lv in levels:
            ok = 0
            for pr in clean_correct:
                ids, pkv = prefix(pr["prompt"])
                pg = dynamiccache_to_pages(pkv)
                noisy = [quant_noise(p, lv, sd * 1000 + p.layer)[0] for p in pg]
                _inject_pages(pkv, noisy, dtype, device)
                t, _ = _decode_under_cache(model, tok, ids, pkv, device)
                ok += int(normalized_match(t, pr["expected"], pr.get("alternatives")))
            rets.append(ok / n_cc if n_cc > 0 else float("nan"))
        grid[sd] = rets
        crossovers.append(crossover_level(levels, rets))
    return clean_acc, n_cc, crossovers, grid


def write_stats_report(clean_acc, crossovers, grid, seeds=SEEDS, levels=LEVELS,
                       path="results/REPORT_phase10_stats.md", n_cc=None):
    import os
    mu, ci = ci95(crossovers)
    n = len(crossovers)
    ncc_note = (f" Clean-correct conditioning (PREREG v3): N_cc={n_cc} probes; sweep runs on "
                "that subset only (clean_acc=1.0 there), retention=corrupt_acc on the subset "
                "(<=1 by construction); seed derivation sd*1000+layer."
                if n_cc is not None else "")
    L = [
        "# REPORT_phase10_stats.md — Phase 10 step (6) / 9.4 statistics (final config)",
        "",
        f"Model: {MODEL_ID} fp16 (CUDA), the step-5 tolerant config. clean_acc={clean_acc:.3f}. "
        f"Stress: quant_noise level in {list(levels)} on every KV page, {n} seeds. "
        f"crossover=level where retention crosses FLOOR={FLOOR} "
        "(linear-interpolated; censoring reported). No cherry-pick — all seeds enter mu."
        + ncc_note,
        "",
        "| seed | " + " | ".join(f"L={lv}" for lv in levels) + " | crossover |",
        "|------|" + "|".join(["------"] * len(levels)) + "|-----------|",
    ]
    for sd, xo in zip(seeds, crossovers):
        rets = grid[sd]
        L.append(f"| {sd} | " + " | ".join(f"{r:.3f}" for r in rets) + f" | {xo:.3f} |")
    L += [
        "",
        "## Interpretation",
        f"Per-seed crossover levels: {[round(x,3) for x in crossovers]}. Mean crossover "
        f"mu={mu:.3f}, 95% CI half-width={ci:.3f} (t-based, n={n}). A tight CI means the "
        "compression-tolerance crossover of the final config is a stable statistic, not a "
        "single-seed artifact. Right-censored seeds (retention never below FLOOR through the "
        f"top level {levels[-1]}) report crossover={levels[-1]} and are visible in the table "
        "(retention row stays >= FLOOR).",
        "",
        f"STATS: crossover={mu:.3f}±{ci:.3f} seeds={n}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return mu, ci, n


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from aepk_paging.harness.eval_set_large import LARGE_PROBES   # >=100 probes (step 4)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
    model.eval()
    print(f"probe set: n={len(LARGE_PROBES)} (granularity {1/len(LARGE_PROBES):.4f})")
    ca, ncc, xos, grid = run_stats(model, tok, "cuda", torch.float16, probes=LARGE_PROBES)
    mu, ci, n = write_stats_report(ca, xos, grid, n_cc=ncc)
    print(f"clean_acc={ca:.3f} N_cc={ncc}")
    print(f"per-seed crossovers: {xos}")
    print(f"STATS: crossover={mu:.3f}±{ci:.3f} seeds={n}")
