"""Phase 10 step (7) — FENCED MOONSHOT: fluctuation-dissipation (FD) analogue.

Physics's deepest gift: equilibrium fluctuations predict response to perturbation. Test the
analogue on KV cache. From CLEAN KV pages only, compute a per-layer fluctuation statistic
(variance of per-token key norms; secondary: value norms). Pre-register the DIRECTIONAL
prediction (higher clean fluctuation => more retention damage when that layer alone is
corrupted). Then corrupt ONE layer-page at a time (quant_noise level=0.2, seeds=[0,1,2],
non-overlapping seed derivation sd*1000+layer), measure per-layer retention drop, and compare
predicted vs observed by Spearman rho.

Pre-registered in PREREG_phase10_fd.md. Honesty spine S9: zero edits to Phase 2-5 source.
Reuses quant_noise, dynamiccache_to_pages, _inject_pages, _decode_under_cache, normalized_match,
LARGE_PROBES, and phase10_grid.normalize_answer. Deterministic. Runtime verdict f-string.
Refuted (null) is a real result — reported, not reframed.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LEVEL = 0.20
SEEDS = (0, 1, 2)
RHO_SUPPORT = 0.60          # rho >= this => FD analogue SUPPORTED (predicted +direction)
RHO_NULL = 0.30             # |rho| < this => null: no FD analogue (REFUTED)
# PREREG v2 positive control: ascending single-layer levels tried on 3 pre-named layers
# (first/mid/last); smallest level with max damage >= CONTROL_DAMAGE_MIN wins; none -> stop.
CONTROL_LEVELS = (0.5, 1.0, 2.0)
CONTROL_DAMAGE_MIN = 0.15


def key_norm_variance(page) -> float:
    """Variance of per-token key norms for one layer-page. K has shape (T, H, D); the per-token
    norm is ||K[t]|| over (H, D). Higher variance = more heterogeneous token importance =
    (predicted) more fragile under corruption."""
    K = np.asarray(page.K, dtype=np.float64)
    norms = np.linalg.norm(K.reshape(K.shape[0], -1), axis=1)
    return float(np.var(norms))


def value_norm_variance(page) -> float:
    """Secondary statistic: variance of per-token value norms."""
    V = np.asarray(page.V, dtype=np.float64)
    norms = np.linalg.norm(V.reshape(V.shape[0], -1), axis=1)
    return float(np.var(norms))


def spearman_rho(a, b) -> float:
    """Spearman rank correlation between two equal-length sequences: rank-then-Pearson with
    average ranks for ties. One deterministic numpy implementation — no scipy dependency."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = _avg_rank(a)
    rb = _avg_rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _avg_rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def fd_verdict(rho: float) -> str:
    """supported iff rho>=RHO_SUPPORT; refuted (null: no analogue) iff |rho|<RHO_NULL; else
    undetermined (weak, or wrong-sign relation — the direction was fixed in the prereg)."""
    if not np.isfinite(rho):
        return "undetermined"
    if rho >= RHO_SUPPORT:
        return "supported"
    if abs(rho) < RHO_NULL:
        return "refuted"
    return "undetermined"


def run_fd(model, tok, device, dtype, *, probes, level=LEVEL, seeds=SEEDS):
    """Return (layers, kvar, vvar, damage, n_clean_correct). Per-layer clean fluctuation +
    per-layer retention damage from corrupting that layer alone, on the clean-correct subset."""
    import torch
    from aepk_paging.harness.phase9_cw import _decode_under_cache
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    from aepk_paging.lossy_tier import quant_noise
    from aepk_paging.harness.phase10_grid import normalize_answer

    def prefix(prompt):
        ids = tok(prompt, return_tensors="pt").to(device).input_ids
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=True)
        return ids, out.past_key_values

    def clean_correct(pr):
        ids, pkv = prefix(pr["prompt"])
        pg = dynamiccache_to_pages(pkv)
        _inject_pages(pkv, pg, dtype, device)
        t, _ = _decode_under_cache(model, tok, ids, pkv, device)
        ok = normalized_match(normalize_answer(t), pr["expected"], pr.get("alternatives"))
        return ok, pg

    subset = []                                    # clean-correct probes + their clean pages
    for pr in probes:
        ok, pg = clean_correct(pr)
        if ok:
            subset.append((pr, pg))
    n_cc = len(subset)
    n_layers = len(subset[0][1]) if subset else 0

    # clean fluctuation per layer (mean over clean-correct probes)
    kvar = np.zeros(n_layers); vvar = np.zeros(n_layers)
    for _, pg in subset:
        for L, page in enumerate(pg):
            kvar[L] += key_norm_variance(page)
            vvar[L] += value_norm_variance(page)
    if n_cc:
        kvar /= n_cc; vvar /= n_cc

    # observed damage: corrupt layer L alone, over seeds, on the clean-correct subset
    damage = np.zeros(n_layers)
    for L in range(n_layers):
        per_seed = []
        for sd in seeds:
            ok = 0
            for pr, _ in subset:
                ids, pkv = prefix(pr["prompt"])
                pg = dynamiccache_to_pages(pkv)
                noisy = [quant_noise(p, level, sd * 1000 + p.layer)[0] if p.layer == L else p
                         for p in pg]
                _inject_pages(pkv, noisy, dtype, device)
                t, _ = _decode_under_cache(model, tok, ids, pkv, device)
                ok += int(normalized_match(normalize_answer(t), pr["expected"],
                                           pr.get("alternatives")))
            per_seed.append(ok / n_cc if n_cc else float("nan"))
        damage[L] = 1.0 - float(np.mean(per_seed))     # retention drop
    return list(range(n_layers)), kvar, vvar, damage, n_cc


def control_layer_ids(n_layers: int) -> list[int]:
    """PREREG v2: the 3 pre-named control layers — first, mid, last."""
    return [0, n_layers // 2, n_layers - 1]


def pick_control_level(control_rows, threshold: float = CONTROL_DAMAGE_MIN):
    """control_rows: (level, layer, damage). Smallest level whose max damage >= threshold,
    else None (no-response-regime)."""
    by_level: dict = {}
    for lv, _, d in control_rows:
        by_level.setdefault(lv, []).append(d)
    for lv in sorted(by_level):
        if max(by_level[lv]) >= threshold:
            return lv
    return None


def run_fd_v2(model, tok, device, dtype, *, probes, seeds=SEEDS,
              control_levels=CONTROL_LEVELS):
    """PREREG v2: positive-control gate first, then (if it passes) the full per-layer sweep
    at the chosen level. Returns dict with layers/kvar/vvar/n_cc/control_rows/chosen_level/
    damage (None when the gate fails)."""
    import torch
    from aepk_paging.harness.phase9_cw import _decode_under_cache
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    from aepk_paging.lossy_tier import quant_noise
    from aepk_paging.harness.phase10_grid import normalize_answer

    def prefix(prompt):
        ids = tok(prompt, return_tensors="pt").to(device).input_ids
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=True)
        return ids, out.past_key_values

    subset = []
    for pr in probes:
        ids, pkv = prefix(pr["prompt"])
        pg = dynamiccache_to_pages(pkv)
        _inject_pages(pkv, pg, dtype, device)
        t, _ = _decode_under_cache(model, tok, ids, pkv, device)
        if normalized_match(normalize_answer(t), pr["expected"], pr.get("alternatives")):
            subset.append((pr, pg))
    n_cc = len(subset)
    n_layers = len(subset[0][1]) if subset else 0

    kvar = np.zeros(n_layers); vvar = np.zeros(n_layers)
    for _, pg in subset:
        for L, page in enumerate(pg):
            kvar[L] += key_norm_variance(page)
            vvar[L] += value_norm_variance(page)
    if n_cc:
        kvar /= n_cc; vvar /= n_cc

    def layer_damage(L, lv):
        per_seed = []
        for sd in seeds:
            ok = 0
            for pr, _ in subset:
                ids, pkv = prefix(pr["prompt"])
                pg = dynamiccache_to_pages(pkv)
                noisy = [quant_noise(p, lv, sd * 1000 + p.layer)[0] if p.layer == L else p
                         for p in pg]
                _inject_pages(pkv, noisy, dtype, device)
                t, _ = _decode_under_cache(model, tok, ids, pkv, device)
                ok += int(normalized_match(normalize_answer(t), pr["expected"],
                                           pr.get("alternatives")))
            per_seed.append(ok / n_cc if n_cc else float("nan"))
        return 1.0 - float(np.mean(per_seed))

    ctl_ids = control_layer_ids(n_layers)
    control_rows = []
    chosen = None
    for lv in control_levels:                       # ascending -> smallest passing level
        dmgs = [layer_damage(L, lv) for L in ctl_ids]
        control_rows += list(zip([lv] * len(ctl_ids), ctl_ids, dmgs))
        if max(dmgs) >= CONTROL_DAMAGE_MIN:
            chosen = lv
            break

    damage = None
    if chosen is not None:
        damage = np.zeros(n_layers)
        for L in range(n_layers):
            damage[L] = layer_damage(L, chosen)
    return dict(layers=list(range(n_layers)), kvar=kvar, vvar=vvar, damage=damage,
                n_cc=n_cc, control_rows=control_rows, chosen_level=chosen)


def write_fd_report_v2(res, path="results/REPORT_phase10_fd_v2.md"):
    """PREREG v2 report: control table always; per-layer sweep + rho only if the gate passed.
    FD verdict line is a runtime f-string either way."""
    import os
    layers, kvar, vvar = res["layers"], res["kvar"], res["vvar"]
    damage, n_cc = res["damage"], res["n_cc"]
    chosen = res["chosen_level"]
    n = len(layers)
    L = [
        "# REPORT_phase10_fd_v2.md — Phase 10 step (7) FD redo with positive control",
        "",
        f"Model: {MODEL_ID} fp16 (CUDA). PREREG_phase10_fd_v2.md: positive-control rule fixed "
        f"FIRST — single-layer levels {list(CONTROL_LEVELS)} tried (ascending) on pre-named "
        f"layers first/mid/last; smallest level with max damage >= {CONTROL_DAMAGE_MIN} is used "
        "for the full sweep; if none reaches it the verdict is undetermined(no-response-regime) "
        f"and NO rho is computed. Clean-correct subset n_cc={n_cc}; seeds={list(SEEDS)}, seed "
        "derivation sd*1000+layer. Supersedes REPORT_phase10_fd.md (see its addendum).",
        "",
        "## Positive control (level x pre-named layer -> retention damage)",
        "",
        "| level | layer | damage |",
        "|-------|-------|--------|",
    ]
    for lv, ly, d in res["control_rows"]:
        L.append(f"| {lv} | {ly} | {d:.4f} |")
    L += ["", f"chosen_level={chosen}"]
    if chosen is None:
        L += [
            "",
            "## Interpretation",
            "No control level produced per-layer damage >= "
            f"{CONTROL_DAMAGE_MIN} on the pre-named layers — single-layer corruption up to "
            f"level {max(CONTROL_LEVELS)} sits below this workload's response threshold. Per "
            "the pre-registered rule NO rho is computed (a Spearman on a flat response can "
            "neither support nor refute FD).",
            "",
            f"FD: spearman=nan n_layers={n} verdict=undetermined(no-response-regime)",
        ]
        verdict = "undetermined(no-response-regime)"
        rho_k = float("nan")
    else:
        rho_k = spearman_rho(kvar, damage)
        rho_v = spearman_rho(vvar, damage)
        verdict = fd_verdict(rho_k)
        L += [
            "",
            f"## Full per-layer sweep at level={chosen}",
            "",
            "| layer | key_norm_var (clean) | value_norm_var (clean) | retention_damage |",
            "|-------|----------------------|------------------------|------------------|",
        ]
        for i in layers:
            L.append(f"| {i} | {kvar[i]:.5f} | {vvar[i]:.5f} | {damage[i]:.4f} |")
        L += [
            "",
            "## Interpretation",
            f"Spearman(key_norm_var, damage) = {rho_k:.4f} (primary). "
            f"Spearman(value_norm_var, damage) = {rho_v:.4f} (secondary). n_layers={n}. "
            f"Damage now has real dynamic range (control gate passed at level={chosen}), so "
            "supported/refuted are both meaningful; either is reported as-is.",
            "",
            f"FD: spearman={rho_k:.4f} n_layers={n} verdict={verdict}",
        ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return rho_k, n, verdict


def write_fd_report(layers, kvar, vvar, damage, n_cc,
                    path="results/REPORT_phase10_fd.md"):
    import os
    rho_k = spearman_rho(kvar, damage)
    rho_v = spearman_rho(vvar, damage)
    verdict = fd_verdict(rho_k)
    n = len(layers)
    L = [
        "# REPORT_phase10_fd.md — Phase 10 step (7) FENCED: fluctuation-dissipation analogue",
        "",
        f"Model: {MODEL_ID} fp16 (CUDA). From CLEAN KV only: per-layer variance of per-token key "
        f"norms (primary) and value norms (secondary), averaged over the {n_cc} clean-correct "
        f"probes. Prediction (PREREG, direction FIXED): higher clean key-norm fluctuation => more "
        f"retention damage when that layer alone is corrupted (quant_noise level={LEVEL}, "
        f"seeds={list(SEEDS)}, seed sd*1000+layer). Compared by Spearman rho. supported iff "
        f"rho>={RHO_SUPPORT}; null/refuted iff |rho|<{RHO_NULL}. Refuted is a real result.",
        "",
        "| layer | key_norm_var (clean) | value_norm_var (clean) | retention_damage |",
        "|-------|----------------------|------------------------|------------------|",
    ]
    for i in layers:
        L.append(f"| {i} | {kvar[i]:.5f} | {vvar[i]:.5f} | {damage[i]:.4f} |")
    L += [
        "",
        "## Interpretation",
        f"Spearman(key_norm_var, damage) = {rho_k:.4f} (primary). "
        f"Spearman(value_norm_var, damage) = {rho_v:.4f} (secondary). n_layers={n}. "
        "If supported, clean equilibrium fluctuations forecast corruption vulnerability — an FD "
        "analogue. If refuted (|rho| small), there is no such analogue on this workload and that "
        "is the finding, reported as-is (not reframed). A wrong-sign or weak rho is undetermined.",
        "",
        f"FD: spearman={rho_k:.4f} n_layers={n} verdict={verdict}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return rho_k, n, verdict


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from aepk_paging.harness.eval_set_large import LARGE_PROBES
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16,
                                                 device_map="cuda").eval()
    res = run_fd_v2(model, tok, "cuda", torch.float16, probes=LARGE_PROBES)
    rho, n, verdict = write_fd_report_v2(res)
    print(f"n_clean_correct={res['n_cc']} n_layers={n} chosen_level={res['chosen_level']}")
    for lv, ly, d in res["control_rows"]:
        print(f"  control level={lv} layer={ly} damage={d:.4f}")
    print(f"FD: spearman={rho:.4f} n_layers={n} verdict={verdict}")
