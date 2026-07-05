"""Phase 10 step (4) — thermodynamic parity allocation.

The free-energy residency law (Phase 5) treats Gibbs attention-mass as a Boltzmann utility
weight. Here we apply that SAME law to PARITY allocation: parity protection is a scarce bit
budget, and it should be spent where the attention-mass is.

Two policies protect the SAME critical set (the pages carrying the top `mass_target` fraction
of Gibbs attention-mass) to one erasure each:
  * uniform  — content-agnostic: cannot tell which pages matter, so it must place a parity
               block on EVERY parity group  → cost = G blocks.
  * thermo   — Gibbs-ranked: places a parity block only on groups that intersect the critical
               set → cost = |groups ∩ critical| blocks.
Both cover the identical critical set (iso_protection), but thermo buys it for fewer parity
bits when attention-mass is concentrated. When mass is diffuse the two coincide (ALLOWED to
FAIL — reported as-is).

Honesty spine S9: zero edits to Phase 2-5 source. Reuses residency.TierCostModel.resident_bits
for the honest per-block bit cost. Deterministic (no RNG). Verdict line is a runtime f-string.
"""

from __future__ import annotations

import numpy as np

from aepk_paging.residency import TierCostModel

GROUP_SIZE = 4          # sibling layer-pages per parity group (matches live-heal harness)
MASS_TARGET = 0.5       # critical set = smallest set reaching this fraction of Gibbs mass
NUM_PARITY = 1          # one parity block per protected group (single-erasure recovery)


def gibbs_weights(pages, kT: float = 1.0) -> np.ndarray:
    """Boltzmann distribution over pages from attention_mass at temperature kT (default 1.0 —
    the pre-registered headline convention, same as residency's Gibbs utility). Returns a
    length-len(pages) prob vector summing to 1. Higher kT flattens the distribution."""
    m = np.array([float(p.attention_mass) for p in pages], dtype=np.float64)
    m = (m - m.max()) / kT
    w = np.exp(m)
    return w / w.sum()


def parity_groups(n_pages: int, group_size: int = GROUP_SIZE) -> list[list[int]]:
    """Chunk page indices [0..n) into groups of `group_size` in order (layer order)."""
    return [list(range(s, min(s + group_size, n_pages))) for s in range(0, n_pages, group_size)]


def critical_set(pages, mass_target: float = MASS_TARGET, kT: float = 1.0) -> set[int]:
    """Smallest set of page indices whose summed Gibbs weight reaches `mass_target`
    (greedy by descending mass) at temperature kT. kT=1.0 is the pre-registered headline.
    These are the pages the free-energy law says to protect."""
    w = gibbs_weights(pages, kT)
    order = sorted(range(len(pages)), key=lambda i: -w[i])
    chosen: set[int] = set()
    acc = 0.0
    for i in order:
        chosen.add(i)
        acc += w[i]
        if acc >= mass_target:
            break
    return chosen


def recoverable_critical(crit, groups, blocks_on) -> set[int]:
    """Critical pages that are single-erasure recoverable, judged from BLOCK PLACEMENT ALONE.

    A critical page is protected iff its parity group carries >=NUM_PARITY parity blocks.
    `blocks_on` maps group index -> block count actually placed by a policy. This deliberately
    does NOT filter by crit membership of the surviving pages — it reads recoverability off the
    physical placement, so a policy that fails to place a block on a critical group yields a
    protected set that is strictly smaller than crit (control can fail)."""
    page_group = {i: g for g, grp in enumerate(groups) for i in grp}
    return {i for i in crit if blocks_on.get(page_group[i], 0) >= NUM_PARITY}


def sweep_temperatures(pages) -> list[tuple[str, float]]:
    """Pre-registered kT sweep (amendment v2, descriptive artifact check): kT=1 (headline),
    mean|Δmass| (typical successive mass gap), std(mass) (spread). Degenerate (<=0)
    temperatures are dropped and NOT silently skipped — caller reports what survived."""
    m = np.sort(np.array([float(p.attention_mass) for p in pages], dtype=np.float64))
    dmass = float(np.mean(np.abs(np.diff(m)))) if len(m) > 1 else 0.0
    smass = float(np.std(m))
    temps = [("kT=1", 1.0), ("mean|dmass|", dmass), ("std(mass)", smass)]
    return [(lbl, t) for lbl, t in temps if t > 0.0]


def crit_sizes_by_temperature(pages, temps, mass_target: float = MASS_TARGET):
    """[(label, kT, crit_set_size)] — critical-set size at each swept temperature (same
    greedy rule and mass_target as the headline)."""
    return [(lbl, t, len(critical_set(pages, mass_target, kT=t))) for lbl, t in temps]


def allocate(pages, *, group_size: int = GROUP_SIZE, mass_target: float = MASS_TARGET):
    """Return (uniform_blocks, thermo_blocks, iso_protection, bits_per_block) for one probe.

    iso_protection is the CONTROL: the identical critical set is single-erasure recoverable
    under both policies. Recoverability is read off the actual per-group block placement (via
    recoverable_critical), NOT asserted by construction — so a broken placement can fail it."""
    n = len(pages)
    groups = parity_groups(n, group_size)
    crit = critical_set(pages, mass_target)
    thermo_group_idx = {g for g, grp in enumerate(groups) if crit.intersection(grp)}

    uniform_blocks = len(groups) * NUM_PARITY
    thermo_blocks = len(thermo_group_idx) * NUM_PARITY

    # physical block placement per policy: uniform blocks every group; thermo blocks only the
    # critical-touching groups. Protection is then derived from placement, independently.
    uniform_on = {g: NUM_PARITY for g in range(len(groups))}
    thermo_on = {g: NUM_PARITY for g in thermo_group_idx}
    uniform_protected = recoverable_critical(crit, groups, uniform_on)
    thermo_protected = recoverable_critical(crit, groups, thermo_on)
    iso_protection = bool(uniform_protected == thermo_protected == crit)

    bits_per_block = TierCostModel().resident_bits(pages[0]) * NUM_PARITY
    return uniform_blocks, thermo_blocks, iso_protection, bits_per_block


# ---------------------------------------------------------------------------
# Step 6 — physics-free baseline (PREREG v3). Does the free-energy vocabulary earn its name?
# topk_norm protects the k highest raw-attention_mass pages directly — NO Gibbs / softmax /
# Boltzmann / free-energy framing — with k = |critical set| (matched protection count). This
# section does NOT touch the pre-registered kT=1 headline path (allocate/critical_set/gibbs).
# ---------------------------------------------------------------------------

def topk_norm_set(pages, k: int) -> set[int]:
    """Physics-free: indices of the k pages with the highest RAW attention_mass. No softmax,
    no temperature, no Boltzmann utility — a plain magnitude sort. Deterministic tie-break
    (mass descending, then page index ascending)."""
    order = sorted(range(len(pages)), key=lambda i: (-float(pages[i].attention_mass), i))
    return set(order[:max(0, k)])


def allocate_topk(pages, *, group_size: int = GROUP_SIZE, mass_target: float = MASS_TARGET):
    """topk_norm parity allocation with k = |thermo critical set|. Returns
    (topk_blocks, topk_protected, thermo_protected, sets_identical, bits_per_block). Protection
    is read off block placement (recoverable_critical), same honest accounting as allocate()."""
    groups = parity_groups(len(pages), group_size)
    crit = critical_set(pages, mass_target)               # free-energy set (for k + comparison)
    topk = topk_norm_set(pages, len(crit))                # physics-free set, matched count
    topk_groups = {g for g, grp in enumerate(groups) if topk.intersection(grp)}
    thermo_groups = {g for g, grp in enumerate(groups) if crit.intersection(grp)}
    topk_blocks = len(topk_groups) * NUM_PARITY
    topk_protected = recoverable_critical(topk, groups, {g: NUM_PARITY for g in topk_groups})
    thermo_protected = recoverable_critical(crit, groups, {g: NUM_PARITY for g in thermo_groups})
    sets_identical = bool(topk == crit)                   # physics-free set == free-energy set?
    bits_per_block = TierCostModel().resident_bits(pages[0]) * NUM_PARITY
    return topk_blocks, topk_protected, thermo_protected, sets_identical, bits_per_block


def run_parity_alloc(model, tok, device, dtype, *, probes=None,
                     group_size: int = GROUP_SIZE, mass_target: float = MASS_TARGET):
    """Get real KV pages per probe, allocate parity two ways, aggregate. Returns
    (rows, uniform_cost_bits, thermo_cost_bits, iso_protection_all, sensitivity, baseline), where
    sensitivity = [(label, mean_kT, mean_crit_size)] over probes (PREREG v2, kT artifact check)
    and baseline = (topk_cost_bits, sets_identical_all) (PREREG v3, physics-free topk_norm).
    Deterministic."""
    import torch
    from aepk_paging.harness.phase9_cw import CW_PROBES
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    probes = probes or CW_PROBES

    rows = []
    tot_uni = tot_thermo = tot_topk = 0
    iso_all = True
    sets_identical_all = True
    sens_acc: dict[str, list[tuple[float, int]]] = {}   # label -> [(kT, crit_size)]
    for pr in probes:
        enc = tok(pr["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(enc.input_ids[:, :-1], use_cache=True)
        pages = dynamiccache_to_pages(out.past_key_values)
        ub, tb, iso, bpb = allocate(pages, group_size=group_size, mass_target=mass_target)
        tk_blocks, tk_prot, th_prot, sets_id, _ = allocate_topk(
            pages, group_size=group_size, mass_target=mass_target)
        crit = sorted(critical_set(pages, mass_target))
        rows.append((pr["prompt"][:32], len(pages), len(parity_groups(len(pages), group_size)),
                     len(crit), ub, tb, iso))
        tot_uni += ub * bpb
        tot_thermo += tb * bpb
        tot_topk += tk_blocks * bpb
        iso_all = iso_all and iso
        sets_identical_all = sets_identical_all and sets_id and (tk_prot == th_prot)
        for lbl, t, sz in crit_sizes_by_temperature(pages, sweep_temperatures(pages), mass_target):
            sens_acc.setdefault(lbl, []).append((t, sz))
    sensitivity = [(lbl, float(np.mean([t for t, _ in v])), float(np.mean([s for _, s in v])))
                   for lbl, v in sens_acc.items()]
    return rows, tot_uni, tot_thermo, iso_all, sensitivity, (tot_topk, bool(sets_identical_all))


def write_parity_alloc_report(rows, uniform_cost, thermo_cost, iso_all,
                              path="results/REPORT_phase10_parity_alloc.md", sensitivity=None,
                              baseline=None):
    import os
    L = [
        "# REPORT_phase10_parity_alloc.md — Phase 10 step (4) thermodynamic parity allocation",
        "",
        "Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). The free-energy law (Gibbs attention-mass "
        "as a Boltzmann utility, Phase 5) allocates a scarce parity-bit budget. Two policies "
        "protect the IDENTICAL critical set (smallest set reaching "
        f"{MASS_TARGET:.2f} of Gibbs mass) to one erasure each: uniform (a parity block on every "
        f"group of {GROUP_SIZE} sibling layer-pages) vs thermo (a block only on groups holding a "
        "critical page). Cost = parity bits = blocks x resident_bits(page). Deterministic (no RNG).",
        "",
        "| probe | pages | groups | crit_pages | uniform_blocks | thermo_blocks | iso |",
        "|-------|-------|--------|-----------|----------------|---------------|-----|",
    ]
    for prompt, npg, ng, nc, ub, tb, iso in rows:
        L.append(f"| {prompt} | {npg} | {ng} | {nc} | {ub} | {tb} | {iso} |")
    ratio = (thermo_cost / uniform_cost) if uniform_cost else float("nan")
    L += [
        "",
        "## Interpretation",
        "iso must be True on every row: both policies protect the identical critical set (the "
        "CONTROL / plumbing check). thermo_blocks <= uniform_blocks because real attention-mass "
        "is concentrated in a minority of layers, so the critical set touches fewer than all "
        f"groups. Aggregate parity cost: thermo is {ratio:.3f}x uniform. If mass were diffuse the "
        "critical set would span every group and the two costs would coincide (the law buys "
        "nothing there) — reported as-is, not tuned away.",
        "",
        f"PARITY_ALLOC: uniform_cost={uniform_cost} thermo_cost={thermo_cost} "
        f"iso_protection={bool(iso_all)}",
    ]
    if sensitivity:
        base = next((sz for lbl, _, sz in sensitivity if lbl == "kT=1"), sensitivity[0][2])
        others = [sz for lbl, _, sz in sensitivity if lbl != "kT=1"]
        hi = max(others) if others else base
        robust = hi <= max(base + 1.0, base * 1.5)
        L += [
            "",
            "## kT sensitivity (PREREG amendment v2 — descriptive artifact check)",
            "`gibbs_weights` softmaxes raw key-norms (O(5-50)); at the pre-registered kT=1.0 the "
            "distribution is near one-hot, so the critical set can collapse to ~1 page. This "
            "section recomputes the mean critical-set size (across the 8 probes, same "
            f"mass_target={MASS_TARGET:.2f}) at swept temperatures. The kT=1.0 HEADLINE above is "
            "unchanged; these rows are additive.",
            "",
            "| kT label | mean kT | mean crit_pages |",
            "|----------|---------|-----------------|",
        ]
        for lbl, kt, sz in sensitivity:
            L.append(f"| {lbl} | {kt:.3f} | {sz:.2f} |")
        verdict = ("temperature-ROBUST" if robust
                   else "likely a kT=1 ARTIFACT (grows when the softmax is flattened)")
        L += [
            "",
            f"KT_SENSITIVITY: crit_pages={base:.2f} at kT=1 vs up to {hi:.2f} at higher kT -> "
            f"{verdict}.",
        ]
    if baseline is not None:
        topk_cost, sets_identical = baseline
        tie = (topk_cost == thermo_cost) and sets_identical
        L += [
            "",
            "## Physics-free baseline (PREREG v3 — does the free-energy vocabulary earn its name?)",
            "topk_norm protects the k highest RAW attention_mass pages directly (no Gibbs / "
            "softmax / Boltzmann / free-energy framing), with k = |critical set| chosen by the "
            "thermo policy (matched protection count). Question: does the free-energy formalism "
            "beat this physics-free heuristic anywhere on these probes?",
            "",
            (f"Answer: NO — topk_norm and thermo select the IDENTICAL protected set for the "
             f"identical parity cost on every probe (softmax is order-preserving in "
             f"attention_mass, so the smallest set reaching {MASS_TARGET:.2f} of Gibbs mass IS "
             f"the top-k by raw mass). The thermodynamic vocabulary is DECORATIVE for allocation "
             f"on this workload; the physics claim rests on steps 5/7, not here."
             if tie else
             "Answer: the policies DIFFER on at least one probe (cost or protected set) — "
             "reported as-is; see per-probe detail."),
            "",
            f"BASELINE_PARITY: thermo_cost={thermo_cost} topk_cost={topk_cost} "
            f"sets_identical={bool(sets_identical)}",
        ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return uniform_cost, thermo_cost, bool(iso_all)


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MID = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map="cuda")
    model.eval()
    rows, uc, tc, iso, sens, baseline = run_parity_alloc(model, tok, "cuda", torch.float16)
    write_parity_alloc_report(rows, uc, tc, iso, sensitivity=sens, baseline=baseline)
    for r in rows:
        print("  ", r)
    print(f"PARITY_ALLOC: uniform_cost={uc} thermo_cost={tc} iso_protection={iso}")
    for lbl, kt, sz in sens:
        print(f"   kT-sens {lbl}: mean_kT={kt:.3f} mean_crit_pages={sz:.2f}")
    print(f"BASELINE_PARITY: thermo_cost={tc} topk_cost={baseline[0]} sets_identical={baseline[1]}")
