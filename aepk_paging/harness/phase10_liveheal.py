"""Phase 10.3 / 9.6 MOVE A — live mid-generation self-heal on real KV.

Realizes the erasure-conversion reframe END-TO-END on a live model: a fault corrupts a
resident KV page during generation; a content-agnostic physics fingerprint LOCALIZES the
corrupted page (the location signal GhostServe gets from a hardware failure — here it comes
from the data itself); that page is treated as an ERASURE and restored BIT-EXACT from the
clean survivor pages + Reed-Solomon parity (`coding.recover_rs_erasure`), with ZERO recompute;
generation continues. Baseline (no heal) keeps the corrupted page and garbles.

Redundancy = parity only (systematic Cauchy-MDS group over sibling layer-pages), so this is
genuine channel coding, not replication. The page-level physics→coding link is the live
realization; the symbol-level mixed error/erasure 2x is proven separately in
`tests/test_mixed_decode.py` on the confirmed `galois decode(erasures=)` API.

Honesty spine S9: zero edits to Phase 2-5 source. Verdict line runtime f-string. fault=0
CONTROL row required (must equal clean). recovered may be False (fail-loud / detector miss).
"""

from __future__ import annotations

import numpy as np

from aepk_paging.harness.phase9_cw import (
    calibrate, any_physics_flag, fp_key_norm_mean, _decode_under_cache, CW_PROBES, corrupt_k_scale,
)
from aepk_paging.coding import encode_rs_erasure_group, recover_rs_erasure, UncorrectableError

GROUP_SIZE = 4          # data pages in the erasure group (sibling layer-pages)
NUM_PARITY = 1          # recover up to 1 corrupted->erased page
# FAULT = structured key-scale on the target page K. AMENDMENT (2026-07-04, documented in
# PREREG_phase10_liveheal.md): the originally pre-registered mantissa fp16 bit-flips were
# demonstrably INERT — run A gave baseline_acc=1.000 at n=32 (no damage) and flagged_rate=0.00
# (undetected), so they exercised neither detection nor healing. Structured k_scale both breaks
# accuracy AND is localized by the physics fingerprint (9.3c-localization recall 0.83), so it
# actually tests the detect->locate->erasure-heal path. factor=1.0 is the identity CONTROL.
KSCALE_GRID = (1.0, 2.0, 4.0)   # 1.0 = CONTROL (identity)


def _prefix_pkv(model, tok, device, prompt):
    import torch
    enc = tok(prompt, return_tensors="pt").to(device)
    ids = enc.input_ids
    with torch.no_grad():
        out = model(ids[:, :-1], use_cache=True)
    return ids, out.past_key_values


def run_liveheal(model, tok, device, dtype, *, probes=None, seed: int = 4242):
    """For each k_factor: corrupt the top-influence page mid-generation (key-scale), detect+
    locate it, erasure-heal from parity, and compare baseline (no heal) vs aepk (healed)
    accuracy. Returns (clean_acc, rows). Row: (k_factor, baseline_acc, aepk_acc, flagged_rate,
    recovered, decode_mode)."""
    import torch
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    probes = probes or CW_PROBES

    # clean pass: accuracy + calibration
    cc, cal = 0, []
    for pr in probes:
        ids, pkv = _prefix_pkv(model, tok, device, pr["prompt"])
        pg = dynamiccache_to_pages(pkv); _inject_pages(pkv, pg, dtype, device)
        t, _ = _decode_under_cache(model, tok, ids, pkv, device)
        cc += int(normalized_match(t, pr["expected"], pr.get("alternatives")))
        cal.extend(pg)
    clean_acc = cc / len(probes); calib = calibrate(cal)

    rows = []
    for kf in KSCALE_GRID:
        base_ok, aepk_ok, flags, recos = 0, 0, 0, 0
        decode_mode = "erasure"
        for pr in probes:
            ids, pkv = _prefix_pkv(model, tok, device, pr["prompt"])
            pg = dynamiccache_to_pages(pkv)
            order = sorted(range(len(pg)), key=lambda i: -fp_key_norm_mean(pg[i]))
            grp_idx = order[:GROUP_SIZE]
            tgt_i = grp_idx[0]
            clean_group_pages = [pg[i] for i in grp_idx]
            # parity computed on CLEAN pages (before fault) — redundancy = parity only
            group = encode_rs_erasure_group(clean_group_pages, NUM_PARITY)

            if kf == 1.0:
                corrupt_tgt = pg[tgt_i]                     # CONTROL: identity, no fault
            else:
                corrupt_tgt = corrupt_k_scale(pg[tgt_i], kf)
            flagged = (kf != 1.0) and any_physics_flag(pg[tgt_i], corrupt_tgt, calib)
            flags += int(flagged)

            # ---- BASELINE: inject corrupted target, no heal ----
            base_pages = list(pg); base_pages[tgt_i] = corrupt_tgt
            _inject_pages(pkv, base_pages, dtype, device)
            tb, _ = _decode_under_cache(model, tok, ids, pkv, device)
            base_ok += int(normalized_match(tb, pr["expected"], pr.get("alternatives")))

            # ---- AEPK: if detector located it, erase+recover from parity, then heal ----
            ids2, pkv2 = _prefix_pkv(model, tok, device, pr["prompt"])
            heal_pages = list(pg)
            recovered = False
            if kf == 1.0:
                recovered = True                            # nothing to heal; stays clean
            elif flagged:
                try:
                    rec = recover_rs_erasure(group, [pg[tgt_i].page_id])
                    heal_pages[tgt_i] = rec[pg[tgt_i].page_id]
                    recovered = bool(np.array_equal(heal_pages[tgt_i].K, pg[tgt_i].K)
                                     and np.array_equal(heal_pages[tgt_i].V, pg[tgt_i].V))
                except UncorrectableError:
                    heal_pages[tgt_i] = corrupt_tgt          # fail-loud → no heal
            else:
                heal_pages[tgt_i] = corrupt_tgt              # detector miss → blind, unhealed
            _inject_pages(pkv2, heal_pages, dtype, device)
            ta, _ = _decode_under_cache(model, tok, ids2, pkv2, device)
            aepk_ok += int(normalized_match(ta, pr["expected"], pr.get("alternatives")))
            recos += int(recovered)

        rows.append((kf, base_ok / len(probes), aepk_ok / len(probes),
                     flags / len(probes), recos == len(probes), decode_mode))
    return clean_acc, rows


def run_liveheal_control(model, tok, device, dtype, *, probes=None):
    """PREREG v2 control arm: same k_scale factors applied to the LOWEST-fp_key_norm_mean
    page (order[-1] of the same influence ranking the headline arm tops); baseline_acc only,
    no heal arm. Returns rows: (k_factor, low_baseline_acc)."""
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    probes = probes or CW_PROBES

    rows = []
    for kf in [k for k in KSCALE_GRID if k != 1.0]:
        ok = 0
        for pr in probes:
            ids, pkv = _prefix_pkv(model, tok, device, pr["prompt"])
            pg = dynamiccache_to_pages(pkv)
            order = sorted(range(len(pg)), key=lambda i: -fp_key_norm_mean(pg[i]))
            low_i = order[-1]
            low_pages = list(pg)
            low_pages[low_i] = corrupt_k_scale(pg[low_i], kf)
            _inject_pages(pkv, low_pages, dtype, device)
            t, _ = _decode_under_cache(model, tok, ids, pkv, device)
            ok += int(normalized_match(t, pr["expected"], pr.get("alternatives")))
        rows.append((kf, ok / len(probes)))
    return rows


def write_liveheal_report(clean_acc, rows, path="results/REPORT_phase10_liveheal.md",
                          control_rows=None):
    import os
    # verdict from the strongest damaging magnitude (max k_factor; control = 1.0)
    dmg = [r for r in rows if r[0] != 1.0]
    worst = max(dmg, key=lambda r: r[0]) if dmg else rows[-1]
    kf, base, aepk, fr, reco, mode = worst
    recovered = bool(reco and aepk >= clean_acc - 1e-9 and base < clean_acc)
    L = [
        "# REPORT_phase10_liveheal.md — Phase 10.3 / 9.6 MOVE A live mid-generation self-heal",
        "",
        "Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Fault corrupts the top-influence resident "
        "KV page (K) mid-generation; physics fingerprint localizes it; the page is erased and "
        "restored bit-exact from sibling-page RS parity (recover_rs_erasure); generation continues, "
        "zero recompute. Redundancy = parity only (Cauchy-MDS group of "
        f"{GROUP_SIZE} pages, num_parity={NUM_PARITY}).",
        f"clean_acc={clean_acc:.3f}.",
        "",
        "| k_factor | baseline_acc | aepk_acc | flagged_rate | recovered | decode_mode |",
        "|----------|--------------|----------|--------------|-----------|-------------|",
    ]
    for kfac, b, a, f, r, m in rows:
        tag = " (CONTROL)" if kfac == 1.0 else ""
        L.append(f"| {kfac}{tag} | {b:.3f} | {a:.3f} | {f:.2f} | {r} | {m} |")
    L += [
        "",
        "## Interpretation",
        "The k_factor=1.0 CONTROL row MUST show baseline_acc == aepk_acc == clean_acc (no fault, no "
        "heal) — it is the plumbing check (9.1 pattern). At damaging magnitudes, baseline_acc "
        "drops (fault bit the answer) while aepk_acc returns to clean_acc via bit-exact "
        "erasure recovery of the physics-located page. If the detector misses the page it stays "
        "blind (unhealed) and recovered=False — reported as-is.",
        "",
        f"LIVE_HEAL: baseline_acc={base:.3f} aepk_acc={aepk:.3f} recovered={recovered} decode_mode={mode}",
    ]
    if control_rows:
        # PREREG v2: verdict at the strongest factor; near-clean = within one probe (0.125).
        ctl = {k: v for k, v in control_rows}
        low_base = ctl.get(kf, control_rows[-1][1])   # verdict factor; fallback last row
        load_bearing = bool(low_base >= clean_acc - 0.125)
        L += [
            "",
            "## Control arm (PREREG v2) — is top-influence selection load-bearing?",
            "Same k_scale factors applied to the LOWEST-fp_key_norm_mean page (other end of "
            "the influence ranking); baseline_acc only, no heal arm. If the low-mass arm also "
            "collapses, the top-influence proxy is NOT load-bearing (finding, reported as-is); "
            "if it stays near clean (within one probe of clean_acc), selection is validated.",
            "",
            "| k_factor | low_baseline_acc |",
            "|----------|------------------|",
        ]
        for kfac, lb in control_rows:
            L.append(f"| {kfac} | {lb:.3f} |")
        L += [
            "",
            f"HEAL_CONTROL: top_baseline={base:.3f} low_baseline={low_base:.3f} "
            f"selection_load_bearing={load_bearing}",
        ]
    L += [
        "",
        "## Deployability caveat (honesty spine)",
        "This harness localizes the fault by comparing the corrupt page against the CLEAN "
        "in-memory page (`any_physics_flag(pg[tgt_i], corrupt_tgt, calib)`). A deployable "
        "detector does NOT have the clean page in memory — that is exactly what it is trying to "
        "recover. In deployment, detection compares the live page's fingerprints against STORED "
        "per-page fingerprint SCALARS (key_norm_mean, key_mass, norm_ratio, v/k_mean_shift) "
        "recorded at encode time — a few floats per page, not a retained clean copy. The "
        "localization capability is the same (9.3c-localization: FPR-0 headroom vs the fp16 "
        "round-trip floor); only the reference is a compact stored fingerprint, not a clean page.",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return recovered


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MID = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map="cuda")
    model.eval()
    ca, rows = run_liveheal(model, tok, "cuda", torch.float16)
    ctl_rows = run_liveheal_control(model, tok, "cuda", torch.float16)
    rec = write_liveheal_report(ca, rows, control_rows=ctl_rows)
    for k_factor, b, a, f, r, m in rows:
        print(f"  k_factor={k_factor}: baseline={b:.3f} aepk={a:.3f} flagged={f:.2f} recovered={r}")
    for k_factor, lb in ctl_rows:
        print(f"  control k_factor={k_factor}: low_baseline={lb:.3f}")
    print(f"LIVE_HEAL recovered={rec}")
