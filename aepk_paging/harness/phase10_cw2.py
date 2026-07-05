"""Phase 10.2 CW-2 — natural-fault confident-wrong retest (GPU sweep runners).

Two runners, both emitting harness-computed verdict lines (tests assert LINE EXISTS only):
  10.2a run_cw2_bitflip -> CW2_BITFLIP  (raw-fp16 single-bit upsets, whole-page)
  10.2b run_cw2_needle  -> CW2_NEEDLE   (needle TOKEN-ROW corruption; see structural note)

STRUCTURAL NOTE (verify-before-code, 2026-07-03): dynamiccache_to_pages returns ONE page
PER LAYER, each spanning ALL tokens (token_range=(0, seq_len)); there is NO page that
isolates the needle tokens. The PREREG phrase "corrupt the page spanning the needle" is
therefore corrected here: needle corruption targets the needle TOKEN ROWS inside layer-pages
(`_corrupt_token_rows`), not a whole page. 10.2a keeps whole-page corruption (matches 9-CW).

Reuses the FIXED, calibrated phase9_cw fingerprints (NOT degenerate detect.py). Confidence =
real model output entropy through the corrupted cache. Nothing tuned to force SHOWN. The
confident-wrong cell definition (dacc<=-0.25 AND blind AND flag_rate>=0.5) is the FIXED bar
from PREREG_phase10_cw2.md.
"""

from __future__ import annotations

import numpy as np

from aepk_paging.harness.phase9_cw import (
    calibrate, any_physics_flag, fp_key_norm_mean, _decode_under_cache, token_entropy, CW_PROBES,
)
from aepk_paging.harness.fault_fp16 import bitflip_fp16, _REGION_BITS
from aepk_paging.kv_page import KVPage

# FIXED grid (PREREG). 3 regions x 3 n_flips x 2 target_k x 2 tensors = 36 cells (+ n=0 control).
CW2_REGIONS = ("exponent", "mantissa", "sign")
CW2_NFLIPS = (1, 3, 5)
CW2_TARGET_KS = (1, 3)
CW2_TENSORS = ("K", "V")
DACC_MIN = 0.25


def _corrupt_token_rows(page: KVPage, token_idx, n_flips: int, region: str, seed: int, tensor: str) -> KVPage:
    """Needle variant: flip fp16 bits ONLY in the given token rows of the chosen tensor.
    Builds a sub-page over those rows, corrupts it, writes the rows back. Deterministic."""
    K = np.asarray(page.K, np.float32).copy()
    V = np.asarray(page.V, np.float32).copy()
    idx = np.asarray(sorted(set(token_idx)), dtype=int)
    sub = KVPage(page.page_id, page.layer, (0, len(idx)),
                 K[idx], V[idx], page.precision_tag, page.attention_mass)
    flipped = bitflip_fp16(sub, n_flips, region, seed, tensor)
    if tensor == "K":
        K[idx] = flipped.K
    else:
        V[idx] = flipped.V
    return KVPage(page.page_id, page.layer, page.token_range, K, V,
                  f"{page.precision_tag}+needle[{tensor},{region},n{n_flips}]", page.attention_mass)


# --- 10.2b needle: a long passage with planted facts at locatable token spans ---
NEEDLE_FILLER = (
    "In the archive report the following records were noted for later review. "
    "Several unrelated notes follow about weather patterns, logistics timetables, "
    "staffing rotations, and scheduling constraints that add surrounding context but "
    "carry no bearing on the specific records themselves. The reviewer is reminded that "
    "only the explicitly stated records below are authoritative, and that the narrative "
    "padding around them exists solely to establish a realistic long-context setting for "
    "retrieval. Additional filler describes routine maintenance windows and archival "
    "procedures that are standard and unremarkable in every respect. "
)
NEEDLE_FACTS = (
    ("The vault access code is 7734.", "What is the vault access code?", "7734"),
    ("The project codename is Falcon.", "What is the project codename?", "Falcon"),
    ("The origin city is Lisbon.", "What is the origin city?", "Lisbon"),
    ("The reserved channel is 88.", "What is the reserved channel?", "88"),
    ("The backup site is Delta.", "What is the backup site?", "Delta"),
    ("The clearance level is Amber.", "What is the clearance level?", "Amber"),
    ("The record year is 1990.", "What is the record year?", "1990"),
    ("The unit prefix is Zeta.", "What is the unit prefix?", "Zeta"),
)
NEEDLE_PASSAGE = NEEDLE_FILLER + " ".join(f for f, _, _ in NEEDLE_FACTS) + " "

# corruption specs applied to the needle rows only (PREREG 10.2b)
def _needle_corruptions():
    from aepk_paging.harness.phase9_cw import corrupt_v_bias
    from aepk_paging.lossy_tier import quant_noise
    return {
        "bitflip_exp_n1": lambda sub, s: bitflip_fp16(sub, 1, "exponent", s, "K"),
        "quant_noise_0.3": lambda sub, s: quant_noise(sub, 0.3, s)[0],
        "v_bias_8.0": lambda sub, s: corrupt_v_bias(sub, 8.0, s),
    }


def _apply_to_rows(page: KVPage, rows, corr_fn, seed: int) -> KVPage:
    """Apply a KVPage->KVPage corruption to ONLY the given token rows, write back."""
    K = np.asarray(page.K, np.float32).copy()
    V = np.asarray(page.V, np.float32).copy()
    idx = np.asarray(sorted(set(rows)), dtype=int)
    sub = KVPage(page.page_id, page.layer, (0, len(idx)),
                 K[idx], V[idx], page.precision_tag, page.attention_mass)
    out = corr_fn(sub, seed)
    K[idx] = np.asarray(out.K, np.float32)
    V[idx] = np.asarray(out.V, np.float32)
    return KVPage(page.page_id, page.layer, page.token_range, K, V,
                  f"{page.precision_tag}+needlerows", page.attention_mass)


def _locate_needle_rows(tok, prompt, value):
    """Token rows covering `value` in `prompt`, verified to decode back to it."""
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].tolist()
    ci = prompt.find(value)
    if ci < 0:
        return None
    lo, hi = ci, ci + len(value)
    rows = [i for i, (a, b) in enumerate(offs) if a < hi and b > lo and b > a]
    ids = enc["input_ids"][0].tolist()
    decoded = tok.decode([ids[i] for i in rows]).strip()
    return rows if value in decoded else None


def _prefix_pkv(model, tok, device, prompt):
    import torch
    enc = tok(prompt, return_tensors="pt").to(device)
    ids = enc.input_ids
    with torch.no_grad():
        out = model(ids[:, :-1], use_cache=True)
    return ids, out.past_key_values


def run_cw2_bitflip(model, tok, device, dtype, *, probes=None, seed: int = 4242):
    """10.2a: sweep {region x n_flips x target_k x tensor} whole-page fp16 bit upsets.
    Returns (clean_acc, clean_entropy, entropy_bar, calib, rows). Each row:
    (region, n_flips, target_k, tensor, dacc, dentropy, blind, flag_rate, confident_wrong)."""
    import torch
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages
    probes = probes or CW_PROBES

    cc, ce, cal = 0, [], []
    for pr in probes:
        ids, pkv = _prefix_pkv(model, tok, device, pr["prompt"])
        pg = dynamiccache_to_pages(pkv); _inject_pages(pkv, pg, dtype, device)
        t, e = _decode_under_cache(model, tok, ids, pkv, device)
        cc += int(normalized_match(t, pr["expected"], pr.get("alternatives")))
        ce.append(e); cal.extend(pg)
    clean_acc = cc / len(probes); calib = calibrate(cal)
    bar = float(np.std(ce)) if len(ce) > 1 else 0.1; clean_ent = float(np.mean(ce))

    rows = []
    for region in CW2_REGIONS:
        for nfl in CW2_NFLIPS:
            for tk in CW2_TARGET_KS:
                for tensor in CW2_TENSORS:
                    corr, des, fh, nonfin = 0, [], 0, 0
                    for pr in probes:
                        ids, pkv = _prefix_pkv(model, tok, device, pr["prompt"])
                        pg = dynamiccache_to_pages(pkv)
                        order = sorted(range(len(pg)), key=lambda i: -fp_key_norm_mean(pg[i]))
                        tgt = set(order[:tk]); npg = []; fl = False; nf = False
                        for i, p in enumerate(pg):
                            if i in tgt:
                                c = bitflip_fp16(p, nfl, region, seed + i, tensor)
                                fl = fl or any_physics_flag(p, c, calib)
                                # a non-finite KV page is caught by the CHEAPEST content-agnostic
                                # check (finiteness) AND makes output logits NaN -> "confidence" is
                                # undefined (token_entropy misreads all-NaN as ~0). Such a probe is
                                # NOT a blind confident-wrong case: confidence/finiteness trivially
                                # fires. Track it so it forces blind=False for the cell.
                                if not (np.isfinite(c.K).all() and np.isfinite(c.V).all()):
                                    nf = True
                                npg.append(c)
                            else:
                                npg.append(p)
                        _inject_pages(pkv, npg, dtype, device)
                        t, e = _decode_under_cache(model, tok, ids, pkv, device)
                        corr += int(normalized_match(t, pr["expected"], pr.get("alternatives")))
                        des.append(e); fh += int(fl); nonfin += int(nf)
                    dacc = corr / len(probes) - clean_acc
                    dent = float(np.mean(des)) - clean_ent
                    nonfinite_rate = nonfin / len(probes)
                    # blind only if confidence would NOT fire: entropy within bar AND no probe went
                    # non-finite (a non-finite cache is a trivially-detected, confidence-undefined case).
                    blind = (dent <= bar) and (nonfinite_rate == 0.0)
                    fr = fh / len(probes)
                    cw = (dacc <= -DACC_MIN) and blind and (fr >= 0.5)
                    rows.append((region, nfl, tk, tensor, dacc, dent, blind, fr, cw, nonfinite_rate))
    return clean_acc, clean_ent, bar, calib, rows


def write_cw2_bitflip_report(clean_acc, clean_ent, bar, calib, rows,
                             path="results/REPORT_phase10_cw2_bitflip.md"):
    import os
    any_cw = any(r[8] for r in rows)
    verdict = "SHOWN" if any_cw else "NOT_SHOWN"
    n_cw = sum(1 for r in rows if r[8])
    n_nonfin = sum(1 for r in rows if r[9] > 0.0)
    L = [
        "# REPORT_phase10_cw2_bitflip.md — Phase 10.2a raw-fp16 bit-upset confident-wrong test",
        "",
        "Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Natural non-ECC DRAM single-event-upset "
        "fault model (bitflip_fp16, whole-page). Confidence = real output entropy through the "
        "corrupted cache. Confident-wrong cell (FIXED, PREREG): dacc<=-0.25 AND blind AND flag>=0.5.",
        f"clean_acc={clean_acc:.3f} clean_entropy={clean_ent:.3f} entropy_bar={bar:.3f}.",
        "Calibrated tau: " + ", ".join(f"{k}={v:.3g}" for k, v in calib.tau.items()) + ".",
        "",
        "| region | n_flips | tk | tensor | dacc | dentropy | blind | flag_rate | nonfinite | confident_wrong |",
        "|--------|---------|----|--------|------|----------|-------|-----------|-----------|-----------------|",
    ]
    for region, nfl, tk, tensor, dacc, dent, blind, fr, cw, nf in rows:
        L.append(f"| {region} | {nfl} | {tk} | {tensor} | {dacc:+.3f} | {dent:+.3f} | "
                 f"{blind} | {fr:.2f} | {nf:.2f} | {'YES' if cw else 'no'} |")
    L += [
        "",
        "## Interpretation",
        "Per PREREG prediction, the primary expectation is coupling (accuracy damage => entropy "
        "rise): exponent flips that flip an answer also spike entropy, mantissa flips are absorbed.",
        "",
        "ARTIFACT GUARD (added after run 1): exponent flips can drive an fp16 value to NaN/Inf. A "
        "non-finite KV page makes the forward's logits all-NaN, and token_entropy(all-NaN) returns "
        "~0 -> the metric MISREADS a garbage cache as maximally confident (a false confident-wrong). "
        "A non-finite cache is also caught by the cheapest content-agnostic check (finiteness) and "
        "leaves 'confidence' undefined, so it is NOT a blind confident-wrong case. The `nonfinite` "
        "column is the fraction of probes whose targeted pages went non-finite; any nonfinite>0 "
        "FORCES blind=False. Run-1's single YES cell (exponent,n5,tk3,K) was exactly this artifact "
        "(verified: 1/3 pages non-finite -> 151936/151936 logits NaN -> entropy=-0.0) and is now "
        "correctly excluded.",
        "",
        f"Cells with any non-finite-cache probe (confidence/finiteness trivially detects): {n_nonfin} of {len(rows)}.",
        "",
        f"CW2_BITFLIP: confident_wrong_cells={n_cw} of {len(rows)}",
        f"CW2_BITFLIP_VERDICT: {verdict}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return verdict, n_cw


def run_cw2_needle(model, tok, device, dtype, *, seed: int = 4242):
    """10.2b: corrupt ONLY the needle token rows (across every layer-page) with each
    corruption, on a long-context passage. Returns (clean_acc, clean_ent, bar, calib, rows).
    Row: (corruption, dacc, dentropy, blind, flag_rate, nonfinite_rate, confident_wrong)."""
    import torch
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages

    probes = []
    for _fact, q, val in NEEDLE_FACTS:
        prompt = NEEDLE_PASSAGE + f"Question: {q} Answer:"
        rows_ = _locate_needle_rows(tok, prompt, val)
        if rows_ is None:
            raise RuntimeError(f"needle {val!r} not locatable — verify-before-code failed")
        probes.append({"prompt": prompt, "expected": val, "rows": rows_})
    tlen = len(tok(probes[0]["prompt"]).input_ids)

    # clean pass: accuracy, entropy, calibration pages
    cc, ce, cal = 0, [], []
    for pr in probes:
        ids, pkv = _prefix_pkv(model, tok, device, pr["prompt"])
        pg = dynamiccache_to_pages(pkv); _inject_pages(pkv, pg, dtype, device)
        t, e = _decode_under_cache(model, tok, ids, pkv, device)
        cc += int(normalized_match(t, pr["expected"], pr.get("alternatives")))
        ce.append(e); cal.extend(pg)
    clean_acc = cc / len(probes); calib = calibrate(cal)
    bar = float(np.std(ce)) if len(ce) > 1 else 0.1; clean_ent = float(np.mean(ce))

    rows = []
    for cname, cfn in _needle_corruptions().items():
        corr, des, fh, nonfin = 0, [], 0, 0
        for pr in probes:
            ids, pkv = _prefix_pkv(model, tok, device, pr["prompt"])
            pg = dynamiccache_to_pages(pkv)
            npg, fl, nf = [], False, False
            for p in pg:  # corrupt needle rows in EVERY layer-page
                c = _apply_to_rows(p, pr["rows"], cfn, seed + p.layer)
                fl = fl or any_physics_flag(p, c, calib)
                if not (np.isfinite(c.K).all() and np.isfinite(c.V).all()):
                    nf = True
                npg.append(c)
            _inject_pages(pkv, npg, dtype, device)
            t, e = _decode_under_cache(model, tok, ids, pkv, device)
            corr += int(normalized_match(t, pr["expected"], pr.get("alternatives")))
            des.append(e); fh += int(fl); nonfin += int(nf)
        dacc = corr / len(probes) - clean_acc
        dent = float(np.mean(des)) - clean_ent
        nonfinite_rate = nonfin / len(probes)
        blind = (dent <= bar) and (nonfinite_rate == 0.0)
        fr = fh / len(probes)
        cw = (dacc <= -DACC_MIN) and blind and (fr >= 0.5)
        rows.append((cname, dacc, dent, blind, fr, nonfinite_rate, cw))
    return clean_acc, clean_ent, bar, tlen, calib, rows


def write_cw2_needle_report(clean_acc, clean_ent, bar, tlen, calib, rows,
                            path="results/REPORT_phase10_cw2_needle.md"):
    import os
    n_cw = sum(1 for r in rows if r[6])
    verdict = "SHOWN" if n_cw else "NOT_SHOWN"
    L = [
        "# REPORT_phase10_cw2_needle.md — Phase 10.2b needle-page confident-wrong test",
        "",
        f"Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Long-context needle: prompt T={tlen} "
        f"tokens; corruption applied to ONLY the needle answer's token ROWS across every "
        f"layer-page (structural correction: pages are per-LAYER over all tokens, so there is no "
        f"single 'needle page'; needle = token rows). {len(rows)} corruptions x "
        f"{len(NEEDLE_FACTS)} planted facts.",
        f"clean_acc={clean_acc:.3f} clean_entropy={clean_ent:.3f} entropy_bar={bar:.3f}. "
        f"Confident-wrong (FIXED): dacc<=-0.25 AND blind AND flag_rate>=0.5; nonfinite>0 forces "
        f"blind=False (artifact guard from 10.2a).",
        "",
        "| corruption | dacc | dentropy | blind | flag_rate | nonfinite | confident_wrong |",
        "|-----------|------|----------|-------|-----------|-----------|-----------------|",
    ]
    for cname, dacc, dent, blind, fr, nf, cw in rows:
        L.append(f"| {cname} | {dacc:+.3f} | {dent:+.3f} | {blind} | {fr:.2f} | {nf:.2f} | "
                 f"{'YES' if cw else 'no'} |")
    L += [
        "",
        "## Interpretation",
        "Tests whether corrupting the token rows that hold a retrieved fact makes the model "
        "confidently hallucinate a plausible substitute (answer wrong, entropy flat). A YES row "
        "locates the error-regime novelty; report as-is either way.",
        "",
        "Honest reading (run A): no confident-wrong cell. bitflip_exp_n1 does break accuracy but "
        "almost entirely by driving the needle rows non-finite (nonfinite~0.88) -> trivially "
        "detectable, confidence undefined (guarded). quant_noise_0.3 does no damage (dacc~0). "
        "v_bias_8.0 damages weakly (below the -0.25 bar) and stays blind. GRANULARITY CAVEAT: the "
        "physics fingerprints are calibrated and evaluated at PAGE level, but a needle is only ~4 "
        "of ~160 token rows, so a coherent few-row perturbation barely moves a page-level scalar "
        "(flag_rate~0 for quant_noise/v_bias). Row-level (per-token) fingerprints would be needed "
        "to detect few-row needle corruption — a real granularity limitation, not a null of "
        "detection in principle.",
        "",
        f"CW2_NEEDLE: confident_wrong_cells={n_cw} of {len(rows)}",
        f"CW2_NEEDLE_VERDICT: {verdict}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return verdict, n_cw


if __name__ == "__main__":
    import sys
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MID = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map="cuda")
    model.eval()
    mode = sys.argv[1] if len(sys.argv) > 1 else "bitflip"
    if mode == "needle":
        ca, cen, bar, tlen, calib, rows = run_cw2_needle(model, tok, "cuda", torch.float16)
        v, n = write_cw2_needle_report(ca, cen, bar, tlen, calib, rows)
        print(f"CW2_NEEDLE: confident_wrong_cells={n} of {len(rows)} -> {v}")
    else:
        ca, cen, bar, calib, rows = run_cw2_bitflip(model, tok, "cuda", torch.float16)
        v, n = write_cw2_bitflip_report(ca, cen, bar, calib, rows)
        print(f"CW2_BITFLIP: confident_wrong_cells={n} of {len(rows)} -> {v}")
