"""Phase 10 step 22 — persisted-cache store demo (deployment polish).

End-to-end reuse story: prefill -> pages -> encode RS parity + record physics fingerprint SCALARS
-> serialize pages+parity+scalars to disk -> corrupt ONE stored page's K bytes on disk
(structured k_scale=2.0, the proven-harmful fault) -> restore -> DETECT the fault against the
STORED SCALARS (the deployable comparison — no clean page in memory, closing the deploy-caveat
loop from REPORT_phase10_liveheal.md) -> erasure-heal the page bit-exact from parity + surviving
stored pages -> verify healed page BYTE-IDENTICAL to the pre-save original. Accuracy arms: baseline
(corrupted, no heal) vs healed, on clean-correct probes. Round-trip control: save/restore with NO
corruption is byte-identical end-to-end.

Honesty spine S9: zero edits to Phase 2-5 source. New harness file. Reuses coding.encode_rs_erasure_group
/ recover_rs_erasure, phase9_cw FINGERPRINTS/calibrate/fp_key_norm_mean/corrupt_k_scale/_decode_under_cache,
phase10_liveheal GROUP_SIZE/NUM_PARITY, dynamiccache_to_pages, _inject_pages, normalized_match.
Serialize is deterministic (np.savez). Runtime f-string verdict. ALLOWED-to-FAIL: a stored-scalar
detector that MISSES the fault is itself the deploy-caveat answer, reported as-is.
"""

from __future__ import annotations

import os

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.coding import (
    encode_rs_erasure_group, recover_rs_erasure, CauchyReedSolomonGroup, UncorrectableError,
)
from aepk_paging.harness.phase9_cw import FINGERPRINTS, corrupt_k_scale
from aepk_paging.harness.phase10_liveheal import GROUP_SIZE, NUM_PARITY

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
KSCALE = 2.0                    # proven-harmful structured fault (phase 9.6 / liveheal)
N_PROBES = 24                   # clean-correct probes for the accuracy arms (>= 20)


# ---- deterministic serialize / deserialize (CPU-testable, no model) ---------

def fingerprint_scalars(page) -> dict[str, float]:
    """The deployable stored scalars: every physics fingerprint of a page (a few floats)."""
    return {name: float(fp(page)) for name, fp in FINGERPRINTS.items()}


def save_group(path, pages, group: CauchyReedSolomonGroup, stored_fps, tau, target_idx: int):
    """Serialize the group's data pages + parity + stored fingerprint scalars + detector thresholds
    to a single .npz. Deterministic. page_ids are re-assigned to their group index on load, so the
    group is self-contained (no complex page_id serialization)."""
    d: dict = {}
    d["num_pages"] = np.int64(len(pages))
    d["k_byte_len"] = np.int64(group.k_byte_len)
    d["num_parity"] = np.int64(group.num_parity)
    d["target_idx"] = np.int64(target_idx)
    d["parity_bytes"] = np.ascontiguousarray(group.parity_bytes)
    d["layers"] = np.array([int(p.layer) for p in pages], dtype=np.int64)
    d["token_ranges"] = np.array([list(p.token_range) for p in pages], dtype=np.int64)
    d["attention_mass"] = np.array([float(p.attention_mass) for p in pages], dtype=np.float64)
    d["precision_tags"] = np.array([str(p.precision_tag) for p in pages])
    d["fp_names"] = np.array(list(stored_fps.keys()))
    d["fp_values"] = np.array([stored_fps[n] for n in stored_fps], dtype=np.float64)
    d["tau_values"] = np.array([tau[n] for n in stored_fps], dtype=np.float64)
    for i, p in enumerate(pages):
        d[f"K_{i}"] = np.ascontiguousarray(p.K)
        d[f"V_{i}"] = np.ascontiguousarray(p.V)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, **d)


def load_group(path):
    """Return (pages, group, stored_fps, tau, target_idx). page_id == group index."""
    z = np.load(path, allow_pickle=False)
    n = int(z["num_pages"])
    pages = []
    for i in range(n):
        pages.append(KVPage(
            page_id=i, layer=int(z["layers"][i]),
            token_range=tuple(int(x) for x in z["token_ranges"][i]),
            K=np.array(z[f"K_{i}"]), V=np.array(z[f"V_{i}"]),
            precision_tag=str(z["precision_tags"][i]),
            attention_mass=float(z["attention_mass"][i])))
    group = CauchyReedSolomonGroup(
        pages=tuple(pages), parity_bytes=np.array(z["parity_bytes"]),
        k_byte_len=int(z["k_byte_len"]), page_ids=tuple(range(n)),
        num_parity=int(z["num_parity"]))
    names = [str(x) for x in z["fp_names"]]
    stored_fps = {nm: float(v) for nm, v in zip(names, z["fp_values"])}
    tau = {nm: float(v) for nm, v in zip(names, z["tau_values"])}
    return pages, group, stored_fps, tau, int(z["target_idx"])


def corrupt_stored_page_k(path, target_idx: int, k_scale: float):
    """Corrupt ONE stored page's K bytes on disk: load, scale target K by k_scale, rewrite the
    .npz. Deterministic (structured scale, no RNG). Other stored pages + parity untouched."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d[f"K_{target_idx}"] = np.ascontiguousarray(
        (np.asarray(d[f"K_{target_idx}"], dtype=np.float32) * np.float32(k_scale)))
    with open(path, "wb") as f:
        np.savez(f, **d)


def stored_scalar_flag(page, stored_fps, tau) -> bool:
    """Deployable detector: recompute the fingerprints on `page` and flag iff any deviates from the
    STORED scalar beyond its calibrated threshold. NO clean page in memory — only stored floats.
    Returns True if the page is flagged as corrupted."""
    for name, fp in FINGERPRINTS.items():
        if abs(float(fp(page)) - stored_fps[name]) > tau[name]:
            return True
    return False


def pages_byte_identical(a, b) -> bool:
    return bool(np.array_equal(a.K, b.K) and np.array_equal(a.V, b.V))


# ---- report -----------------------------------------------------------------

def write_persist_report(res, path="results/REPORT_phase10_persist.md"):
    L = [
        "# REPORT_phase10_persist.md — Phase 10 step 22 persisted-cache store/heal demo",
        "",
        f"Model: {MODEL_ID} fp16 (CUDA). Live-heal config GROUP_SIZE={GROUP_SIZE}, "
        f"NUM_PARITY={NUM_PARITY}. Flow per probe: prefill -> pages -> encode RS parity + record "
        "physics fingerprint SCALARS -> np.savez to disk -> corrupt ONE stored page's K on disk "
        f"(k_scale={KSCALE}) -> restore -> DETECT against STORED scalars (deployable: no clean "
        "page in memory) -> erasure-heal bit-exact from parity + survivors -> verify healed page "
        f"byte-identical to the pre-save original. Accuracy arms on n_cc={res['n_cc']} clean-correct "
        "probes. Round-trip control: save/restore with NO corruption, byte-identical end-to-end.",
        "",
        "| quantity | value |",
        "|----------|-------|",
        f"| n_clean_correct | {res['n_cc']} |",
        f"| roundtrip_exact (no-corruption control byte-identical) | {res['roundtrip_exact']} |",
        f"| detected (stored-scalar detector flagged the fault, all probes) | {res['detected']} |",
        f"| detection_rate | {res['detection_rate']:.3f} |",
        f"| healed_exact (recovered page byte-identical to original) | {res['healed_exact']} |",
        f"| baseline_acc (corrupted, no heal) | {res['baseline_acc']:.3f} |",
        f"| healed_acc (detect->erasure-heal) | {res['healed_acc']:.3f} |",
        f"| clean_acc | {res['clean_acc']:.3f} |",
        "",
        "## Interpretation",
        "The round-trip control MUST be byte-identical (save/restore is lossless) — the plumbing "
        "check. Under the k_scale fault, the DEPLOYABLE detector compares the restored page's "
        "recomputed fingerprints against the STORED scalars (a few floats per page recorded at "
        "encode time), not a clean copy it does not have. When flagged, the page is erased and "
        "recovered bit-exact from RS parity + the surviving stored pages, so healed_acc returns to "
        "clean_acc while baseline_acc (keep the corruption) drops. If the stored-scalar detector "
        "MISSES the fault (detection_rate < 1), that miss is the deploy-caveat finding, reported "
        "as-is — the page then stays unhealed and healed_acc reflects it.",
        "",
        f"PERSIST_HEAL: roundtrip_exact={res['roundtrip_exact']} detected={res['detected']} "
        f"healed_exact={res['healed_exact']} baseline_acc={res['baseline_acc']:.3f} "
        f"healed_acc={res['healed_acc']:.3f}",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return res


# ---- GPU flow ---------------------------------------------------------------

def run_persist(model, tok, device, dtype, *, probes, work_dir, n_probes=N_PROBES,
                kscale=KSCALE):
    """Full save/corrupt/restore/detect/heal cycle per clean-correct probe. Returns a result dict
    with the PERSIST_HEAL fields. Deterministic -> byte-identical accuracy/flag rows across runs."""
    import torch
    from aepk_paging.harness.phase9_cw import _decode_under_cache, fp_key_norm_mean, calibrate
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages

    def prefix(prompt):
        ids = tok(prompt, return_tensors="pt").to(device).input_ids
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=True)
        return ids, out.past_key_values

    # clean pass: clean-correct subset + calibration over all clean pages
    clean_correct, cal_pages = [], []
    for pr in probes:
        ids, pkv = prefix(pr["prompt"])
        pg = dynamiccache_to_pages(pkv)
        _inject_pages(pkv, pg, dtype, device)
        t, _ = _decode_under_cache(model, tok, ids, pkv, device)
        cal_pages.extend(pg)
        if normalized_match(t, pr["expected"], pr.get("alternatives")):
            clean_correct.append(pr)
        if len(clean_correct) >= n_probes:
            break
    clean_acc_n = len(clean_correct)
    calib = calibrate(cal_pages)
    tau = calib.tau

    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, "persist_group.npz")
    ctrl_path = os.path.join(work_dir, "persist_ctrl.npz")

    roundtrip_all = True
    detected_count = 0
    healed_exact_all = True
    base_ok = 0
    heal_ok = 0
    n = clean_correct and len(clean_correct) or 0

    for pr in clean_correct:
        ids, pkv = prefix(pr["prompt"])
        pg = dynamiccache_to_pages(pkv)
        order = sorted(range(len(pg)), key=lambda i: -fp_key_norm_mean(pg[i]))
        grp_idx = order[:GROUP_SIZE]
        target_local = 0                              # top-influence page = group index 0
        clean_group = [pg[i] for i in grp_idx]
        original_target = clean_group[target_local]
        group = encode_rs_erasure_group(clean_group, NUM_PARITY)
        stored_fps = fingerprint_scalars(original_target)

        # ---- round-trip control (no corruption) ----
        save_group(ctrl_path, clean_group, group, stored_fps, tau, target_local)
        cpages, _, _, _, _ = load_group(ctrl_path)
        roundtrip_all = roundtrip_all and all(
            pages_byte_identical(a, b) for a, b in zip(clean_group, cpages))

        # ---- save -> corrupt on disk -> restore ----
        save_group(path, clean_group, group, stored_fps, tau, target_local)
        corrupt_stored_page_k(path, target_local, kscale)
        lpages, lgroup, lfps, ltau, tidx = load_group(path)
        corrupt_target = lpages[tidx]

        # ---- DETECT against STORED scalars (deployable) ----
        detected = stored_scalar_flag(corrupt_target, lfps, ltau)
        detected_count += int(detected)

        # ---- HEAL (erasure recovery) if detected ----
        heal_pages = [pg[i] for i in range(len(pg))]      # full page list for injection
        base_pages = [pg[i] for i in range(len(pg))]
        base_pages[grp_idx[target_local]] = corrupt_target       # baseline keeps the corruption
        if detected:
            try:
                rec = recover_rs_erasure(lgroup, [tidx])
                healed = rec[tidx]
                healed_exact_all = healed_exact_all and pages_byte_identical(healed, original_target)
                heal_pages[grp_idx[target_local]] = healed
            except UncorrectableError:
                heal_pages[grp_idx[target_local]] = corrupt_target    # fail-loud
                healed_exact_all = False
        else:
            heal_pages[grp_idx[target_local]] = corrupt_target        # miss -> unhealed

        # ---- accuracy arms ----
        ids_b, pkv_b = prefix(pr["prompt"])
        _inject_pages(pkv_b, base_pages, dtype, device)
        tb, _ = _decode_under_cache(model, tok, ids_b, pkv_b, device)
        base_ok += int(normalized_match(tb, pr["expected"], pr.get("alternatives")))

        ids_h, pkv_h = prefix(pr["prompt"])
        _inject_pages(pkv_h, heal_pages, dtype, device)
        th, _ = _decode_under_cache(model, tok, ids_h, pkv_h, device)
        heal_ok += int(normalized_match(th, pr["expected"], pr.get("alternatives")))

    det_rate = detected_count / n if n else float("nan")
    return {
        "n_cc": clean_acc_n,
        "clean_acc": 1.0,                            # clean-correct subset by construction
        "roundtrip_exact": bool(roundtrip_all),
        "detected": bool(detected_count == n and n > 0),
        "detection_rate": float(det_rate),
        "healed_exact": bool(healed_exact_all),
        "baseline_acc": base_ok / n if n else float("nan"),
        "healed_acc": heal_ok / n if n else float("nan"),
    }
