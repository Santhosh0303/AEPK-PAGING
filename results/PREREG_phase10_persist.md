# PREREG_phase10_persist.md — Phase 10 step 22 persisted-cache store/heal demo

status: PRE-REGISTERED (written BEFORE the GPU run)
created: 2026-07-06T16:18:47Z (UTC)
harness: aepk_paging/harness/phase10_persist.py · tests: tests/test_phase10_persist.py
new files only; zero edits to Phase 2-5 source (honesty spine S9).

## Purpose
Close the persisted-cache deployment objection AND the deploy-caveat loop left open in
REPORT_phase10_liveheal.md: show detection working against STORED fingerprint SCALARS (a few
floats per page recorded at encode time), not against a clean in-memory page the deployed system
would not have.

## Flow (FIXED before the run)
Model Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Live-heal config GROUP_SIZE=4, NUM_PARITY=1.
Per clean-correct probe (n_cc >= 20 target, cap 24):
1. prefill -> pages; the erasure group = top-GROUP_SIZE pages by fp_key_norm_mean; target =
   top-influence page (group index 0).
2. encode RS parity over the CLEAN group (redundancy = parity only); record the target's physics
   fingerprint SCALARS (all 5 FINGERPRINTS) + the calibrated per-fingerprint thresholds tau
   (calibrate over all clean pages, sigma_mult=3.0).
3. serialize pages + parity + stored scalars + tau to disk (np.savez, deterministic).
4. corrupt ONE stored page's K bytes on disk: target K *= k_scale=2.0 (the proven-harmful
   structured fault; deterministic, no RNG). Siblings + parity untouched.
5. restore from disk; DETECT the fault with the DEPLOYABLE detector: recompute the restored
   page's fingerprints and flag iff any deviates from the STORED scalar beyond tau. No clean page
   in memory.
6. if flagged, erase the page and recover it BIT-EXACT from parity + surviving stored pages
   (recover_rs_erasure); verify healed page byte-identical to the pre-save original. If the
   detector misses, the page stays unhealed (fail-loud) — that miss is the finding.
7. accuracy arms: baseline (inject corrupted target, no heal) vs healed (detect->erasure-heal),
   decode + normalized_match.
Round-trip CONTROL: save/restore with NO corruption must be byte-identical end-to-end.

## Verdict line (runtime f-string)
`PERSIST_HEAL: roundtrip_exact=<bool> detected=<bool> healed_exact=<bool> baseline_acc=<a>
healed_acc=<b>` where detected = all probes' faults flagged by the stored-scalar detector
(detection_rate=1.0), healed_exact = every recovered page byte-identical to its original.

## Determinism gate
The whole demo runs TWICE. Accuracy rows (baseline_acc, healed_acc), the byte-exact flags
(roundtrip_exact, healed_exact) and the detection outcome are EXACT-MATCH across runs (clean
prefill is deterministic; k_scale is deterministic; no RNG). Byte-level heal is exact by RS
construction.

## ACCEPT
- PERSIST_HEAL runtime line present; healed page byte-identical to original (when detected);
  round-trip control exact; x2 runs identical on accuracy/bytes rows.
- ALLOWED-to-FAIL: if the stored-scalar detector MISSES the k_scale fault (detection_rate < 1),
  that is the deploy-caveat answer, reported as-is (healed_acc then reflects the unhealed pages);
  the demo is still valid — the honest negative is the result.
- CPU tests (serialize round-trip, corruption determinism, stored-scalar detector, bit-exact
  heal, line-exists) green; zero Phase 2-5 diff.
