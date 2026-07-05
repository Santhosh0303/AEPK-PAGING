# REPORT_phase10_liveheal.md — Phase 10.3 / 9.6 MOVE A live mid-generation self-heal

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Fault corrupts the top-influence resident KV page (K) mid-generation; physics fingerprint localizes it; the page is erased and restored bit-exact from sibling-page RS parity (recover_rs_erasure); generation continues, zero recompute. Redundancy = parity only (Cauchy-MDS group of 4 pages, num_parity=1).
clean_acc=1.000.

| k_factor | baseline_acc | aepk_acc | flagged_rate | recovered | decode_mode |
|----------|--------------|----------|--------------|-----------|-------------|
| 1.0 (CONTROL) | 1.000 | 1.000 | 0.00 | True | erasure |
| 2.0 | 0.000 | 1.000 | 1.00 | True | erasure |
| 4.0 | 0.000 | 1.000 | 1.00 | True | erasure |

## Interpretation
The k_factor=1.0 CONTROL row MUST show baseline_acc == aepk_acc == clean_acc (no fault, no heal) — it is the plumbing check (9.1 pattern). At damaging magnitudes, baseline_acc drops (fault bit the answer) while aepk_acc returns to clean_acc via bit-exact erasure recovery of the physics-located page. If the detector misses the page it stays blind (unhealed) and recovered=False — reported as-is.

LIVE_HEAL: baseline_acc=0.000 aepk_acc=1.000 recovered=True decode_mode=erasure

## Control arm (PREREG v2) — is top-influence selection load-bearing?
Same k_scale factors applied to the LOWEST-fp_key_norm_mean page (other end of the influence ranking); baseline_acc only, no heal arm. If the low-mass arm also collapses, the top-influence proxy is NOT load-bearing (finding, reported as-is); if it stays near clean (within one probe of clean_acc), selection is validated.

| k_factor | low_baseline_acc |
|----------|------------------|
| 2.0 | 1.000 |
| 4.0 | 1.000 |

HEAL_CONTROL: top_baseline=0.000 low_baseline=1.000 selection_load_bearing=True

## Deployability caveat (honesty spine)
This harness localizes the fault by comparing the corrupt page against the CLEAN in-memory page (`any_physics_flag(pg[tgt_i], corrupt_tgt, calib)`). A deployable detector does NOT have the clean page in memory — that is exactly what it is trying to recover. In deployment, detection compares the live page's fingerprints against STORED per-page fingerprint SCALARS (key_norm_mean, key_mass, norm_ratio, v/k_mean_shift) recorded at encode time — a few floats per page, not a retained clean copy. The localization capability is the same (9.3c-localization: FPR-0 headroom vs the fp16 round-trip floor); only the reference is a compact stored fingerprint, not a clean page.
