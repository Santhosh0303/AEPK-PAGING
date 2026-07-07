# REPORT_phase10_persist.md — Phase 10 step 22 persisted-cache store/heal demo

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Live-heal config GROUP_SIZE=4, NUM_PARITY=1. Flow per probe: prefill -> pages -> encode RS parity + record physics fingerprint SCALARS -> np.savez to disk -> corrupt ONE stored page's K on disk (k_scale=2.0) -> restore -> DETECT against STORED scalars (deployable: no clean page in memory) -> erasure-heal bit-exact from parity + survivors -> verify healed page byte-identical to the pre-save original. Accuracy arms on n_cc=24 clean-correct probes. Round-trip control: save/restore with NO corruption, byte-identical end-to-end.

| quantity | value |
|----------|-------|
| n_clean_correct | 24 |
| roundtrip_exact (no-corruption control byte-identical) | True |
| detected (stored-scalar detector flagged the fault, all probes) | True |
| detection_rate | 1.000 |
| healed_exact (recovered page byte-identical to original) | True |
| baseline_acc (corrupted, no heal) | 0.000 |
| healed_acc (detect->erasure-heal) | 1.000 |
| clean_acc | 1.000 |

## Interpretation
The round-trip control MUST be byte-identical (save/restore is lossless) — the plumbing check. Under the k_scale fault, the DEPLOYABLE detector compares the restored page's recomputed fingerprints against the STORED scalars (a few floats per page recorded at encode time), not a clean copy it does not have. When flagged, the page is erased and recovered bit-exact from RS parity + the surviving stored pages, so healed_acc returns to clean_acc while baseline_acc (keep the corruption) drops. If the stored-scalar detector MISSES the fault (detection_rate < 1), that miss is the deploy-caveat finding, reported as-is — the page then stays unhealed and healed_acc reflects it.

PERSIST_HEAL: roundtrip_exact=True detected=True healed_exact=True baseline_acc=0.000 healed_acc=1.000
