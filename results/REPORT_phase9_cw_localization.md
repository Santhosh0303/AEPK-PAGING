# REPORT_phase9_cw_localization.md — detector localization with FIXED detect.py

Re-opens the 9.3c ablation ('detection doesn't help'), which used the degenerate detect.py detector (FLAW A/B, now FLAW-A fixed + FLAW-B documented). recall = fraction of corrupted pages a calibrated fingerprint flags; FPR = fraction of pages a BENIGN fp16 round-trip flags (the real noise floor). sigma_mult = threshold tightness (tau = sigma_mult * clean-page spread).

| sigma_mult | corruption | recall | FPR |
|-----------|------------|--------|-----|
| 3.00 | quant_noise_0.3 | 0.000 | 0.000 |
| 3.00 | quant_noise_0.5 | 0.000 | 0.000 |
| 3.00 | k_scale_1.6 | 0.036 | 0.000 |
| 3.00 | v_bias_8.0 | 0.000 | 0.000 |
| 1.00 | quant_noise_0.3 | 0.036 | 0.000 |
| 1.00 | quant_noise_0.5 | 0.089 | 0.000 |
| 1.00 | k_scale_1.6 | 0.179 | 0.000 |
| 1.00 | v_bias_8.0 | 0.701 | 0.000 |
| 0.25 | quant_noise_0.3 | 0.219 | 0.000 |
| 0.25 | quant_noise_0.5 | 0.411 | 0.000 |
| 0.25 | k_scale_1.6 | 0.826 | 0.000 |
| 0.25 | v_bias_8.0 | 1.000 | 0.000 |

DETECTOR_LOCALIZATION: at sigma=0.25 (FPR=0 vs fp16 floor) v_bias recall=1.000, quant_noise_0.3 recall=0.219 -> FUNCTIONAL_FOR_STRUCTURED

## Interpretation (honest)
FPR is 0.000 at EVERY tested threshold, including the tightest (sigma=0.25): the benign fp16 round-trip never trips the detector, so there is real headroom and a clean operating point. At sigma=0.25 the FIXED detector LOCALIZES structured corruption cleanly (v_bias recall 1.00, k_scale 0.83) with zero false positives. So 9.3c's 'detection doesn't help' WAS partly the degenerate detector: with FLAW-A fixed and an FPR-safe threshold, content-agnostic detection is FUNCTIONAL for structured corruption. This vindicates the detector as a real capability (a genuine bug was masking it).

TWO honest caveats keep this from reviving the error-regime NOVELTY: (1) quant_noise — the exact corruption 9.3c injected — stays the weakest at every threshold (recall 0.22-0.41 even tight) because it is broadband-subtle; and it is accuracy-benign (9.3c/9-CW). So for that corruption detection genuinely offers little, and the 9.3c null on quant_noise is real. (2) The structured corruptions the detector CAN localize (k_scale, v_bias at damaging magnitudes) also RAISE output entropy (9-CW) -> the model's own logprob already catches them. Net: fixing the detector restores a real, functional detection capability and de-confounds 9.3c, but does NOT establish the error-regime novelty (no regime where physics detection uniquely beats logprob). Remaining open item: a principled FPR-calibrated threshold (detector-guarantee.md's hand-set-tau gap) — now shown to have FPR-0 headroom.
