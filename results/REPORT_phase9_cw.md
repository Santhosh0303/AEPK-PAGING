# REPORT_phase9_cw.md — Phase 9-CW confident-wrong error-regime test

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Probes: 8 short factual. clean_acc=1.000 clean_entropy=0.382 nats. entropy_bar(confident)=0.282.
Physics fingerprints (correct, flatten-aware; detect.py is degenerate on 3D pages — see phase9_cw docstring FLAW A/B). Calibrated tau (FPR-controlled): key_norm_mean=602, key_mass=0.239, norm_ratio=104, v_mean_shift=10.4, k_mean_shift=605.

A CONFIDENT-WRONG cell needs ALL THREE: dacc<=-0.25 (accuracy broken), blind=True (corrupt entropy within entropy_bar of clean -> logprob would NOT flag), flag_rate>=0.5 (calibrated physics DOES catch it).

| kind | mag | tk | dacc | dentropy | blind | flag_rate | confident_wrong |
|------|-----|----|------|----------|-------|-----------|-----------------|
| k_scale | 0.70 | 1 | +0.000 | -0.094 | True | 0.00 | no |
| k_scale | 0.70 | 3 | -0.375 | +4.877 | False | 0.00 | no |
| k_scale | 0.85 | 1 | +0.000 | -0.076 | True | 0.00 | no |
| k_scale | 0.85 | 3 | -0.875 | +8.844 | False | 0.00 | no |
| k_scale | 1.15 | 1 | -1.000 | +2.559 | False | 0.00 | no |
| k_scale | 1.15 | 3 | -1.000 | +2.183 | False | 0.00 | no |
| k_scale | 1.30 | 1 | -1.000 | +2.658 | False | 0.00 | no |
| k_scale | 1.30 | 3 | -1.000 | +1.980 | False | 0.00 | no |
| k_scale | 1.60 | 1 | -1.000 | +2.532 | False | 1.00 | no |
| k_scale | 1.60 | 3 | -1.000 | +1.967 | False | 1.00 | no |
| v_scale | 0.30 | 1 | +0.000 | +0.320 | False | 1.00 | no |
| v_scale | 0.30 | 3 | -0.125 | +0.165 | True | 1.00 | no |
| v_scale | 3.00 | 1 | -0.125 | +0.426 | False | 1.00 | no |
| v_scale | 3.00 | 3 | -0.250 | +1.163 | False | 1.00 | no |
| v_bias | 4.00 | 1 | +0.000 | -0.074 | True | 0.00 | no |
| v_bias | 4.00 | 3 | +0.000 | -0.178 | True | 0.00 | no |
| v_bias | 8.00 | 1 | -0.125 | +0.062 | True | 0.00 | no |
| v_bias | 8.00 | 3 | -0.125 | +0.322 | False | 0.00 | no |
| v_bias | 16.00 | 1 | -0.875 | +2.203 | False | 1.00 | no |
| v_bias | 16.00 | 3 | -0.250 | +2.142 | False | 1.00 | no |

## Interpretation
Every cell that BREAKS accuracy (dacc<=-0.25) also RAISES output entropy (dentropy>0, blind=False): the model becomes visibly uncertain, so its own logprob/confidence is an effective corruption detector. Cells that stay confident (blind=True) do NOT break accuracy. Accuracy damage and confidence loss are COUPLED for structured KV corruption on this model.

CONFIDENT_WRONG_NOVELTY: NOT_SHOWN

Honest reading: the confident-wrong blind spot — the premise motivating content-agnostic physics detection — is NOT demonstrated for KV corruption here. Calibrated physics fingerprints DO fire on the larger corruptions, but only on ones logprob already catches (entropy up), so they add no unique value in the error regime. Caveat: a gradient-optimized adversary purpose-built to flip the answer while minimizing output entropy was NOT tested; that is not a natural cache fault. Surviving honest contributions: compression (non-novel) + erasure resilience (non-novel).
