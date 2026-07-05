# REPORT_phase10_cw2_needle.md — Phase 10.2b needle-page confident-wrong test

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Long-context needle: prompt T=171 tokens; corruption applied to ONLY the needle answer's token ROWS across every layer-page (structural correction: pages are per-LAYER over all tokens, so there is no single 'needle page'; needle = token rows). 3 corruptions x 8 planted facts.
clean_acc=0.375 clean_entropy=1.212 entropy_bar=0.295. Confident-wrong (FIXED): dacc<=-0.25 AND blind AND flag_rate>=0.5; nonfinite>0 forces blind=False (artifact guard from 10.2a).

| corruption | dacc | dentropy | blind | flag_rate | nonfinite | confident_wrong |
|-----------|------|----------|-------|-----------|-----------|-----------------|
| bitflip_exp_n1 | -0.250 | -1.152 | False | 1.00 | 0.88 | no |
| quant_noise_0.3 | +0.000 | -0.028 | True | 0.00 | 0.00 | no |
| v_bias_8.0 | -0.125 | -0.016 | True | 0.00 | 0.00 | no |

## Interpretation
Tests whether corrupting the token rows that hold a retrieved fact makes the model confidently hallucinate a plausible substitute (answer wrong, entropy flat). A YES row locates the error-regime novelty; report as-is either way.

Honest reading (run A): no confident-wrong cell. bitflip_exp_n1 does break accuracy but almost entirely by driving the needle rows non-finite (nonfinite~0.88) -> trivially detectable, confidence undefined (guarded). quant_noise_0.3 does no damage (dacc~0). v_bias_8.0 damages weakly (below the -0.25 bar) and stays blind. GRANULARITY CAVEAT: the physics fingerprints are calibrated and evaluated at PAGE level, but a needle is only ~4 of ~160 token rows, so a coherent few-row perturbation barely moves a page-level scalar (flag_rate~0 for quant_noise/v_bias). Row-level (per-token) fingerprints would be needed to detect few-row needle corruption — a real granularity limitation, not a null of detection in principle.

CW2_NEEDLE: confident_wrong_cells=0 of 3
CW2_NEEDLE_VERDICT: NOT_SHOWN
