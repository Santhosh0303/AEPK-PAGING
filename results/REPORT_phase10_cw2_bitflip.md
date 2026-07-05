# REPORT_phase10_cw2_bitflip.md — Phase 10.2a raw-fp16 bit-upset confident-wrong test

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Natural non-ECC DRAM single-event-upset fault model (bitflip_fp16, whole-page). Confidence = real output entropy through the corrupted cache. Confident-wrong cell (FIXED, PREREG): dacc<=-0.25 AND blind AND flag>=0.5.
clean_acc=1.000 clean_entropy=0.382 entropy_bar=0.282.
Calibrated tau: key_norm_mean=602, key_mass=0.239, norm_ratio=104, v_mean_shift=10.4, k_mean_shift=605.

| region | n_flips | tk | tensor | dacc | dentropy | blind | flag_rate | nonfinite | confident_wrong |
|--------|---------|----|--------|------|----------|-------|-----------|-----------|-----------------|
| exponent | 1 | 1 | K | +0.000 | +0.104 | True | 0.62 | 0.00 | no |
| exponent | 1 | 1 | V | +0.000 | +0.004 | True | 0.00 | 0.00 | no |
| exponent | 1 | 3 | K | +0.000 | +0.072 | True | 0.75 | 0.00 | no |
| exponent | 1 | 3 | V | +0.000 | +0.002 | True | 0.00 | 0.00 | no |
| exponent | 3 | 1 | K | -0.125 | +0.345 | False | 0.12 | 0.00 | no |
| exponent | 3 | 1 | V | +0.000 | +0.002 | True | 0.00 | 0.00 | no |
| exponent | 3 | 3 | K | -0.125 | +0.209 | False | 0.88 | 0.12 | no |
| exponent | 3 | 3 | V | -1.000 | +5.022 | False | 0.88 | 0.12 | no |
| exponent | 5 | 1 | K | +0.000 | +0.011 | True | 0.00 | 0.00 | no |
| exponent | 5 | 1 | V | +0.000 | +0.035 | True | 0.00 | 0.00 | no |
| exponent | 5 | 3 | K | -0.875 | -0.380 | False | 1.00 | 0.88 | no |
| exponent | 5 | 3 | V | -1.000 | +5.308 | False | 1.00 | 0.00 | no |
| mantissa | 1 | 1 | K | +0.000 | +0.003 | True | 0.00 | 0.00 | no |
| mantissa | 1 | 1 | V | +0.000 | +0.002 | True | 0.00 | 0.00 | no |
| mantissa | 1 | 3 | K | +0.000 | +0.004 | True | 0.00 | 0.00 | no |
| mantissa | 1 | 3 | V | +0.000 | +0.002 | True | 0.00 | 0.00 | no |
| mantissa | 3 | 1 | K | +0.000 | +0.004 | True | 0.00 | 0.00 | no |
| mantissa | 3 | 1 | V | +0.000 | +0.003 | True | 0.00 | 0.00 | no |
| mantissa | 3 | 3 | K | +0.000 | +0.004 | True | 0.00 | 0.00 | no |
| mantissa | 3 | 3 | V | +0.000 | +0.003 | True | 0.00 | 0.00 | no |
| mantissa | 5 | 1 | K | +0.000 | +0.003 | True | 0.00 | 0.00 | no |
| mantissa | 5 | 1 | V | +0.000 | +0.005 | True | 0.00 | 0.00 | no |
| mantissa | 5 | 3 | K | +0.000 | +0.004 | True | 0.00 | 0.00 | no |
| mantissa | 5 | 3 | V | +0.000 | +0.003 | True | 0.00 | 0.00 | no |
| sign | 1 | 1 | K | +0.000 | -0.020 | True | 0.00 | 0.00 | no |
| sign | 1 | 1 | V | +0.000 | +0.001 | True | 0.00 | 0.00 | no |
| sign | 1 | 3 | K | +0.000 | -0.022 | True | 0.00 | 0.00 | no |
| sign | 1 | 3 | V | +0.000 | -0.001 | True | 0.00 | 0.00 | no |
| sign | 3 | 1 | K | +0.000 | +0.000 | True | 0.00 | 0.00 | no |
| sign | 3 | 1 | V | +0.000 | -0.002 | True | 0.00 | 0.00 | no |
| sign | 3 | 3 | K | +0.000 | +0.073 | True | 0.00 | 0.00 | no |
| sign | 3 | 3 | V | +0.000 | -0.010 | True | 0.00 | 0.00 | no |
| sign | 5 | 1 | K | +0.000 | -0.004 | True | 0.00 | 0.00 | no |
| sign | 5 | 1 | V | +0.000 | +0.025 | True | 0.00 | 0.00 | no |
| sign | 5 | 3 | K | +0.000 | -0.042 | True | 0.00 | 0.00 | no |
| sign | 5 | 3 | V | +0.000 | +0.044 | True | 0.00 | 0.00 | no |

## Interpretation
Per PREREG prediction, the primary expectation is coupling (accuracy damage => entropy rise): exponent flips that flip an answer also spike entropy, mantissa flips are absorbed.

ARTIFACT GUARD (added after run 1): exponent flips can drive an fp16 value to NaN/Inf. A non-finite KV page makes the forward's logits all-NaN, and token_entropy(all-NaN) returns ~0 -> the metric MISREADS a garbage cache as maximally confident (a false confident-wrong). A non-finite cache is also caught by the cheapest content-agnostic check (finiteness) and leaves 'confidence' undefined, so it is NOT a blind confident-wrong case. The `nonfinite` column is the fraction of probes whose targeted pages went non-finite; any nonfinite>0 FORCES blind=False. Run-1's single YES cell (exponent,n5,tk3,K) was exactly this artifact (verified: 1/3 pages non-finite -> 151936/151936 logits NaN -> entropy=-0.0) and is now correctly excluded.

Cells with any non-finite-cache probe (confidence/finiteness trivially detects): 3 of 36.

CW2_BITFLIP: confident_wrong_cells=0 of 36
CW2_BITFLIP_VERDICT: NOT_SHOWN
