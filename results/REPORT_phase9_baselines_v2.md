# REPORT_phase9_baselines_v2.md — Phase 9.2 ISO-ACCURACY Baseline Comparison

Pre-registration: results/PREREG_phase9_baselines.md (commit f1b529e)
Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Eval set: 100 probes (30 EVAL_PROBES + 70 allenai/sciq@validation rows 0-69)
ISO-ACCURACY reference: AEPK B3 at noise=0.2, acc=0.324±0.010 (Phase 9.1, 5 seeds)
AEPK accuracy labeled: recovery-on, uninterpreted pending Phase 9.3

## UNVERIFIED methods (excluded from dominance)
UNVERIFIED: KVQuant — NUQ calibration pipeline required; pre-RoPE hooks
unavailable for Qwen2.5 + transformers 5.12.1. Excluded per BUILD_SPEC 9.2.

## No-damage control results
KIVI-fp16-control accuracy: 0.330 (B0_sdpa=0.330)
  OK: True (threshold: |KIVI_fp16 - B0_sdpa| <= 0.01)
B0_eager accuracy: 0.290 (B0_sdpa=0.330)
  Eager≈SDPA: True (threshold: |B0_eager - B0_sdpa| <= 0.05)
All controls passed: True

## ISO-ACCURACY comparison table

| Method | Accuracy | bits/elem | storage% of fp16 | Notes |
|--------|----------|-----------|-------------------|-------|
| B0_sdpa (fp16 ref) | 0.330 | 16.00 | 100.0% | clean, no compression |
| B0_eager | 0.290 | 16.00 | 100.0% | eager-attn clean |
| AEPK_B3_noise=0.2 | 0.310 | 3.81 | 23.8% | recovery-on; uninterpreted pending 9.3 |
| KIVI-fp16-ctrl | 0.330 | 16.00 | 100.0% | no-damage control |
| KIVI-2-official (g32,r32) | 0.330 | 16.00 | 100.0% | short-prompt: T<32 falls back to fp16 |
| KIVI-2-small (g4,r0) | 0.320 | 10.43 | 65.2% | small-group config; compresses short prompts |
| KIVI-4-official (g32,r32) | 0.330 | 16.00 | 100.0% | 4-bit; short-prompt fallback |
| SnapKV-r100-ctrl | 0.290 | 16.00 | 100.0% | no-op control (eager) |
| SnapKV-r75 | 0.290 | 15.96 | 99.7% | keep 75%; short prompts: T≤window |
| SnapKV-r50 | 0.300 | 15.95 | 99.7% | keep 50%; short prompts: T≤window |
| SnapKV-r25 | 0.300 | 15.93 | 99.5% | keep 25%; short prompts: T≤window |

## Short-prompt regime note
Our 100-probe eval set uses short prompts (typical T=7-25 tokens).
KIVI-official (group_size=32) requires T>=32 for K quantization; short prompts
fall back to fp16 (no compression). SnapKV (window_size=32) requires T>window_size
for eviction; short prompts keep all positions (no eviction).
AEPK achieves storage savings through LAYER-LEVEL eviction regardless of T.
This is an honest regime difference: KIVI/SnapKV are designed for long-context.

## ISO-ACCURACY analysis
AEPK reference: accuracy=0.310, bits/elem=3.81
At accuracy≈0.310:
  KIVI competitors: ['KIVI_2_official(0.330,16.00)', 'KIVI_2_small_g4(0.320,10.43)', 'KIVI_4_official(0.330,16.00)']
  SnapKV competitors: ['SnapKV_r75(0.290,15.96)', 'SnapKV_r50(0.300,15.95)', 'SnapKV_r25(0.300,15.93)']

## Per-method verdicts
AEPK_vs_KIVI: AEPK_WINS
AEPK_vs_SNAPKV: AEPK_WINS

BASELINE_DOMINANCE: DOMINATES_ALL
