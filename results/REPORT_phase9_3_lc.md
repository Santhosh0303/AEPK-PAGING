# REPORT_phase9_3_lc.md — Phase 9.3-LC Long-Context Ablation

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Probes: 10 long-context (LONG_CONTEXT_PASSAGE prepended)
Token length: min_T=307 max_T=311 (ALL >= 150)
Seeds per cell: 2
B0_lc: 0.8000  (freshly measured; NOT the short-prompt 0.330)

## Root problem (HITL 2026-07-02)
9.1 and 9.2 used SHORT prompts (T=7-25):
  (a) RS over-recovers: few-token pages have tiny MSE → trivially restored.
  (b) KIVI/SnapKV fall back to fp16 at T<32 → never compress → inert win.
Long-context (T>=150) forces RS to compete against 28 high-noise pages.

## Stage 9.3a — damage_only vs recovery_on on long context

damage_only: quant_noise applied; NO recover_rs_erasure call.
recovery_on: quant_noise applied; recover_rs_erasure(worst-2 by MSE).
noise=0.0: control row — both retentions must equal 1.0 (bit-exact).

| noise | damage_only_ret | ±ci | recovery_on_ret | ±ci |
|-------|----------------|-----|----------------|-----|
| 0.00 | 1.0000 | ±0.0000 | 1.0000 | ±0.0000 |
| 0.20 | 0.8750 | ±0.0000 | 0.9375 | ±0.0980 |
| 0.30 | 1.0000 | ±0.0000 | 0.9375 | ±0.0980 |

## Stage 9.3b — LC_OVERRECOVERY interpretation

LC_OVERRECOVERY: noise=0.3 damage_only=1.0000 recovery_on=0.9375

## Stage 9.3c — Ablation: strip bricks one at a time

Bricks compared at each ablation noise level:
  damage_only  : RS OFF, no page selection.
  ro_mse       : RS ON, recover worst-2 by MSE (AEPK physics proxy).
  ro_uniform   : RS ON, recover 2 random pages (no physics signal).
  ro_detector  : RS ON, recover 2 highest-deviation (Phase 4 detector).

Δ coding = ro_mse - damage_only   (RS ON vs OFF; positive = RS helps).
Δ physics = ro_mse - ro_uniform    (MSE-guided vs random; positive = MSE helps).
Δ detect  = ro_detector - ro_mse   (detector vs MSE; positive = detector helps).

| noise | do_ret | ro_mse | ro_uni | ro_det | Δcoding | Δphysics | Δdetect |
|-------|--------|--------|--------|--------|---------|----------|---------|
| 0.20 | 0.8750 | 0.9375 | 0.8750 | 0.8750 | +0.0625 | +0.0625 | -0.0625 |
| 0.30 | 1.0000 | 0.9375 | 1.0000 | 1.0000 | -0.0625 | -0.0625 | +0.0625 |

Ablation levels: [0.2, 0.3]

ABLATION: coding=+0.0000 physics=+0.0000 detect=-0.0000
