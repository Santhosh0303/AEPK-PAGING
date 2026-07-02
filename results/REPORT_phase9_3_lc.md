# REPORT_phase9_3_lc.md — Phase 9.3-LC Long-Context Ablation

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Probes: 100 long-context (LONG_CONTEXT_PASSAGE prepended)
Token length: min_T=305 max_T=330 (ALL >= 150)
Seeds per cell: 5
B0_lc: 0.3800  (freshly measured; NOT the short-prompt 0.330)

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
| 0.20 | 0.9895 | ±0.0100 | 0.9895 | ±0.0078 |
| 0.30 | 0.9526 | ±0.0169 | 0.9684 | ±0.0144 |

## Stage 9.3b — LC_OVERRECOVERY interpretation

LC_OVERRECOVERY: noise=0.3 damage_only=0.9526 recovery_on=0.9684

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
| 0.20 | 0.9895 | 0.9895 | 0.9895 | 0.9789 | +0.0000 | +0.0000 | -0.0105 |
| 0.30 | 0.9526 | 0.9684 | 0.9474 | 0.9895 | +0.0158 | +0.0211 | +0.0211 |

Ablation levels: [0.2, 0.3]

ABLATION: coding=+0.0079 physics=+0.0105 detect=+0.0053

## Stage 9.3d — Fair fight: KIVI + SnapKV on long context

At T=307: KIVI-official compresses 275 tokens to 2-bit (group_size=32).
At T=307: SnapKV-r50 evicts 137 of 275 non-window positions (window=32).
9.3d probes: 38% clean accuracy reference

| method                | accuracy | bits/elem | storage% |
|----------------------|----------|-----------|----------|
| KIVI_fp16_control    | 0.3800   |     16.00 | 1.000    |
| KIVI_2_official      | 0.3800   |      5.10 | 0.319    |
| KIVI_2_small_g4      | 0.3600   |     10.03 | 0.627    |
| SnapKV_r100_control  | 0.3300   |     16.00 | 1.000    |
| SnapKV_r50           | 0.2000   |      8.80 | 0.550    |
| AEPK_B3_LC_noise02   | 0.3700   |      3.14 | 0.196    |

control_ok: True

LC_BASELINE_DOMINANCE: DOMINATES_SOME (vs_kivi=AEPK_WINS vs_snapkv=SNAPKV_NOT_APPLICABLE)
