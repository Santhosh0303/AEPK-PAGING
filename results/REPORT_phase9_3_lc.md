# REPORT_phase9_3_lc.md — Phase 9.3-LC Long-Context Ablation

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Probes: 10 long-context (LONG_CONTEXT_PASSAGE prepended)
Token length: min_T=307 max_T=311 (ALL >= 150)
Seeds per cell: 2
B0_lc: 0.8000  (freshly measured; NOT the short-prompt 0.330)

## ⚠ PROVISIONAL — REDUCED GRID (HITL review 2026-07-02): DO NOT cite as findings
This run used N=10 probes, 2 seeds. Retention is quantized to 1/16 — ONE probe
moves any number by 0.0625. Two of the three verdicts are UNDERPOWERED, not real:
- LC_OVERRECOVERY (damage_only=1.0 / recovery_on=0.9375): raw counts are
  16/20 vs 15/20 — a ONE-probe difference. INCONCLUSIVE. Does NOT show "RS
  doesn't help" or "RS hurts."
- ABLATION coding=+0.0000 physics=+0.0000 detect=-0.0000: each Δ FLIPS SIGN
  between noise 0.2 and 0.3 (±0.0625) and averages to 0. This is NOISE, NOT a
  null result. At N=10 there is zero power to detect any brick effect.
Only LC_BASELINE_DOMINANCE (bits/elem = deterministic) is trustworthy:
AEPK 3.14 vs KIVI 5.20 bits at iso-accuracy 0.800 = a real compression win
(but from residency/quant, NOT the self-healing thesis — the ablation cannot
attribute it to coding/physics/detection at this N).
REQUIRED next: re-run on the FULL grid (100 probes, 5 seeds) AND add an ERASURE
regime test (total page loss, where recovery is deterministically necessary),
because quant-noise barely dents long-context task accuracy (damage_only ~0.9-1.0)
so the error regime cannot demonstrate healing value. See PROGRESS Phase 9.3-LC-2.

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

## Stage 9.3d — Fair fight: KIVI + SnapKV on long context

At T=307: KIVI-official compresses 275 tokens to 2-bit (group_size=32).
At T=307: SnapKV-r50 evicts 137 of 275 non-window positions (window=32).
9.3d probes: 80% clean accuracy reference

| method                | accuracy | bits/elem | storage% |
|----------------------|----------|-----------|----------|
| KIVI_fp16_control    | 0.8000   |     16.00 | 1.000    |
| KIVI_2_official      | 0.8000   |      5.20 | 0.325    |
| KIVI_2_small_g4      | 0.7000   |     10.05 | 0.628    |
| SnapKV_r100_control  | 0.6000   |     16.00 | 1.000    |
| SnapKV_r50           | 0.4000   |      8.81 | 0.550    |
| AEPK_B3_LC_noise02   | 0.8000   |      3.14 | 0.196    |

control_ok: True

LC_BASELINE_DOMINANCE: DOMINATES_SOME (vs_kivi=AEPK_WINS vs_snapkv=SNAPKV_NOT_APPLICABLE)
