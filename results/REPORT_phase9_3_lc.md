# REPORT_phase9_3_lc.md — Phase 9.3-LC Long-Context Ablation

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Probes: 10 long-context (LONG_CONTEXT_PASSAGE prepended)
Token length: min_T=307 max_T=311 (ALL >= 150, asserted by tokenizer)
Seeds per cell: 2
B0_lc: 0.8000  (freshly measured; NOT the short-prompt 0.330)

## Root problem (HITL 2026-07-02)
9.1 and 9.2 used SHORT prompts (T=7-25):
  (a) RS over-recovers: few-token pages have tiny MSE → trivially restored.
  (b) KIVI/SnapKV fall back to fp16 at T<32 → never compress → inert win.
Long-context (T>=150) forces RS to compete against 28 high-noise pages
and ensures KIVI/SnapKV actually engage compression.

## Stage 9.3a — damage_only vs recovery_on on long context

damage_only: quant_noise applied; NO recover_rs_erasure call.
recovery_on: quant_noise applied; recover_rs_erasure(worst-2 pages).
noise=0.0: control row — both retentions must equal 1.0 (bit-exact).

| noise | damage_only_ret | ±ci | recovery_on_ret | ±ci |
|-------|----------------|-----|----------------|-----|
| 0.00 | 1.0000 | ±0.0000 | 1.0000 | ±0.0000 |
| 0.20 | 0.8750 | ±0.0000 | 0.9375 | ±0.0980 |
| 0.30 | 1.0000 | ±0.0000 | 0.9375 | ±0.0980 |

## Stage 9.3b — LC_OVERRECOVERY interpretation

9.1 observed retention~1.0 at ALL noise on SHORT prompts (UNINTERPRETED).
On long context (T>=150), damage_only reveals whether accuracy survives
noise WITHOUT RS recovery. Two possible outcomes (both honest):
  damage_only~1.0 → model tolerates noise structurally (RS irrelevant).
  damage_only<1.0 AND recovery_on>damage_only → RS genuinely restores.
  damage_only~recovery_on → RS recovery makes no difference on LC.

LC_OVERRECOVERY: noise=0.3 damage_only=1.0000 recovery_on=0.9375
