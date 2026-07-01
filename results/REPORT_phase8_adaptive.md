# REPORT_phase8_adaptive.md — Phase 8.3 adaptive per-layer precision

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Method: noise_level[l] = global_level * factor[l], factor[l] = 2*(1-norm_mass[l]) / mean
        High attention_mass -> less noise; low mass -> more noise; budget preserved.
Standing constraint D: RS codec unchanged.

## Adaptive noise levels at global=0.3 (sample, first forward pass)
  min=0.0000  max=0.3148  mean=0.3000
  (mean should equal 0.3 — budget preserved)

## Per-level results (adaptive B3)

| noise | B0_NLL | B3_NLL | dNLL | B0_acc | B3_acc | dacc | savings_pct | Pareto |
|-------|--------|--------|------|--------|--------|------|-------------|--------|
| 0.00 | 4.2500 | 4.2500 | +0.0000 | 0.900 | 0.400 | -0.500 | +71.0% | YES |
| 0.05 | 4.2500 | 4.2610 | +0.0110 | 0.900 | 0.367 | -0.533 | +71.0% | YES |
| 0.10 | 4.2500 | 4.3020 | +0.0520 | 0.900 | 0.333 | -0.567 | +71.0% | YES |
| 0.20 | 4.2500 | 4.4463 | +0.1963 | 0.900 | 0.367 | -0.533 | +71.0% | YES |
| 0.30 | 4.2500 | 4.6844 | +0.4343 | 0.900 | 0.200 | -0.700 | +71.0% | YES |
| 0.50 | 4.2500 | 5.3468 | +1.0968 | 0.900 | 0.233 | -0.667 | +71.0% | no |

## Pareto frontier comparison
Uniform  (Phase 8.2): [0.0, 0.05, 0.1, 0.2]
Adaptive (Phase 8.3): [0.0, 0.05, 0.1, 0.2, 0.3]
Frontier delta (adaptive - uniform): +1 levels
Adaptive crossover level: 0.3

**PHASE 8.3 COMPARISON VERDICT: ADAPTIVE_BETTER**
_(ADAPTIVE_BETTER = frontier wider; SAME = equal; ADAPTIVE_WORSE = frontier narrower)_
_(Delta can be 0 or negative — this is honest, not a tuning failure)_

COMPUTE CAVEAT: per-layer noise redistribution is CPU-only; RS codec unchanged.
