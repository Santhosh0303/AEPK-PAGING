# REPORT_phase10_stats.md — Phase 10 step (6) / 9.4 statistics (final config)

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA), the step-5 tolerant config. clean_acc=0.429. Stress: quant_noise level in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] on every KV page, 5 seeds. crossover=level where retention crosses FLOOR=0.7 (linear-interpolated; censoring reported). No cherry-pick — all seeds enter mu. Clean-correct conditioning (PREREG v3): N_cc=45 probes; sweep runs on that subset only (clean_acc=1.0 there), retention=corrupt_acc on the subset (<=1 by construction); seed derivation sd*1000+layer.

| seed | L=0.1 | L=0.2 | L=0.3 | L=0.4 | L=0.5 | L=0.6 | L=0.7 | L=0.8 | crossover |
|------|------|------|------|------|------|------|------|------|-----------|
| 0 | 0.933 | 0.844 | 0.733 | 0.622 | 0.600 | 0.467 | 0.289 | 0.089 | 0.330 |
| 1 | 0.911 | 0.911 | 0.889 | 0.844 | 0.689 | 0.533 | 0.267 | 0.067 | 0.493 |
| 2 | 0.911 | 0.911 | 0.889 | 0.778 | 0.622 | 0.533 | 0.311 | 0.089 | 0.450 |
| 3 | 0.889 | 0.911 | 0.800 | 0.711 | 0.667 | 0.422 | 0.178 | 0.044 | 0.425 |
| 4 | 0.933 | 0.800 | 0.689 | 0.644 | 0.578 | 0.222 | 0.022 | 0.000 | 0.290 |

## Interpretation
Per-seed crossover levels: [0.33, 0.493, 0.45, 0.425, 0.29]. Mean crossover mu=0.398, 95% CI half-width=0.105 (t-based, n=5). A tight CI means the compression-tolerance crossover of the final config is a stable statistic, not a single-seed artifact. Right-censored seeds (retention never below FLOOR through the top level 0.8) report crossover=0.8 and are visible in the table (retention row stays >= FLOOR).

STATS: crossover=0.398±0.105 seeds=5
