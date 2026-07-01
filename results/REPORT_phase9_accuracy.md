# REPORT_phase9_accuracy.md — Phase 9.1 Task-Accuracy Axis

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Eval set: 100 probes (30 EVAL_PROBES + 70 allenai/sciq@validation rows 0-69)
Dataset: allenai/sciq@validation  rows=1000  fields=question,correct_answer
Noise levels: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
Retention crossover threshold: 0.5

## NLL vs Accuracy divergence
Phase 7.4 found B3 answered 'Paris' CORRECT at +0.64 NLL.
This report measures whether that pattern holds at scale (100 probes).
NLL and task-accuracy may diverge in either direction — both reported honestly.

## Per-level results

| noise | B0_NLL | B3_NLL | ΔNLL | B0_acc | B3_acc | Δacc | retention |
|-------|--------|--------|------|--------|--------|------|-----------|
| 0.00 | 4.2500 | 4.2500 | +0.0000 | 0.520 | 0.180 | -0.340 | 0.3462 |
| 0.05 | 4.2500 | 4.3254 | +0.0754 | 0.520 | 0.170 | -0.350 | 0.3269 |
| 0.10 | 4.2500 | 4.4274 | +0.1774 | 0.520 | 0.190 | -0.330 | 0.3654 |
| 0.20 | 4.2500 | 4.6651 | +0.4150 | 0.520 | 0.140 | -0.380 | 0.2692 |
| 0.30 | 4.2500 | 4.8881 | +0.6381 | 0.520 | 0.120 | -0.400 | 0.2308 |
| 0.50 | 4.2500 | 5.5091 | +1.2591 | 0.520 | 0.050 | -0.470 | 0.0962 |

## Accuracy retention
retention = acc(B3)/acc(B0); crossover = max noise where retention >= 0.5

Crossover noise: none
Retention at crossover: 0.3654

Interpretation: accuracy-axis crossover may differ from NLL-axis crossover
(Phase 8.2 NLL crossover = 0.2). Task accuracy and NLL measure different things.

COMPUTE CAVEAT: RS encode/decode CPU time not measured (same caveat as Phase 7.4/8.2).

ACCURACY_AXIS: retention=0.3654 at crossover=none
