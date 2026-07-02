# REPORT_phase9_accuracy.md — Phase 9.1-FIX Task-Accuracy Axis

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Eval set: 100 probes (30 EVAL_PROBES + 70 allenai/sciq@validation rows 0-69)
Dataset: allenai/sciq@validation  rows=1000  fields=question,correct_answer
Noise levels: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
Seeds per noise level: 5
Retention crossover threshold: 0.5
RS config: num_parity=2 / recover-worst-2 (aligned with NLL path)

## Fix notes
commit 1c0f7a5 bug: model.generate(ids, past_key_values=pkv) double-processed ids.
Fix: both B0 and B3 use manual greedy loop from prefill logit; no ids re-feed.
Control: noise=0 row must show retention=1.0000 (bit-exact KV round-trip).

## NLL vs Accuracy divergence
Phase 7.4 found B3 answered Paris CORRECT at +0.64 NLL.
This report measures whether that holds at scale (100 probes, 5 seeds).
NLL and task-accuracy may diverge in either direction — both reported honestly.

## Per-level results (B3 mean ± 95% CI over 5 seeds)

| noise | B0_NLL | B3_NLL | ΔNLL | B0_acc | B3_acc_mean | B3_acc_ci | retention_mean | retention_ci |
|-------|--------|--------|------|--------|-------------|-----------|----------------|--------------|
| 0.00 | 4.2500 | 4.2500 | +0.0000 | 0.330 | 0.330 | ±0.000 | 1.0000 | ±0.0000 |
| 0.05 | 4.2500 | 4.3254 | +0.0754 | 0.330 | 0.314 | ±0.013 | 0.9515 | ±0.0403 |
| 0.10 | 4.2500 | 4.4274 | +0.1774 | 0.330 | 0.318 | ±0.017 | 0.9636 | ±0.0511 |
| 0.20 | 4.2500 | 4.6651 | +0.4150 | 0.330 | 0.324 | ±0.010 | 0.9818 | ±0.0303 |
| 0.30 | 4.2500 | 4.8881 | +0.6381 | 0.330 | 0.322 | ±0.028 | 0.9758 | ±0.0848 |
| 0.50 | 4.2500 | 5.5091 | +1.2591 | 0.330 | 0.350 | ±0.032 | 1.0606 | ±0.0976 |

## Accuracy retention
retention_mean = mean(acc(B3)/acc(B0)) over 5 seeds per level
crossover = max noise where retention_mean >= 0.5

Crossover noise: 0.5
Retention at crossover: 1.0606
Curve monotone non-increasing: False

Interpretation: accuracy-axis crossover may differ from NLL-axis crossover
(Phase 8.2 NLL crossover = 0.2). Task accuracy and NLL measure different things.

INTERPRETATION CAVEAT: retention~1.0 at all noise (incl 0.5, where ΔNLL=+1.26) is NOT yet a "noise doesn't hurt accuracy" finding — it is uninterpreted. Likely RS recover-worst-2 over-recovers on SHORT probe prompts (few KV pages) vs the long NLL held-out text. Resolve in Phase 9.3 damage-only ablation (strip RS recovery): if damage-only degrades with noise while recovery-on stays ~1.0 = "RS recovery restores accuracy" (positive); do NOT interpret this line before that control. Flat curve (non-monotone) is structural signal-erasure, not sampling noise — more seeds won't change it.

COMPUTE CAVEAT: RS encode/decode CPU time not measured (same caveat as Phase 7.4/8.2).

STATS: crossover=0.5 retention_ci=±0.0976 seeds=5
ACCURACY_AXIS: retention=1.0606 at crossover=0.5
