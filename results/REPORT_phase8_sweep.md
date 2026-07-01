# REPORT_phase8_sweep.md — Phase 8.2 quant_noise sweep

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Noise levels swept: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
NLL threshold (unchanged from Phase 6 / 7): 0.5

## Per-level results

| noise | B0_NLL | B3_NLL | ΔNLL | B0_acc | B3_acc | Δacc | B3_storage_bits | savings_pct | Pareto |
|-------|--------|--------|------|--------|--------|------|-----------------|-------------|--------|
| 0.00 | 4.2500 | 4.2500 | +0.0000 | 0.900 | 0.400 | -0.500 | 931,840 | +71.0% | YES |
| 0.05 | 4.2500 | 4.3254 | +0.0754 | 0.900 | 0.367 | -0.533 | 931,840 | +71.0% | YES |
| 0.10 | 4.2500 | 4.4274 | +0.1774 | 0.900 | 0.400 | -0.500 | 931,840 | +71.0% | YES |
| 0.20 | 4.2500 | 4.6651 | +0.4150 | 0.900 | 0.300 | -0.600 | 931,840 | +71.0% | YES |
| 0.30 | 4.2500 | 4.8881 | +0.6381 | 0.900 | 0.200 | -0.700 | 931,840 | +71.0% | no |
| 0.50 | 4.2500 | 5.5091 | +1.2591 | 0.900 | 0.133 | -0.767 | 931,840 | +71.0% | no |

## Pareto frontier
Noise levels where ΔNLL ≤ 0.5 AND B3 saves storage vs B0:
  [0.0, 0.05, 0.1, 0.2]

Crossover level (max Pareto noise): 0.2

Interpretation: below the crossover level, AEPK's storage savings come at
acceptable NLL cost (≤0.5 nats). Above it, the damage exceeds the threshold.

COMPUTE CAVEAT: RS encode/decode CPU time not measured in sweep (same caveat as Phase 7.4).

**PHASE 8 SWEEP VERDICT: PASS**
_(PASS = crossover exists; FAIL = AEPK never within NLL threshold at any noise level)_
