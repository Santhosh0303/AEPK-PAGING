# REPORT_phase8_scale.md — Phase 8.5 scale generalization check

Noise level: 0.2 (uniform crossover from Phase 8.2)
NLL threshold: 0.5
Pareto criterion: ΔNLL ≤ threshold AND b3_storage < b0_storage

## Results
| Model | Ctx | Prefix toks | B0_NLL | B3_NLL | ΔNLL | savings% | Pareto |
|-------|-----|-------------|--------|--------|------|----------|--------|
| Qwen2.5-1.5B-Instruct | short | 7 | 4.2500 | 4.6036 | +0.3535 | +71.0% | YES |
| Qwen2.5-1.5B-Instruct | long | 148 | 3.6699 | 4.0086 | +0.3388 | +80.4% | YES |
| Qwen2.5-0.5B-Instruct | short | 7 | 5.2198 | 8.4475 | +3.2277 | +57.3% | no |
| Qwen2.5-0.5B-Instruct | long | 148 | 4.4800 | 6.4926 | +2.0126 | +79.2% | no |

Cells on Pareto: 2/4

**PHASE 8.5 GENERALIZATION VERDICT: GENERALIZES_SOME**
_(GENERALIZES_ALL = crossover exists at every scale; SOME = partial; NONE = no crossover)_
_(NONE is honest — not a build failure)_
