# REPORT_phase8_baselines.md — Phase 8.4 baseline comparison

## Baselines
| Method | NLL | Accuracy | Storage bits | Storage% | AEPK dominates? |
|--------|-----|----------|-------------|----------|-----------------|
| UQ-8bit | 4.2148 | 0.367 | 1,605,632 | 50.0% | YES |
| UQ-4bit | 5.3386 | 0.000 | 802,816 | 25.0% | no |
| H2O-25pct | 4.6351 | 0.033 | 2,408,448 | 75.0% | YES |
| H2O-50pct | 4.8593 | 0.000 | 1,605,632 | 50.0% | YES |
| H2O-75pct | 6.6404 | 0.000 | 802,816 | 25.0% | no |

## AEPK adaptive reference points (Phase 8.3)
| NLL | Storage bits |
|-----|-------------|
| 4.2500 | 931,840 |
| 4.2610 | 931,840 |
| 4.3020 | 931,840 |
| 4.4463 | 931,840 |
| 4.6844 | 931,840 |
| 5.3468 | 931,840 |

Dominance criterion: AEPK has storage <= baseline AND NLL within 0.05 nats.

**PHASE 8.4 DOMINANCE VERDICT: AEPK_DOMINATES_SOME**
_(AEPK_DOMINATES_ALL = dominates every baseline; SOME = partial; NONE = no dominance)_
_(Verdict may be NONE — this is honest, not a failure)_
