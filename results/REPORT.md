# AEPK-Paging Phase 6 REPORT

This is a numpy-only simulation net-overhead report. It is necessary but not sufficient; Bar 2 on real model KV is Phase 7.

## Gate Definition
- Rate-distortion currency uses `[Shannon]`: `total_cost(λ) = storage_bits + λ * residual_error`.
- λ sweep: `1.00e+00` to `1.00e+09` bits per unit residual MSE.
- PASS iff B3 is Pareto-non-dominated and B3 wins total-cost for at least one reported λ-range.
- Compute proxy is reported as a caveat only; it is not mixed into the rate-distortion gate.
- 12x compute caveat: B3 uses 12.00 detector/recovery proxy ops in the primary scenario.

## Corruption Scenario
- Seed: `611`
- Corruptions: `quant_noise(level=0.35)`, `bit_flip(p=0.0008)`, `forced_evict(page_ids=['p0'])`

## Scenario: primary
- AEPK residency budget bits: `11264`

### Baseline Matrix
| Baseline | Quality loss MSE | Storage bits | Compute proxy | Residual error | Notes |
|---|---:|---:|---:|---:|---|
| B0 no protection | 0.23508541 | 0 | 0.00 | 0.23508541 | Takes quant-noise, raw bit-flip damage, and forced eviction. |
| B1 keep-all-RESIDENT | 0.00000000 | 32768 | 0.00 | 0.00000000 | Cost ceiling: clean resident pages, no damage. |
| B2 erasure-parity only | 0.06088495 | 8192 | 1.00 | 0.06088495 | GhostServe-like known-erasure recovery; no unknown-location bit-flip correction. |
| B3 full AEPK stack | 0.00002333 | 20992 | 12.00 | 0.00002333 | Detection + parity/SECDED recovery + thermodynamic residency decision. |

### Pareto Table
| Baseline | Storage bits | Residual error | Dominated |
|---|---:|---:|---:|
| B0 | 0 | 0.23508541 | False |
| B1 | 32768 | 0.00000000 | False |
| B2 | 8192 | 0.06088495 | False |
| B3 | 20992 | 0.00002333 | False |

### λ Win Ranges
| Winner | λ start | λ end |
|---|---:|---:|
| B0 | 1.00000000e+00 | 4.46683592e+04 |
| B2 | 5.01187234e+04 | 1.99526231e+05 |
| B3 | 2.23872114e+05 | 5.01187234e+08 |
| B1 | 5.62341325e+08 | 1.00000000e+09 |

### AEPK Residency Decisions
| Page | Tier | Detector flagged |
|---|---|---:|
| p0 | CODED | True |
| p1 | CODED | True |
| p2 | CODED | True |
| p3 | CODED | False |

### Corrected Gate
- B3 Pareto-non-dominated: `True`
- B3 λ win range(s): `2.24e+05..5.01e+08`
- Scenario verdict: `PASS`
- Tier distribution: `RESIDENT=0, CODED=4, EVICTED=0`

## Scenario: tight-budget tier stress
- AEPK residency budget bits: `9216`
- Tight-budget tier stress uses higher attention_mass values to exercise residency tiers; Phase 2-5 constants are unchanged.

### Baseline Matrix
| Baseline | Quality loss MSE | Storage bits | Compute proxy | Residual error | Notes |
|---|---:|---:|---:|---:|---|
| B0 no protection | 0.23508541 | 0 | 0.00 | 0.23508541 | Takes quant-noise, raw bit-flip damage, and forced eviction. |
| B1 keep-all-RESIDENT | 0.00000000 | 32768 | 0.00 | 0.00000000 | Cost ceiling: clean resident pages, no damage. |
| B2 erasure-parity only | 0.06088495 | 8192 | 1.00 | 0.06088495 | GhostServe-like known-erasure recovery; no unknown-location bit-flip correction. |
| B3 full AEPK stack | 0.53381635 | 26112 | 12.00 | 0.53381635 | Detection + parity/SECDED recovery + thermodynamic residency decision. |

### Pareto Table
| Baseline | Storage bits | Residual error | Dominated |
|---|---:|---:|---:|
| B0 | 0 | 0.23508541 | False |
| B1 | 32768 | 0.00000000 | False |
| B2 | 8192 | 0.06088495 | False |
| B3 | 26112 | 0.53381635 | True |

### λ Win Ranges
| Winner | λ start | λ end |
|---|---:|---:|
| B0 | 1.00000000e+00 | 4.46683592e+04 |
| B2 | 5.01187234e+04 | 3.98107171e+05 |
| B1 | 4.46683592e+05 | 1.00000000e+09 |

### AEPK Residency Decisions
| Page | Tier | Detector flagged |
|---|---|---:|
| p0 | RESIDENT | True |
| p1 | CODED | True |
| p2 | EVICTED | True |
| p3 | EVICTED | True |

### Corrected Gate
- B3 Pareto-non-dominated: `False`
- B3 λ win range(s): `none`
- Scenario verdict: `FAIL`
- Tier distribution: `RESIDENT=1, CODED=1, EVICTED=2`

## Corrected Gate Verdict
- Primary scenario verdict: `PASS`
- Tight-budget scenario verdict: `FAIL`
GATE VERDICT: FAIL
