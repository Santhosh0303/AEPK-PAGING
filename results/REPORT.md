# AEPK-Paging Phase 6 REPORT

This is a numpy-only simulation net-overhead report. It is necessary but not sufficient; Bar 2 on real model KV is Phase 7.

## Scenario
- Seed: `611`
- Corruptions: `quant_noise(level=0.35)`, `bit_flip(p=0.0008)`, `forced_evict(page_ids=['p0'])`
- Gate recovery target X: `0.80`

## Baseline Matrix
| Baseline | Quality loss MSE | Storage bits | Compute proxy | Residual error | Total overhead proxy | Notes |
|---|---:|---:|---:|---:|---:|---|
| B0 no protection | 0.23508541 | 0 | 0.00 | 0.23508541 | 0.23508541 | Takes quant-noise, raw bit-flip damage, and forced eviction. |
| B1 keep-all-RESIDENT | 0.00000000 | 32768 | 0.00 | 0.00000000 | 32768.00000000 | Cost ceiling: clean resident pages, no damage. |
| B2 erasure-parity only | 0.06088495 | 8192 | 1.00 | 0.06088495 | 8193.06088495 | GhostServe-like known-erasure recovery; no unknown-location bit-flip correction. |
| B3 full AEPK stack | 0.00002333 | 20992 | 12.00 | 0.00002333 | 21004.00002333 | Detection + parity/SECDED recovery + thermodynamic residency decision. |

## AEPK Residency Decisions
| Page | Tier | Detector flagged |
|---|---|---:|
| p0 | CODED | True |
| p1 | CODED | True |
| p2 | CODED | True |
| p3 | CODED | False |

## Net-Overhead Gate
- `damage_cost = B0_quality_loss - B1_quality_loss = 0.23508541`
- `heal_overhead = B3_extra_bits + compute_proxy + residual_error = 21004.00002333`
- Recovery condition: `0.99990076 >= 0.80000000` -> `True`
- Overhead condition: `21004.00002333 < 0.23508541` -> `False`
- Error-regime condition: `B3_quality_loss < B2_quality_loss` -> `True`
- B3 beats B2 on error regime: `True`

GATE VERDICT: FAIL
