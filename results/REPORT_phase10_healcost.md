# REPORT_phase10_healcost.md — Phase 10 step (18) heal-cost MICROBENCHMARKS (Path B minimal)

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Live-heal config: GROUP_SIZE=4 sibling layer-pages, NUM_PARITY=1 (single-erasure recovery). Group page K-shape=(306, 2, 128). N=100 timed reps per quantity; MEDIAN + IQR (ms). Suite run TWICE; PREREG gate = run-2 median within +/-20% of run-1 (timings never reproduce byte-exactly). Parity storage BYTES are exact-match (measured vs analytic).

**MICROBENCHMARKS — primitive-operation latencies, NOT serving throughput.** No request stream, no batching, no vLLM/paged-attention integration; end-to-end serving cost is future work. The load-bearing comparison is HEAL vs RECOMPUTE (a full prefix re-prefill).

| quantity | run1 median (ms) | run1 IQR | run2 median (ms) | within +/-20% |
|----------|------------------|----------|------------------|---------------|
| encode | 39.1531 | 2.1506 | 38.9002 | True |
| heal | 67.3512 | 2.2326 | 67.3874 | True |
| recompute | 164.2378 | 15.7208 | 174.3506 | True |
| fingerprint | 0.0406 | 0.0005 | 0.0404 | True |

Parity storage: measured=626688 bytes analytic=626688 bytes exact_match=True (overhead=25.00% of protected data = NUM_PARITY/GROUP_SIZE).

## Interpretation
HEAL restores one page from parity by inverting a k x k Cauchy submatrix over GF(2^8) — a fixed-size linear-algebra op independent of prompt length. RECOMPUTE re-prefills the whole prompt prefix through the model — it grows with context. On this single-page, short-prompt microbenchmark the ratio below is the per-operation speedup of erasure healing over recompute; it is NOT an end-to-end serving number. FINGERPRINT is the per-page detection cost paid at encode time; ENCODE is the one-time parity build per group.

HEAL_COST: heal_ms=67.3512 recompute_ms=164.2378 ratio=2.44 parity_overhead_pct=25.00

DETERMINISM: all_medians_within_tol=True (per-quantity: {'encode': True, 'heal': True, 'recompute': True, 'fingerprint': True}); parity_bytes_exact=True.
