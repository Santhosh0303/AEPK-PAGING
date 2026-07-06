# PRE-REGISTRATION — Phase 10 step (18): heal-cost MICROBENCHMARKS (Path B minimal)

> WRITTEN BEFORE THE TIMING RUN. The quantities measured, the config, the rep count, the
> summary statistic, the double-run determinism gate, and the storage exact-match check are
> FIXED here so nothing is tuned. Honesty spine S9 unchanged. Deterministic math (median/IQR/
> overhead) is CPU-tested with no model. ALLOWED to land anywhere — heal may or may not beat
> recompute; a run-2 variance breach is reported as the finding, not smoothed away.

## Scope (FIXED — labeled MICROBENCHMARKS)
Primitive-operation latencies on the live-heal path, NOT serving throughput. There is NO
request stream, NO batching, NO vLLM / paged-attention integration; end-to-end serving cost is
explicitly out of scope (future work). This measures the cost of the operations the
erasure-conversion reframe adds/saves, on real Qwen2.5-1.5B KV pages.

## Config (FIXED — the live-heal config, reused from phase10_liveheal)
- Model: Qwen/Qwen2.5-1.5B-Instruct, fp16, CUDA.
- GROUP_SIZE = 4 sibling layer-pages per erasure group; NUM_PARITY = 1 (single-erasure recovery).
- Group = the top-GROUP_SIZE pages by `fp_key_norm_mean` influence (same selection as the live
  heal harness), for one representative probe from the step-16 combined pool (`get_combined_probes()[0]`).

## Quantities (FIXED — 5)
- (a) ENCODE: `encode_rs_erasure_group(group_pages, 1)` — parity build per group.
- (b) HEAL: `recover_rs_erasure(group, [target_page_id])` — restore ONE erased page from parity.
- (c) RECOMPUTE: re-prefill the same prompt prefix `model(ids[:, :-1], use_cache=True)` +
  `cuda.synchronize()` — the honest alternative to healing.
- (d) FINGERPRINT: `fp_key_norm_mean(page)` — per-page physics detection cost.
- (e) OVERHEAD: parity storage bytes, MEASURED (`group.parity_bytes.nbytes`) vs ANALYTIC
  (`num_parity * per-page-row-bytes`). Percentage = parity bytes / protected data bytes
  (= NUM_PARITY / GROUP_SIZE = 25% for equal-size sibling pages).

## Protocol (FIXED)
- N = 100 timed reps per timed quantity (a–d), one un-counted warm-up call each.
- Summary statistic = MEDIAN + IQR (P75 − P25), in milliseconds. (Median over mean: timing
  distributions are right-skewed by scheduler jitter.)
- The WHOLE timing suite runs TWICE (run-1, run-2).
- **Determinism gate (timing exemption, RUNTIME ECONOMY):** each run-2 median must be within
  **±20%** of the corresponding run-1 median. Timings never reproduce byte-exactly, so byte
  equality is NOT required for (a–d). If any quantity breaches ±20%, the report records
  `all_medians_within_tol=False` and the variance IS the finding — not hidden.
- **Storage bytes are exact-match (NOT timing):** measured == analytic, both runs. Asserted exact.

## Runtime verdict (FIXED f-string)
`HEAL_COST: heal_ms=<med> recompute_ms=<med> ratio=<recompute/heal> parity_overhead_pct=<p>`
plus `DETERMINISM: all_medians_within_tol=<bool> parity_bytes_exact=<bool>`. `ratio` is the
per-operation speedup of healing over a full prefix recompute on THIS microbenchmark — NOT an
end-to-end serving number (scope caveat above). ALLOWED to be < 1 (heal slower) — reported as-is.

## Honesty / determinism
Zero edits to Phase 2-5 source. New harness `phase10_healcost.py` reuses
`coding.encode_rs_erasure_group` / `recover_rs_erasure` / `_page_row_bytes`,
`phase9_cw.fp_key_norm_mean`, and the live-heal `GROUP_SIZE`/`NUM_PARITY`. The median/IQR/overhead/
tolerance math is deterministic and CPU-tested on synthetic timings + synthetic KV pages
(`tests/test_phase10_healcost.py`). Report: results/REPORT_phase10_healcost.md.
