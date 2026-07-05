# PRE-REGISTRATION v2 — Phase 10 step (6) / 9.4: statistics on >=100 probes

> WRITTEN BEFORE THE GPU RUN. Only the PROBE SET changes vs v1 (short 8-probe CW_PROBES ->
> the >=100-probe leakage-free large set). FLOOR, noise LEVELS, SEEDS, crossover definition,
> CI method, and the STATS verdict line are UNCHANGED. Honesty spine S9 unchanged.
> Deterministic (fixed construction + fixed seed loop, no RNG elsewhere).

## Motivation
The v1 stats used the 8-probe CW set, so accuracy moved in steps of 1/8 = 0.125 — too coarse
for a retention curve / crossover estimate. This amendment reruns the SAME statistics on a
>=100-probe set so accuracy granularity is <= 1/100.

## Probe set (FIXED — this amendment)
`aepk_paging.harness.eval_set_large.LARGE_PROBES`, built deterministically as:
- Phase 9.3a long-context probes: `build_lc_probe_set()` = 30 curated EVAL_PROBES + 70
  allenai/sciq rows, each prefixed with the shared LONG_CONTEXT_PASSAGE.
- Short-factual `CW_PROBES` (8).
- Every probe normalized to carry `expected` (str) and `alternatives` (list).
- **Leakage filter**: any probe whose gold answer or an alternative appears as a whole word
  (case-insensitive) in its own prompt is DROPPED (it would be answerable by copying from the
  context passage rather than by parametric recall). This removes the Jupiter / Darwin /
  "theory" long-context probes whose answer sits in the passage.
Result: n=105 unique probes, granularity 1/105 ≈ 0.0095 <= 1/100. Fixed for this run.

## Unchanged from v1 (FIXED)
- Model: Qwen/Qwen2.5-1.5B-Instruct fp16, CUDA (the step-5 tolerant config).
- Stress: quant_noise on every KV page at LEVELS=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8).
- SEEDS=(0,1,2,3,4) — 5 seeds; ALL seeds enter mu (no cherry-pick).
- retention = corrupt_acc / clean_acc. crossover = level where retention crosses FLOOR=0.70,
  linear-interpolated; left/right censoring reported.
- mu +/- 95% CI (t-based) across seeds.
- Prompt formatting: RAW (completion-style probes), consistent with the Step-3 HITL decision.

## Verdict line (FIXED)
```
STATS: crossover=<mu>±<ci> seeds=<n>
```
Runtime f-string; tests assert the LINE EXISTS and seeds==5, never a crossover value.
GPU run foreground TWICE; per-seed rows byte-identical. Report: results/REPORT_phase10_stats.md.

## Honesty / determinism
Zero edits to Phase 2-5 source. New file eval_set_large.py (reuses build_lc_probe_set,
CW_PROBES) + CPU tests only; phase10_stats.py __main__ wired to LARGE_PROBES.
