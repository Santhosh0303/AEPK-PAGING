# PRE-REGISTRATION v3 — Phase 10 step (6) / 9.4: stats rerun, grid-consistent methodology

> WRITTEN BEFORE THE GPU RUN. FLOOR, noise LEVELS, SEEDS, probe set, crossover definition,
> CI method, and the STATS verdict-line format are UNCHANGED from v2. Two methodology changes
> only, both pre-registered here with their motivation recorded verbatim. Honesty spine S9
> unchanged: zero edits to Phase 2-5 source; verdict line is a runtime f-string; tests assert
> line-exists, never a value.

## Motivation (recorded verbatim, backlog step 10)
"(a) seed derivation `sd*1000 + p.layer` (kill cross-seed RNG overlap; same fix as grid/fd).
Also fix the same `sd + p.layer` in `phase10_floor_law.py` (code-only; no floor-law rerun —
grid supersedes it). (b) clean-correct conditioning: compute per-probe clean pass first, keep
only clean-correct probes, retention = corrupt_acc on that subset (clean_acc=1.0 by
construction; retention can no longer exceed 1 systematically). Inclusion note N_cc in report.
PREREG_phase10_stats_v3.md BEFORE GPU (same FLOOR/levels/seeds; methodology change motivated
by C2 inconsistency + retention>1 rows, recorded verbatim)."

C2 inconsistency: the step-5 grid (PREREG_phase10_grid.md) measures retention on the model's
clean-correct subset with seed derivation sd*1000+layer; the v2 stats run used the whole probe
set (retention = corrupt_acc/clean_acc, which produced retention>1 rows when corruption
"fixed" probes the model got wrong clean) and overlapping seeds (sd + layer collides across
seeds: seed s, layer l+1 == seed s+1, layer l). This v3 aligns the stats methodology with the
grid so the two are comparable.

## Changes vs v2 (FIXED — this amendment)
1. Seed derivation: `quant_noise(page, level, seed = sd*1000 + page.layer)` — non-overlapping
   across seeds (layer < 1000 always here).
2. Clean-correct conditioning: per-probe clean pass computed first on LARGE_PROBES;
   the per-level/per-seed sweep runs ONLY on the clean-correct subset;
   retention(level, seed) = corrupt_acc on that subset (clean_acc = 1.0 there by
   construction, so retention <= 1 always). N_cc reported in the report header.

## Unchanged from v2 (FIXED)
- Probe set: `eval_set_large` large set (n=105, granularity <= 1/100), RAW prompt formatting.
- Model: Qwen/Qwen2.5-1.5B-Instruct fp16, CUDA (the step-5 tolerant config).
- Stress: quant_noise on every KV page at LEVELS=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8).
- SEEDS=(0,1,2,3,4) — 5 seeds; ALL seeds enter mu (no cherry-pick).
- crossover = level where retention crosses FLOOR=0.70, linear-interpolated; censoring reported.
- mu +/- 95% CI (t-based) across seeds.

## Verdict line (FIXED)
```
STATS: crossover=<mu>±<ci> seeds=<n>
```
Runtime f-string; tests assert the LINE EXISTS and seeds==5, never a crossover value.
GPU run TWICE; per-seed rows byte-identical. Report: results/REPORT_phase10_stats.md
(header gains `N_cc=<n>` inclusion note).

## Floor-law seed fix (code-only, no rerun)
`phase10_floor_law.py` line ~101 gets the same `sd*1000 + p.layer` derivation. The floor-law
harness is NOT rerun — the step-5 grid supersedes it (its report already carries the step-3
exclusion decision); this only stops the stale derivation from propagating into future code.
