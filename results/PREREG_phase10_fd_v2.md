# PRE-REGISTRATION v2 — Phase 10 step (7) FD redo with POSITIVE CONTROL

> WRITTEN BEFORE THE GPU RUN. The v1 run (level=0.2 single-layer) produced per-layer damage
> 0-0.033 — a flat, tie-heavy response on which Spearman has no refuting power (see the
> ADDENDUM in REPORT_phase10_fd.md: verdict reclassified refuted -> UNDETERMINED). This v2
> fixes the positive-control rule FIRST, before any full sweep.

## Positive-control rule (FIXED FIRST — this is the gate)
1. Sweep single-layer corruption level in {0.5, 1.0, 2.0} (ascending) on exactly 3
   pre-named layers: FIRST (index 0), MID (index n_layers//2), LAST (index n_layers-1)
   — for Qwen2.5-1.5B n_layers=28, so layers 0, 14, 27.
2. At each level compute per-layer retention damage on the clean-correct subset
   (same subset construction as v1; seeds=(0,1,2); seed derivation sd*1000+layer).
3. Pick the SMALLEST level whose max per-layer damage across the 3 control layers
   >= 0.15 (real dynamic range).
4. If NO level reaches 0.15: verdict = `undetermined(no-response-regime)`, STOP the step —
   NO rho is computed (a correlation on a flat response is meaningless either way).

## Full sweep (only if the control gate passes)
- Corrupt ONE layer at a time, ALL n_layers layers, at the chosen level.
- seeds=(0,1,2), seed derivation sd*1000+layer (non-overlapping).
- rho = Spearman(clean per-layer key-norm variance, per-layer damage) — same primary
  statistic, same fixed direction (higher clean fluctuation => more damage).
- Same thresholds as v1: supported iff rho >= 0.60; null/refuted iff |rho| < 0.30;
  else undetermined. Refuted at a level WITH real dynamic range is now a meaningful null.

## Verdict line (FIXED)
```
FD: spearman=<rho> n_layers=<n> verdict=<supported|refuted|undetermined|undetermined(no-response-regime)>
```
Runtime f-string; tests assert line-exists, never a value.

## Reporting / honesty
- Results land in results/REPORT_phase10_fd_v2.md (NEW file — the v1 report keeps its
  addendum untouched; no scrubbing). Control table (level x 3 layers) always reported.
- GPU run TWICE; report byte-identical.
- Zero edits to Phase 2-5 source. v1 functions kept; v2 adds run_fd_v2/write_fd_report_v2.
- Either outcome (supported / refuted / no-response) reported as-is.
