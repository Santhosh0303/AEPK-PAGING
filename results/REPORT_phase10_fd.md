# REPORT_phase10_fd.md — Phase 10 step (7) FENCED: fluctuation-dissipation analogue

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). From CLEAN KV only: per-layer variance of per-token key norms (primary) and value norms (secondary), averaged over the 50 clean-correct probes. Prediction (PREREG, direction FIXED): higher clean key-norm fluctuation => more retention damage when that layer alone is corrupted (quant_noise level=0.2, seeds=[0, 1, 2], seed sd*1000+layer). Compared by Spearman rho. supported iff rho>=0.6; null/refuted iff |rho|<0.3. Refuted is a real result.

| layer | key_norm_var (clean) | value_norm_var (clean) | retention_damage |
|-------|----------------------|------------------------|------------------|
| 0 | 0.96660 | 0.99934 | 0.0200 |
| 1 | 0.10783 | 0.28069 | 0.0133 |
| 2 | 3.88538 | 2.45343 | 0.0000 |
| 3 | 11.85167 | 1.98235 | 0.0133 |
| 4 | 12.50666 | 3.50056 | 0.0000 |
| 5 | 8.23088 | 2.97097 | 0.0267 |
| 6 | 14.52641 | 2.51018 | 0.0133 |
| 7 | 1.15478 | 2.74112 | 0.0067 |
| 8 | 12.15048 | 3.24897 | 0.0267 |
| 9 | 16.54094 | 2.23345 | 0.0067 |
| 10 | 17.10430 | 3.66734 | 0.0267 |
| 11 | 22.66313 | 1.98879 | 0.0267 |
| 12 | 8.83260 | 2.64867 | 0.0133 |
| 13 | 8.80877 | 2.33244 | 0.0000 |
| 14 | 13.28410 | 2.86287 | 0.0200 |
| 15 | 0.87146 | 2.58970 | 0.0333 |
| 16 | 9.73225 | 3.12657 | 0.0200 |
| 17 | 8.98503 | 4.42835 | 0.0333 |
| 18 | 3.35235 | 3.09028 | 0.0333 |
| 19 | 9.68235 | 4.91510 | 0.0333 |
| 20 | 9.76627 | 10.62481 | 0.0200 |
| 21 | 12.00539 | 6.92692 | 0.0200 |
| 22 | 13.13840 | 12.38391 | 0.0133 |
| 23 | 8.82605 | 22.39492 | 0.0267 |
| 24 | 5.31568 | 78.52467 | 0.0067 |
| 25 | 10.67472 | 139.83084 | 0.0067 |
| 26 | 8.82089 | 317.81094 | 0.0133 |
| 27 | 3.16098 | 250.25219 | 0.0000 |

## Interpretation
Spearman(key_norm_var, damage) = 0.0664 (primary). Spearman(value_norm_var, damage) = 0.0264 (secondary). n_layers=28. If supported, clean equilibrium fluctuations forecast corruption vulnerability — an FD analogue. If refuted (|rho| small), there is no such analogue on this workload and that is the finding, reported as-is (not reframed). A wrong-sign or weak rho is undetermined.

FD: spearman=0.0664 n_layers=28 verdict=refuted

## ADDENDUM (2026-07-05, step 13 — appended, existing text unmodified)
Verdict RECLASSIFIED: refuted -> UNDETERMINED. The measured per-layer damage range in this
run was 0 to 0.033 (tie-heavy, quantized at the 1/50 probe granularity) — the perturbation
(single-layer quant_noise at level=0.2) produced essentially NO response on any layer, so
the response variable is flat. A Spearman rank correlation computed on a flat, tie-dominated
response has no power to refute the fluctuation-dissipation prediction; the honest reading is
that level=0.2 on one layer is BELOW the response threshold of this workload, not that clean
fluctuation fails to forecast vulnerability. The v2 redo (PREREG_phase10_fd_v2.md) fixes a
positive-control rule FIRST (find a single-layer level with real dynamic range, else declare
undetermined(no-response-regime) and stop); its results land in
results/REPORT_phase10_fd_v2.md, which supersedes this report.
