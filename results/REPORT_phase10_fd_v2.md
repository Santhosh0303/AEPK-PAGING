# REPORT_phase10_fd_v2.md — Phase 10 step (7) FD redo with positive control

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). PREREG_phase10_fd_v2.md: positive-control rule fixed FIRST — single-layer levels [0.5, 1.0, 2.0] tried (ascending) on pre-named layers first/mid/last; smallest level with max damage >= 0.15 is used for the full sweep; if none reaches it the verdict is undetermined(no-response-regime) and NO rho is computed. Clean-correct subset n_cc=50; seeds=[0, 1, 2], seed derivation sd*1000+layer. Supersedes REPORT_phase10_fd.md (see its addendum).

## Positive control (level x pre-named layer -> retention damage)

| level | layer | damage |
|-------|-------|--------|
| 0.5 | 0 | 0.0400 |
| 0.5 | 14 | 0.0400 |
| 0.5 | 27 | 0.0200 |
| 1.0 | 0 | 0.2000 |
| 1.0 | 14 | 0.1800 |
| 1.0 | 27 | 0.0533 |

chosen_level=1.0

## Full per-layer sweep at level=1.0

| layer | key_norm_var (clean) | value_norm_var (clean) | retention_damage |
|-------|----------------------|------------------------|------------------|
| 0 | 0.96660 | 0.99934 | 0.2000 |
| 1 | 0.10783 | 0.28069 | 0.1933 |
| 2 | 3.88538 | 2.45343 | 0.2133 |
| 3 | 11.85167 | 1.98235 | 0.1467 |
| 4 | 12.50666 | 3.50056 | 0.1133 |
| 5 | 8.23088 | 2.97097 | 0.1533 |
| 6 | 14.52641 | 2.51018 | 0.0800 |
| 7 | 1.15478 | 2.74112 | 0.0933 |
| 8 | 12.15048 | 3.24897 | 0.1133 |
| 9 | 16.54094 | 2.23345 | 0.1267 |
| 10 | 17.10430 | 3.66734 | 0.1133 |
| 11 | 22.66313 | 1.98879 | 0.1333 |
| 12 | 8.83260 | 2.64867 | 0.1200 |
| 13 | 8.80877 | 2.33244 | 0.1267 |
| 14 | 13.28410 | 2.86287 | 0.1800 |
| 15 | 0.87146 | 2.58970 | 0.1600 |
| 16 | 9.73225 | 3.12657 | 0.2000 |
| 17 | 8.98503 | 4.42835 | 0.2400 |
| 18 | 3.35235 | 3.09028 | 0.1000 |
| 19 | 9.68235 | 4.91510 | 0.1800 |
| 20 | 9.76627 | 10.62481 | 0.1533 |
| 21 | 12.00539 | 6.92692 | 0.1467 |
| 22 | 13.13840 | 12.38391 | 0.1133 |
| 23 | 8.82605 | 22.39492 | 0.1067 |
| 24 | 5.31568 | 78.52467 | 0.0533 |
| 25 | 10.67472 | 139.83084 | 0.0733 |
| 26 | 8.82089 | 317.81094 | 0.0867 |
| 27 | 3.16098 | 250.25219 | 0.0533 |

## Interpretation
Spearman(key_norm_var, damage) = -0.1049 (primary). Spearman(value_norm_var, damage) = -0.4888 (secondary). n_layers=28. Damage now has real dynamic range (control gate passed at level=1.0), so supported/refuted are both meaningful; either is reported as-is.

FD: spearman=-0.1049 n_layers=28 verdict=refuted
