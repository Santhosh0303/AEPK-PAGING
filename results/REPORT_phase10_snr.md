# REPORT_phase10_snr.md — Phase 10 SNR CAMPAIGN (mechanism day, steps 19-21)

Derived law (see PREREG_phase10_snr.md for the verbatim theory): L_c = C * sqrt(head_dim) * RMS_K. SNR susceptibility score s = sqrt(head_dim)*RMS_K.

## P2 — SNR score vs the 7 included grid_v2 models (EXPLORATORY; retentions already seen)

| model | head_dim | N_cc | RMS_K | SNR score s | tolerant |
|-------|----------|------|-------|-------------|----------|
| qwen0.5b | 64 | 127 | 9.73003 | 77.8402 | False |
| qwen1.5b | 128 | 151 | 13.32013 | 150.7000 | True |
| tinyllama | 64 | 77 | 2.47144 | 19.7715 | False |
| pythia-410m | 64 | 52 | 6.51225 | 52.0980 | False |
| pythia-1b | 256 | 74 | 1.52930 | 24.4687 | True |
| pythia-1.4b | 128 | 72 | 10.58207 | 119.7224 | True |
| smollm2-360m | 64 | 91 | 2.13248 | 17.0598 | False |

Scores sorted (ascending): [['smollm2-360m', 17.0598], ['tinyllama', 19.7715], ['pythia-1b', 24.4687], ['pythia-410m', 52.098], ['qwen0.5b', 77.8402], ['pythia-1.4b', 119.7224], ['qwen1.5b', 150.7]]. `separable` asks whether ONE threshold on s splits the 3 tolerant (high) from the 4 intolerant (low). `margin_vs_hd` = relative s-separation band / relative head_dim-separation band (>1 => the continuous SNR score separates with a wider relative margin than the binary head_dim split). Exploratory — no gate.

SNR_RANK: scores=[['smollm2-360m', 17.0598], ['tinyllama', 19.7715], ['pythia-1b', 24.4687], ['pythia-410m', 52.098], ['qwen0.5b', 77.8402], ['pythia-1.4b', 119.7224], ['qwen1.5b', 150.7]] separable=False margin_vs_hd=-1.1981

## P3 — per-layer damage vs per-layer MEAN key RMS (qwen1.5b; damage reused from fd_v2)

Magnitude (not variance) is the predicted susceptibility variable: higher clean key RMS => MORE per-layer damage is the FD sign, but the theory predicts the ANTICORRELATION (a layer whose keys are large has a large signal logit gap -> the fixed absolute noise is relatively weaker -> LESS damage). Gate fixed pre-measurement: supported iff rho<=-0.5; refuted iff rho>=-0.2. Per-layer damage reused verbatim from REPORT_phase10_fd_v2.md (level=1.0 sweep, n_cc=50).

| layer | mean key RMS (clean) | retention_damage (fd_v2) |
|-------|----------------------|--------------------------|
| 0 | 68.94664 | 0.2000 |
| 1 | 9.76171 | 0.1933 |
| 2 | 2.40938 | 0.2133 |
| 3 | 1.91546 | 0.1467 |
| 4 | 1.66566 | 0.1133 |
| 5 | 1.58150 | 0.1533 |
| 6 | 1.74044 | 0.0800 |
| 7 | 3.30338 | 0.0933 |
| 8 | 1.76850 | 0.1133 |
| 9 | 1.84109 | 0.1267 |
| 10 | 1.79110 | 0.1133 |
| 11 | 1.89217 | 0.1333 |
| 12 | 1.57956 | 0.1200 |
| 13 | 1.77052 | 0.1267 |
| 14 | 1.68038 | 0.1800 |
| 15 | 5.97431 | 0.1600 |
| 16 | 1.63155 | 0.2000 |
| 17 | 1.50794 | 0.2400 |
| 18 | 2.00844 | 0.1000 |
| 19 | 1.64821 | 0.1800 |
| 20 | 1.60671 | 0.1533 |
| 21 | 1.59358 | 0.1467 |
| 22 | 1.72255 | 0.1133 |
| 23 | 1.54549 | 0.1067 |
| 24 | 1.61491 | 0.0533 |
| 25 | 1.51384 | 0.0733 |
| 26 | 1.47091 | 0.0867 |
| 27 | 1.34172 | 0.0533 |

Spearman(mean key RMS, damage) = 0.2995 (primary, P3). Spearman(mean value RMS, damage) = -0.4073 (secondary — the -0.4888 value-norm thread; reported as-is, no gate). n_layers=28.

SNR_FD: spearman=0.2995 n_layers=28 verdict=refuted

## P1 — out-of-sample crossover prediction (qwen0.5b), step-20 GATE

Predicted from the step-19 LOCKED addendum: qwen0.5b crossover = 0.398 * sqrt(64/128) * (RMS_K_q0.5b/RMS_K_q1.5b) = 0.2056 (prediction written down BEFORE the sweep launched). Success band half-width = 0.105*(pred/0.398) + measured_CI = 0.0542. Same pool as calibration (LARGE_PROBES n=105), same levels/seeds/FLOOR/conditioning as PREREG v3. A clean refutation is a publishable falsification, reported at equal prominence (ALLOWED-to-FAIL).

| seed | L=0.1 | L=0.2 | L=0.3 | L=0.4 | L=0.5 | L=0.6 | L=0.7 | L=0.8 | crossover |
|------|------|------|------|------|------|------|------|------|-----------|
| 0 | 0.590 | 0.103 | 0.026 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.100 |
| 1 | 0.590 | 0.077 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.100 |
| 2 | 0.513 | 0.051 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.100 |
| 3 | 0.513 | 0.051 | 0.051 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.100 |
| 4 | 0.641 | 0.205 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.100 |

Predicted=0.2056; measured crossover mu=0.1000 +/-0.0000 (n=5); |pred-measured|=0.1056 vs band 0.0542.

SNR_LAW: predicted=0.2056 measured=0.1000±0.0000 verdict=refuted

## P4 — stress-family invariance: relative (multiplicative) noise grid

Under relative noise K,V *= (1+level*N(0,1)) the RMS_K factor cancels in the SNR ratio, so tolerance should follow the PURE sqrt(head_dim) split (H1 set) and the RMS_K-driven gradations AMONG the hd=64 models should COMPRESS vs the absolute-noise grid. h1_consistent = does the relative-noise tolerant set equal the head_dim>=128 set at some level. spread = max-min retention among hd=64 models; P4 predicts rel < abs. ALLOWED to fail.

| model | head_dim | ret@0.1 | ret@0.2 | tolerant(any) |
|-------|----------|------|------|------------|
| qwen0.5b | 64 | 0.743 | 0.433 | True |
| qwen1.5b | 128 | 0.044 | 0.033 | False |
| tinyllama | 64 | 0.892 | 0.745 | True |
| pythia-410m | 64 | 0.128 | 0.019 | False |
| pythia-1b | 256 | 0.928 | 0.851 | True |
| pythia-1.4b | 128 | 0.625 | 0.190 | False |
| smollm2-360m | 64 | 0.872 | 0.674 | True |

hd=64 spread under relative noise = 0.7254; under absolute noise (grid_v2 retentions) = 0.3480. P4 predicts rel < abs (does NOT hold).

STRESS_INV: family=relative levels=[0.1, 0.2] h1_consistent=False hd64_spread_rel=0.7254 hd64_spread_abs=0.3480
