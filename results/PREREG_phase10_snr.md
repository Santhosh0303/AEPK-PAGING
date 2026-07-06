# PREREG_phase10_snr.md — Phase 10 SNR CAMPAIGN (steps 19-21, the mechanism day)

status: PRE-REGISTERED (written BEFORE any GPU measurement)
created: 2026-07-05T15:45:36Z (UTC)
harness: aepk_paging/harness/phase10_snr.py · tests: tests/test_phase10_snr.py
supersedes nothing; new files only; zero edits to Phase 2-5 source (honesty spine S9).

## THEORY (HITL-derived from lossy_tier.py:90-105 — recorded VERBATIM)

> `quant_noise` adds ABSOLUTE Gaussian noise, sigma = level, per K/V component. Attention logit
> l = q.k/sqrt(d). The SIGNAL logit is coherent over d dims: scale = RMS_q*RMS_K*cos(theta)*sqrt(d)
> (grows with sqrt(head_dim)). The NOISE logit q.eps/sqrt(d) is incoherent: std = RMS_q*level
> (flat in d). SNR = sqrt(d)*RMS_K*cos/level -> retention collapses when noise reaches the logit
> gap -> CRITICAL LEVEL LAW: **L_c = C * sqrt(head_dim) * RMS_K** (C global, absorbs alignment +
> softmax scale). Predictions: P1 out-of-sample crossover ratio (step 20 gate); P2 SNR score ranks
> the 7 grid models (exploratory — retentions already seen); P3 per-layer damage anticorrelates
> with per-layer MEAN key RMS (the FD moonshot asked VARIANCE; theory says MAGNITUDE is the
> susceptibility variable — the wrong-sign value-norm rho=-0.4888 becomes right-sign); P4 under
> RELATIVE (multiplicative) noise RMS_K cancels -> tolerance follows pure sqrt(d) (step 21).
> GPU serialization: step-19 measurement (minutes) FIRST -> lock P1 number -> launch step-20 sweep
> background -> build step 21 on CPU -> step-21 sweep background. All spine rules unchanged
> (PREREG before GPU, x2 byte-identical, runtime f-string verdicts, tests assert line-exists only,
> ALLOWED-to-FAIL, zero Phase 2-5 diff — new harness files only).

## MEASUREMENT (step 19a — GPU, deterministic clean prefill, x2 byte-identical)

Per-model per-component RMS_K = RMS over ALL clean K elements, ALL layers, on that model's
v2-grid clean-correct subset (reuse the grid_v2 clean-correct conditioning: normalize_answer +
normalized_match on the COMBINED pool n=215; n_cc must equal the stored grid_v2_run1.json count
for that model — byte-stable determinism check), for all 7 INCLUDED grid models
(qwen0.5b, qwen1.5b, tinyllama, pythia-410m, pythia-1b, pythia-1.4b, smollm2-360m).
Plus per-LAYER mean key RMS and mean value RMS for qwen1.5b on its LARGE_PROBES clean-correct
subset (the SAME subset that produced the fd_v2 per-layer damage, n_cc=50) — for P3.
Clean prefill has no RNG; the RMS is a deterministic function of the cached weights -> the two
runs are byte-identical by construction (gate = byte-identical run1/run2 RMS dumps).

## P2 — SNR RANK (EXPLORATORY, no gate; retentions already seen in grid_v2)

Score s = sqrt(head_dim) * RMS_K per model. Report scores sorted ascending, and whether ONE
threshold on s separates the 4 intolerant (grid_v2 tolerant=false: qwen0.5b, tinyllama,
pythia-410m, smollm2-360m — all head_dim=64) from the 3 tolerant (qwen1.5b hd128, pythia-1.4b
hd128, pythia-1b hd256), and the relative margin of that split vs the head_dim-only split.
margin_vs_hd = (relative s-separation band) / (relative head_dim-separation band). Labeled
EXPLORATORY. Line: `SNR_RANK: scores=<sorted pairs> separable=<bool> margin_vs_hd=<x>`.

## P3 — SNR_FD directional gate (FIXED BEFORE computing rho)

Spearman(per-layer MEAN key RMS [step-19a measurement], per-layer damage [reused verbatim from
REPORT_phase10_fd_v2.md, level=1.0 sweep, 28 layers, n_cc=50]).
Direction predicted: ANTICORRELATION (a layer with larger clean key RMS has a larger signal
logit gap, so the fixed absolute noise is relatively weaker -> LESS damage). Gate:
- supported iff rho <= -0.5
- refuted   iff rho >= -0.2
- else undetermined
Line: `SNR_FD: spearman=<r> n_layers=28 verdict=<supported|refuted|undetermined>`.
Secondary (no gate, reported as-is): Spearman(per-layer mean VALUE RMS, damage) — the -0.4888
value-norm thread that was WRONG-sign for the FD variance hypothesis and is predicted RIGHT-sign
(negative) here.

## P1 — CRITICAL-LEVEL LAW, out-of-sample crossover prediction (step 20 GATE)

Comparison formula (FIXED here; the NUMBER is locked in a timestamped addendum below AFTER the
step-19 RMS_K measurement and BEFORE the step-20 sweep launches):

    predicted qwen0.5b crossover = CAL_CROSSOVER * sqrt(PRED_HEAD_DIM / CAL_HEAD_DIM)
                                              * (RMS_K_q0.5b / RMS_K_q1.5b)

with CAL_CROSSOVER = 0.398 (qwen1.5b LARGE-pool crossover mu, PREREG v3 / REPORT_phase10_stats.md),
PRED_HEAD_DIM = 64 (qwen0.5b), CAL_HEAD_DIM = 128 (qwen1.5b). The global C cancels in the ratio.

Success band (fixed formula): |predicted - measured_mu| <= 0.105 * (predicted / 0.398)
                                                        + measured_CI_half_width.
Step 20 reruns the phase10_stats crossover sweep on qwen0.5b on the SAME pool as calibration
(LARGE_PROBES n=105; qwen0.5b had N_cc=39 >= 30 there — same-pool comparison, no pool confound),
SAME LEVELS/SEEDS/FLOOR/clean-correct conditioning/seed derivation as PREREG v3, fused double-run
byte-identical. Line: `SNR_LAW: predicted=<p> measured=<m>±<ci> verdict=<confirmed|refuted>`.
ALLOWED refuted — a clean falsification of a derived law goes in the manuscript at equal prominence.

## P4 — STRESS-FAMILY INVARIANCE, relative-noise grid (step 21)

phase10-local injector relative_noise(page, level, seed): K,V *= (1 + level*N(0,1)) elementwise
(multiplicative analogue of quant_noise; NOT an edit to the frozen lossy_tier.quant_noise).
Sweep levels {0.1, 0.2} on the 7 included grid models, COMBINED pool, clean-correct conditioning,
seeds [0,1,2], seed derivation sd*1000+layer, double-run byte-identical.
P4 statement (verbatim): under relative noise RMS_K cancels -> the tolerant set should be the pure
head_dim>=128 set at some level; RMS_K-driven gradations among the hd=64 models should COMPRESS
vs the absolute-noise grid. Line: `STRESS_INV: family=relative levels=[0.1,0.2]
h1_consistent=<bool> hd64_spread_rel=<x> hd64_spread_abs=<y>` (spread = max-min retention among
hd=64 models; P4 predicts rel < abs). ALLOWED to fail.

## Honesty spine (unchanged)
PREREG before every GPU run; each GPU job runs the experiment TWICE, row dumps byte-identical;
verdict lines are runtime f-strings (SNR_RANK, SNR_FD, SNR_LAW, STRESS_INV); CPU tests assert the
line EXISTS, never its value; ALLOWED-to-FAIL honored (a refuted derived law is reported at equal
prominence, never reframed); zero diff to Phase 2-5 source (new harness file only).

## P1 LOCK — timestamped ADDENDUM (written AFTER step-19 RMS measurement, BEFORE step-20 launch)

locked: 2026-07-05T21:07:33Z (UTC)
provenance: results/snr_rms_run1.json == results/snr_rms_run2.json (BYTE_IDENTICAL=True);
n_cc for all 7 models == stored grid_v2_run1.json (NCC_MATCH_ALL=True).

Measured (clean, deterministic):
  RMS_K_q0.5b = 9.730026
  RMS_K_q1.5b = 13.320126

Locked P1 prediction (formula from the pre-registered body, evaluated at the measured RMS_K):
  predicted qwen0.5b crossover = 0.398 * sqrt(64/128) * (9.730026 / 13.320126)
                               = 0.205577

Success band half-width (formula fixed pre-measurement) = 0.105*(0.205577/0.398) + measured_CI
= 0.054238 + measured_CI_half_width. Step-20 verdict:
  SNR_LAW: confirmed iff |0.205577 - measured_mu| <= 0.054238 + measured_CI; else refuted.
This number 0.205577 is LOCKED now, before the step-20 qwen0.5b sweep launches (its addendum
timestamp precedes the job-start line in the step-20 log). ALLOWED refuted.

### Step-19 exploratory outcomes (recorded for the audit trail; P2/P3 have no gate on step 19)
- SNR_RANK: scores=[('smollm2-360m', 17.0598), ('tinyllama', 19.7715), ('pythia-1b', 24.4687),
  ('pythia-410m', 52.098), ('qwen0.5b', 77.8402), ('pythia-1.4b', 119.7224),
  ('qwen1.5b', 150.7)] separable=False margin_vs_hd=-1.1981 — the continuous SNR score does NOT
  single-threshold-separate tolerant from intolerant (qwen0.5b is intolerant yet has a high RMS_K,
  so s ranks it above the tolerant pythia-1b). The binary head_dim>=128 split still separates;
  the SNR score does not improve on it here. EXPLORATORY, reported as-is.
- SNR_FD: spearman=0.2995 n_layers=28 verdict=refuted — the P3 anticorrelation prediction is
  REFUTED (weak POSITIVE correlation between per-layer mean key RMS and per-layer damage; the
  predicted sign was negative). A refuted derived prediction is a publishable falsification,
  reported at equal prominence, not reframed. Secondary Spearman(mean value RMS, damage)=-0.4073
  (negative, the predicted sign; no gate).

## Step-21 ADDENDUM — relative-noise grid (written BEFORE the step-21 GPU sweep)

locked: 2026-07-05T21:14:12Z (UTC)
injector: aepk_paging.harness.phase10_snr.relative_noise — K,V *= (1 + level*N(0,1)) elementwise,
phase10-local (frozen lossy_tier.quant_noise untouched). CPU-tested BEFORE GPU: determinism in
seed, zero-level identity, seed independence, multiplicative scale (tests/test_phase10_snr.py).
sweep config (FIXED here): levels [0.1, 0.2] on the 7 INCLUDED grid models (qwen0.5b, qwen1.5b,
tinyllama, pythia-410m, pythia-1b, pythia-1.4b, smollm2-360m), COMBINED pool n=215, clean-correct
conditioning (same as grid), seeds [0,1,2], seed derivation sd*1000+layer, FLOOR=0.70. Fused
double-run, per-model per-level retention dumps byte-identical.
H1 set (head_dim>=128) = {qwen1.5b, pythia-1b, pythia-1.4b}. h1_consistent = does the relative-noise
tolerant set equal H1 at SOME level. hd=64 models = {qwen0.5b, tinyllama, pythia-410m, smollm2-360m};
hd64_spread = max-min retention among them. spread comparison level = 0.2 (grid_v2's absolute-noise
level, for a like-for-like abs-vs-rel comparison); hd64_spread_abs read from stored grid_v2 retentions.
P4 predicts hd64_spread_rel < hd64_spread_abs (RMS_K gradations COMPRESS under relative noise).
Line: STRESS_INV: family=relative levels=[0.1,0.2] h1_consistent=<bool> hd64_spread_rel=<x>
hd64_spread_abs=<y>. ALLOWED to fail.
