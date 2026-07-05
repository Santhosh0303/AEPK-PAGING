# REPORT_phase10_floor_law.md — Phase 10 step (5) / 9.5 redundancy-floor law

Stress: quant_noise level=0.2 on every KV page. retention = mean_over_seeds(corrupt_acc)/clean_acc, seeds=[0, 1, 2]. tolerant <=> retention >= 0.7. PRIMARY predictor H1 (head_dim law): tolerant <=> head_dim>=128. Alternative H2 (KV-width): tolerant <=> n_kv*head_dim>=256. See proofs/redundancy-floor-law.md.

| model | head_dim | KV-width | clean_acc | retention | tolerant | H1_pred | H2_pred |
|-------|----------|----------|-----------|-----------|----------|---------|---------|
| qwen0.5b | 64 | 128 | 1.000 | 0.167 | False | False | False |
| qwen1.5b | 128 | 256 | 1.000 | 0.917 | True | True | True |
| tinyllama | 64 | 256 | 0.500 | 0.000 | False | False | True |

## Interpretation
match compares the observed tolerant-set against the H1 (head_dim) prediction. The 3rd size TinyLlama (head_dim=64, KV-width=256) is the discriminator: H1 predicts FAIL, H2 predicts PASS. If TinyLlama is tolerant, H1 is falsified and the KV-width/GQA law (H2) is supported — reported as the finding, not tuned away. Caveat: TinyLlama is cross-family (Llama), so a single point is suggestive, not conclusive.
H2 (KV-width) predicted pass = ['qwen1.5b', 'tinyllama'].
CAVEAT: ['tinyllama'] had clean_acc<0.70, so their retention (and thus tolerant verdict) is a weaker signal — a model that barely answers clean gives a noisy ratio. The discriminator TinyLlama sits here (clean_acc 0.50); its FAIL is consistent with H1 but the low clean baseline means H2 is disfavoured, not conclusively refuted. A head_dim-64 / width-256 model with high clean_acc would sharpen this.

## Inclusion (PREREG v2: clean_acc >= 0.90)
EXCLUDED (clean_acc too low to give a meaningful retention verdict): ['tinyllama(0.500)'].
Discriminator TinyLlama is UNAVAILABLE (clean_acc below threshold): H1-vs-H2 is UNDETERMINED this run; match is computed over the included anchors only, reported as-is rather than forced.

FLOOR_LAW: predicted=['qwen1.5b'] observed=['qwen1.5b'] match=True

## ADDENDUM (2026-07-05, step 11 — appended, existing text unmodified)
The v2 inclusion rule (clean_acc >= 0.90) EXCLUDED the discriminator TinyLlama (clean_acc
0.500), so this report's FLOOR_LAW line covers the two Qwen anchors only: H1 is consistent
there, and H1-vs-H2 remains UNDETERMINED from this run. The step-5 factorial grid
(results/REPORT_phase10_grid.md, PREREG_phase10_grid.md) supersedes this report as the
authoritative H1-vs-H2 test: it uses clean-correct conditioning (N_cc >= 30 inclusion) and
the sd*1000+layer seed derivation, and its gate line reads verdict=indistinguishable (every
H1/H2-discriminating model fell below the inclusion gate; exploratory pattern on excluded
rows favors H1, reported as EXPLORATORY ONLY there). The seed derivation in
phase10_floor_law.py was updated to sd*1000+layer in step 10 (code-only hygiene); this
report's numbers were NOT regenerated — the grid supersedes them.
