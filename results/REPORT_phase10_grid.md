# REPORT_phase10_grid.md — Phase 10 step (5) factorial grid: floor-law + transition

Stress: quant_noise LEVEL=0.2 on every KV page, SEEDS=[0, 1, 2] (seed derivation sd*1000+layer). Inclusion: clean-correct conditioning, a model enters iff N_clean_correct >= 30; retention measured on that subset (clean_acc=1.0 there). tolerant <=> retention >= FLOOR=0.7. H1: head_dim>=128; H2: n_kv*head_dim>=256. See PREREG_phase10_grid.md.

| model | family | head_dim | KV-width | N_clean_correct | retention | tolerant | H1_pred | H2_pred | status |
|-------|--------|----------|----------|-----------------|-----------|----------|---------|---------|--------|
| qwen0.5b | qwen2 | 64 | 128 | 39 | 0.077 | False | False | False | included |
| qwen1.5b | qwen2 | 128 | 256 | 50 | 0.907 | True | True | True | included |
| tinyllama | llama | 64 | 256 | 20 | 0.000 | False | False | True | excluded(N_cc=20<30) |
| pythia-160m | gpt_neox | 64 | 768 | 0 | nan | False | False | True | excluded(N_cc=0<30) |
| pythia-410m | gpt_neox | 64 | 1024 | 11 | 0.152 | False | False | True | excluded(N_cc=11<30) |
| pythia-1b | gpt_neox | 256 | 2048 | 25 | 0.773 | True | True | True | excluded(N_cc=25<30) |
| pythia-1.4b | gpt_neox | 128 | 2048 | 23 | 0.754 | True | True | True | excluded(N_cc=23<30) |
| smollm2-360m | llama | 64 | 320 | 25 | 0.507 | False | False | True | excluded(N_cc=25<30) |

## Interpretation
Included models (N_clean_correct >= 30): ['qwen0.5b', 'qwen1.5b']. Excluded: [('tinyllama', 'excluded(N_cc=20<30)'), ('pythia-160m', 'excluded(N_cc=0<30)'), ('pythia-410m', 'excluded(N_cc=11<30)'), ('pythia-1b', 'excluded(N_cc=25<30)'), ('pythia-1.4b', 'excluded(N_cc=23<30)'), ('smollm2-360m', 'excluded(N_cc=25<30)')] (reason per entry). The FLOOR_LAW_GRID verdict compares the observed tolerant-set against H1 (head_dim) and H2 (KV-width) predictions over the included models; reported as-is even if neither law holds. The TRANSITION verdict fits retention vs redundancy with a logistic (sharp threshold) and a linear (gradual) curve and compares AIC — sharp = phase-transition-style critical threshold, gradual = smooth law; both are laws, reported as-is.

transition-by-KV-width detail: {'n': 2, 'reason': 'fewer than 5 included models'}
transition-by-head_dim detail: {'n': 2, 'reason': 'fewer than 5 included models'}

verdict=indistinguishable: every model where H1 and H2 disagree was excluded by the N_clean_correct >= 30 gate, so both laws predict the SAME tolerant-set on the included models and the data cannot separate them there. EXPLORATORY ONLY (all grid rows, inclusion gate ignored, under-powered — not a pre-registered comparison): H1 consistent on 8/8 rows; H2 consistent on 4/8 rows, contradicted by ['pythia-160m(width-768)', 'pythia-410m(width-1024)', 'smollm2-360m(width-320)', 'tinyllama(width-256)'].

FLOOR_LAW_GRID: predicted_H1=['qwen1.5b'] predicted_H2=['qwen1.5b'] observed=['qwen1.5b'] verdict=indistinguishable
TRANSITION: form=undetermined (by KV-width; by head_dim=undetermined)
