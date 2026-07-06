# REPORT_phase10_grid.md — Phase 10 step (5) factorial grid: floor-law + transition

Stress: quant_noise LEVEL=0.2 on every KV page, SEEDS=[0, 1, 2] (seed derivation sd*1000+layer). Inclusion: clean-correct conditioning, a model enters iff N_clean_correct >= 30; retention measured on that subset (clean_acc=1.0 there). tolerant <=> retention >= FLOOR=0.7. H1: head_dim>=128; H2: n_kv*head_dim>=256. See PREREG_phase10_grid.md.

| model | family | head_dim | KV-width | N_clean_correct | retention | tolerant | H1_pred | H2_pred | status |
|-------|--------|----------|----------|-----------------|-----------|----------|---------|---------|--------|
| qwen0.5b | qwen2 | 64 | 128 | 127 | 0.045 | False | False | False | included |
| qwen1.5b | qwen2 | 128 | 256 | 151 | 0.914 | True | True | True | included |
| tinyllama | llama | 64 | 256 | 77 | 0.000 | False | False | True | included |
| pythia-160m | gpt_neox | 64 | 768 | 2 | 0.000 | False | False | True | excluded(N_cc=2<30) |
| pythia-410m | gpt_neox | 64 | 1024 | 52 | 0.321 | False | False | True | included |
| pythia-1b | gpt_neox | 256 | 2048 | 74 | 0.856 | True | True | True | included |
| pythia-1.4b | gpt_neox | 128 | 2048 | 72 | 0.861 | True | True | True | included |
| smollm2-360m | llama | 64 | 320 | 91 | 0.348 | False | False | True | included |

## Interpretation
Included models (N_clean_correct >= 30): ['pythia-1.4b', 'pythia-1b', 'pythia-410m', 'qwen0.5b', 'qwen1.5b', 'smollm2-360m', 'tinyllama']. Excluded: [('pythia-160m', 'excluded(N_cc=2<30)')] (reason per entry). The FLOOR_LAW_GRID verdict compares the observed tolerant-set against H1 (head_dim) and H2 (KV-width) predictions over the included models; reported as-is even if neither law holds. The TRANSITION verdict fits retention vs redundancy with a logistic (sharp threshold) and a linear (gradual) curve and compares AIC — sharp = phase-transition-style critical threshold, gradual = smooth law; both are laws, reported as-is.

transition-by-KV-width detail: {'n': 7, 'aic_linear': -13.689, 'aic_logistic': -11.871, 'sse_linear': 0.55926, 'sse_logistic': 0.54495, 'reason': 'AIC difference within margin'}
transition-by-head_dim detail: {'n': 7, 'aic_linear': -15.634, 'aic_logistic': -23.688, 'sse_linear': 0.42358, 'sse_logistic': 0.10074}

FLOOR_LAW_GRID: predicted_H1=['pythia-1.4b', 'pythia-1b', 'qwen1.5b'] predicted_H2=['pythia-1.4b', 'pythia-1b', 'pythia-410m', 'qwen1.5b', 'smollm2-360m', 'tinyllama'] observed=['pythia-1.4b', 'pythia-1b', 'qwen1.5b'] verdict=H1
TRANSITION: form=undetermined (by KV-width; by head_dim=sharp)
