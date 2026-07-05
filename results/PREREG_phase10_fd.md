# PRE-REGISTRATION — Phase 10 step (7) FENCED: fluctuation-dissipation (FD) analogue

> WRITTEN BEFORE ANY CORRUPTION RUN. The clean fluctuation statistic, the DIRECTIONAL
> prediction, the corruption protocol, the comparison metric, and the success/null criteria are
> FIXED here so nothing is tuned to win. Honesty spine S9 unchanged. Deterministic. This is a
> FENCED moonshot: refuted (null) is a real, reported result — not reframed. ALLOWED to fail.

## Hypothesis (FD analogue)
In statistical physics the fluctuation-dissipation theorem says a system's equilibrium
fluctuations predict its linear response to perturbation. Analogue for the KV cache: a layer
whose CLEAN key representations fluctuate more across tokens is more fragile — corrupting it
should cost more task accuracy.

## Clean statistic (FIXED, computed from CLEAN KV only)
- Primary: per-layer **variance of per-token key norms**. For layer-page K (shape T x H x D),
  per-token norm = ||K[t]|| over (H,D); the statistic is Var_t(norm), averaged over the
  clean-correct probes.
- Secondary (reported, not the verdict): per-layer variance of per-token value norms.
- Probe set: `eval_set_large.LARGE_PROBES` on Qwen2.5-1.5B-Instruct. Clean-correct conditioning
  (as in step 5): evaluate on the probes the model answers correctly on the clean cache, using
  the phase10-local answer normalization (truncate at Human:/Assistant:). Report N_clean_correct.

## Directional prediction (FIXED; sign allowed to be wrong)
Higher clean key-norm variance => predicted GREATER retention damage when that layer alone is
corrupted. The direction is fixed here; the observed sign may come out wrong (that is a result).

## Corruption protocol (FIXED)
For each layer L independently: corrupt ONLY that layer's page with quant_noise at LEVEL=0.20,
SEEDS=(0,1,2), non-overlapping seed derivation **sd*1000 + p.layer**; leave all other layers
clean. retention_L = mean_over_seeds(fraction of clean-correct probes still correct).
damage_L = 1 - retention_L. n_layers = number of layer-pages (28 for Qwen2.5-1.5B).

## Comparison + verdict (FIXED)
Spearman rank correlation rho between per-layer clean key-norm variance and per-layer damage.
- **supported** iff rho >= 0.60 (fluctuations forecast vulnerability in the predicted direction).
- **refuted (null: no FD analogue)** iff |rho| < 0.30.
- **undetermined** otherwise (weak 0.30<=|rho|<0.60, or a wrong-sign relation rho<=-0.30 that
  contradicts the fixed direction).
```
FD: spearman=<rho> n_layers=<n> verdict=<supported|refuted|undetermined>
```
Runtime f-string; CPU tests assert the statistic/rank/verdict code, never a hard-coded rho.

## Honesty / determinism
Zero edits to Phase 2-5 source. New harness `phase10_fd.py` (reuses quant_noise,
dynamiccache_to_pages, _inject_pages, _decode_under_cache, normalized_match, LARGE_PROBES,
phase10_grid.normalize_answer) + CPU tests only. GPU run foreground TWICE; per-layer rows
byte-identical. Report: results/REPORT_phase10_fd.md. This prereg is timestamped BEFORE the
corruption run by its creation in the repo prior to executing phase10_fd.__main__.
