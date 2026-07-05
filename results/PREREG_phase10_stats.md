# PRE-REGISTRATION — Phase 10 step (6) / 9.4 statistics

> WRITTEN BEFORE THE GPU RUN. Final config, noise grid, crossover definition, seed count, CI
> method, and verdict line FIXED here. No cherry-pick. Honesty spine S9 unchanged.

## Final config (FIXED)
Model = Qwen/Qwen2.5-1.5B-Instruct fp16 (the tolerant config per step-5 FLOOR_LAW). Probes =
the 8 `CW_PROBES`. Stress = `lossy_tier.quant_noise(page, level, seed+layer)` on every KV page.
Decode = `phase9_cw._decode_under_cache`; score = `eval_set.normalized_match`.

## Crossover definition (FIXED)
For each seed: retention(level) = corrupt_acc(level)/clean_acc over the 8 probes. `crossover` =
the quant_noise level at which retention crosses the FLOOR=0.70 going DOWN, by linear
interpolation between the bracketing grid levels. Censoring rules (reported, not dropped):
- if retention < FLOOR already at the lowest grid level → crossover = lowest level (left-censored);
- if retention ≥ FLOOR at every grid level → crossover = highest level (right-censored).
Noise grid (FIXED) = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]. seeds (FIXED) = [0,1,2,3,4] (n=5).

## Statistic + verdict line (FIXED)
Across the 5 per-seed crossover estimates: μ = mean, ci = t(0.975, n−1)·sd/√n (95% CI half-width;
t=2.776 for n=5). No cherry-pick — all 5 seeds enter μ.
```
STATS: crossover=<μ>±<ci> seeds=<n>
```
Runtime f-string; test asserts the LINE EXISTS, never a value.

## Predicted outcome (pre-registered)
Qwen2.5-1.5B retention was 0.917 at level=0.20 (step 5), so the crossover sits well above 0.2;
expected μ in the ~0.4–0.6 band with a modest CI. A right-censored result (retention never
falls below 0.70 through level 0.8) is allowed and reported as crossover≈0.8 (censored).

## Honesty / determinism
Zero edits to Phase 2–5 source (new harness `phase10_stats.py` + CPU tests; reuses quant_noise,
dynamiccache_to_pages, _inject_pages, _decode_under_cache, normalized_match, CW_PROBES).
Deterministic (seeded quant_noise, do_sample=False). GPU run foreground TWICE; per-seed
crossovers must match. Report: `results/REPORT_phase10_stats.md`.
