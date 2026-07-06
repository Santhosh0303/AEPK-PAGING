# PRE-REGISTRATION v2 — Phase 10 step (17): POWERED factorial model grid (the law test) + transition sharpness

> WRITTEN BEFORE THE v2 GRID GPU RUN. Supersedes PREREG_phase10_grid.md for the probe POOL only.
> Everything that could be tuned to win — inclusion gate, H1/H2 pass-lists, arch facts, stress
> level, floor, seeds + seed derivation, transition-fit comparison metric — is UNCHANGED from v1
> and re-fixed here. Honesty spine S9 unchanged. Deterministic. ALLOWED to FAIL (either law wrong,
> transition gradual, or undetermined — reported as-is).

## What changed vs v1 (and ONLY this)
v1 ran on the 105-probe HARD pool (`eval_set_large.LARGE_PROBES`: a factual question after a
1783-char distractor passage + SciQ). Only 2/8 models cleared the N_clean_correct >= 30 gate
(qwen0.5b N_cc=39, qwen1.5b N_cc=50), so the law test was UNDER-POWERED (H1==H2 on 2 points →
verdict `neither`/indistinguishable) and the transition fit was `undetermined` (< 5 points).
v2 changes the PROBE POOL ONLY: the grid now runs on the COMBINED pool
`eval_set_easy.get_combined_probes()` = LARGE_PROBES + ~100 EASY short-factual probes
(one-word capital/color/arithmetic/planet/misc, CW_PROBES style), deduped by prompt (LARGE wins)
and passed through the SAME leakage filter (>= 200 probes, granularity <= 1/200). Easier probes
raise each model's clean-correct subset size so more models clear the UNCHANGED N_cc >= 30 gate.
No metric, threshold, predictor, or gate is altered — only more (easier) probes are added.

## CONTAMINATION DISCLOSURE (verbatim, required)
near-gate directions were observed in the v1 run (pythia-1b/1.4b tolerant at N_cc 25/23);
predictions are architecture-determined and unchanged.

The H1/H2 pass-lists below are DETERMINISTIC FUNCTIONS of each model's published architecture
(head_dim, n_kv from config) — nothing in them is fit to, or informed by, any observed retention.
Seeing which models sat just under the v1 gate does not and cannot change an
architecture-determined prediction; it only motivated adding easier probes to power the test.

## Inclusion policy (FIXED — UNCHANGED from v1, HITL clean-correct conditioning)
Per model, evaluate ONLY the probes the model answers correctly on the CLEAN cache (its own
clean-correct subset). Retention is measured on that subset (clean_acc = 1.0 by construction).
- Answer normalization (FIXED, phase10-local, NOT a Phase 2-5 edit): before `normalized_match`,
  truncate the decoded string at the first "Human:" or "Assistant:" marker (the raw-prompt
  chat-continuation artifact). Nothing else altered.
- **A model is INCLUDED iff N_clean_correct >= 30.** Report N_cc per model. Models with N_cc < 30
  are EXCLUDED and listed with their N.

## Model grid (FIXED — UNCHANGED from v1; all VRAM-verified <= 3.4GB fp16)
| model | family | head_dim | n_kv | KV-width | footprint_GB |
|-------|--------|----------|------|----------|--------------|
| qwen0.5b     | qwen2    | 64  | 2  | 128  | 0.99 |
| qwen1.5b     | qwen2    | 128 | 2  | 256  | 3.09 |
| tinyllama    | llama    | 64  | 4  | 256  | 2.20 |
| pythia-160m  | gpt_neox | 64  | 12 | 768  | 0.33 |
| pythia-410m  | gpt_neox | 64  | 16 | 1024 | 0.81 |
| pythia-1b    | gpt_neox | 256 | 8  | 2048 | 2.03 |
| pythia-1.4b  | gpt_neox | 128 | 16 | 2048 | 2.83 |
| smollm2-360m | llama    | 64  | 5  | 320  | 0.72 |
Crosses head_dim {64,128,256} x KV-width {128..2048} x family {qwen2,llama,gpt_neox}.

## Predictors + pass-lists over THIS grid (FIXED — UNCHANGED definitions)
- **H1 head_dim law**: tolerant <=> head_dim >= 128. Predicted-tolerant =
  {qwen1.5b, pythia-1b, pythia-1.4b}.
- **H2 KV-width law**: tolerant <=> n_kv*head_dim >= 256. Predicted-tolerant =
  {qwen1.5b, tinyllama, pythia-160m, pythia-410m, pythia-1b, pythia-1.4b, smollm2-360m}.
(pass-lists are over the full grid; the runtime verdict compares against the INCLUDED subset.)

## Stress + verdict (FIXED — UNCHANGED from v1)
- quant_noise at LEVEL=0.20 on every KV page. SEEDS=[0,1,2]. FLOOR=0.70.
- Per-page seed derivation: **`sd*1000 + p.layer`** (non-overlapping seed streams).
- retention = mean_over_seeds(corrupt_acc on the clean-correct subset). tolerant <=> retention >= FLOOR.
- `FLOOR_LAW_GRID: predicted_H1=<..> predicted_H2=<..> observed=<..> verdict=<H1|H2|neither|indistinguishable>`
  computed at runtime over the INCLUDED models. verdict=H1 if observed==predicted_H1!=predicted_H2;
  =H2 if observed==predicted_H2!=predicted_H1; =indistinguishable if observed==predicted_H1==predicted_H2;
  else neither. Reported as-is even if neither law holds.

## Transition sharpness (FIXED — UNCHANGED metric)
Ordered table of retention vs redundancy (by head_dim, and by KV-width) across INCLUDED models.
Fit BOTH retention(x): a **logistic** (sharp, 3 params) and a **linear** (gradual, 2 params).
Comparison metric = **AIC** (= 2k + n*ln(SSE/n)), margin **2.0**, pre-registered here BEFORE the run.
`TRANSITION: form=<sharp|gradual|undetermined>`:
- **sharp** iff AIC_logistic + 2.0 < AIC_linear,
- **gradual** iff AIC_linear + 2.0 < AIC_logistic,
- **undetermined** otherwise, OR if fewer than 5 included models, OR if a fit fails to converge.
ALLOWED to be gradual or undetermined — reported as-is.

## Dependency note (RIDER 1, step 16 — reproducibility fix BEFORE this run)
`scipy==1.18.0` was RESTORED to requirements.txt. `phase10_grid.transition_verdict` fits the
logistic via `scipy.optimize.curve_fit` (phase10_grid.py:109) inside a try/except that falls back
to `undetermined` on ImportError. Step 14 had dropped the scipy pin (leaving it importable only in
the maintainer's warm env), so a FRESH environment would silently degrade every TRANSITION verdict
to `undetermined` regardless of the data. Pinning scipy restores a genuine sharp/gradual test on
clean installs. This is a dependency-manifest fix only — no code or metric changed.

## Honesty / determinism
Zero edits to Phase 2-5 source. Reuses phase10_grid harness (quant_noise, dynamiccache_to_pages,
_inject_pages, _decode_under_cache, normalized_match, predict_head_dim/predict_kv_width) + the new
`eval_set_easy` combined pool; CPU tests only for the pool. GPU sweep runs TWICE back-to-back in
ONE background job (RUNTIME ECONOMY), sequential load/unload; per-model rows byte-identical.
Report: results/REPORT_phase10_grid_v2.md.
