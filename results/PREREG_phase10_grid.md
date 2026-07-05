# PRE-REGISTRATION — Phase 10 step (5): factorial model grid (the law test) + transition sharpness

> WRITTEN BEFORE THE GRID GPU RUN. Model grid, arch facts, H1/H2 pass-lists, inclusion policy,
> stress level, floor, seeds + seed derivation, and the transition-fit comparison metric are
> FIXED here so nothing is tuned to win. Honesty spine S9 unchanged. Deterministic. ALLOWED to
> FAIL (either law wrong, or transition gradual, or undetermined — reported as-is).

## [REFUTED] Original inclusion rule (clean_acc >= 0.9 on the large probe set)
The step-4 large probe set (`eval_set_large.LARGE_PROBES`, 105 probes) is hard by design
(long-context: a factual question after a 1783-char distractor passage + SciQ). The strongest
grid model, Qwen2.5-1.5B, scores clean_acc = **0.429** on it (measured in
REPORT_phase10_stats.md). No grid model can reach 0.9 -> the rule excludes every model and the
grid is vacuous. The 0.9-on-LARGE rule is therefore REFUTED and abandoned.

### Audit of the 0.429 (why it is not genuine incapability)
Dumped 20 clean-wrong Qwen2.5-1.5B answers. ~10/20 are MATCHER-REJECTIONS, not wrong answers:
- Dominant artifact: the model emits the correct answer then a hallucinated chat turn, e.g.
  "` 4.Human:`", "` the sun.Human:`", "` blood vessels.Human beings`", "` meteorology.Human:`" —
  the strict first-token `normalized_match` rejects the glued "`.Human:`" suffix.
- Multiple-choice wrapping ("` A. Cold`"), singular/plural ("nucleotide" vs "nucleotides"),
  non-English ("湿地" for "wetland"). Genuine errors were the other ~10/20.
True clean capability is ~0.7. `normalized_match` is Phase 2-5 source (NOT edited).

## Inclusion policy (FIXED — HITL, clean-correct conditioning)
Per model, evaluate ONLY the probes the model answers correctly on the CLEAN cache (the
model's own clean-correct subset). Retention is then measured on that subset, where clean_acc
= 1.0 by construction, so the metric is not diluted by probes the model simply cannot do.
- Answer normalization (FIXED, phase10-local, NOT a Phase 2-5 edit): before `normalized_match`,
  truncate the decoded string at the first "Human:" or "Assistant:" marker, removing the
  raw-prompt chat-continuation artifact identified in the audit. Nothing else is altered.
- **A model is INCLUDED iff its clean-correct subset size N_clean_correct >= 30.** Report
  N_clean_correct per model. Models with N_clean_correct < 30 are EXCLUDED (too few clean-correct
  probes for a stable retention estimate) and listed with their N.

## Model grid (FIXED — all VRAM-verified loadable <= 3.4GB fp16, footprint recorded)
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
EXCLUDED before prereg: SmolLM2-1.7B (footprint 3.42 > 3.4GB); OpenELM-450M/1.1B and
gemma-2-2b-it (gated repos, no HF token available).

## Predictors + pass-lists over THIS grid (FIXED, UNCHANGED definitions)
- **H1 head_dim law**: tolerant <=> head_dim >= 128. Predicted-tolerant =
  {qwen1.5b, pythia-1b, pythia-1.4b}.
- **H2 KV-width law**: tolerant <=> n_kv*head_dim >= 256. Predicted-tolerant =
  {qwen1.5b, tinyllama, pythia-160m, pythia-410m, pythia-1b, pythia-1.4b, smollm2-360m}.
(pass-lists are over the full grid; the runtime verdict compares against the INCLUDED subset.)

## Stress + verdict (FIXED)
- quant_noise at LEVEL=0.20 on every KV page. SEEDS=[0,1,2]. FLOOR=0.70.
- Per-page seed derivation: **`sd*1000 + p.layer`** (fixes the `sd + p.layer` collision so seed
  streams never overlap across seeds).
- retention = mean_over_seeds(corrupt_acc on the clean-correct subset). tolerant <=> retention
  >= FLOOR.
- `FLOOR_LAW_GRID: predicted_H1=<..> predicted_H2=<..> observed=<..> verdict=<H1|H2|neither>`
  computed at runtime over the INCLUDED models. verdict=H1 if observed==predicted_H1!=predicted_H2;
  =H2 if observed==predicted_H2!=predicted_H1; else neither (incl. when both match or neither
  matches). Reported as-is even if neither law holds.

## Transition sharpness (FIXED — physics claim, metric pre-registered)
Ordered table of retention vs redundancy (by head_dim, and by KV-width) across INCLUDED models.
Fit BOTH to retention(x): a **logistic** (sharp threshold, 3 params k,x0,L) and a **linear**
(gradual, 2 params). Comparison metric = **AIC** (AIC = 2k + n*ln(SSE/n)); pre-registered
BEFORE the run. Verdict token computed at runtime:
`TRANSITION: form=<sharp|gradual|undetermined>` where
- **sharp** iff AIC_logistic + 2 < AIC_linear (logistic decisively better),
- **gradual** iff AIC_linear + 2 < AIC_logistic,
- **undetermined** otherwise, OR if fewer than 5 included models (too few points to fit a
  3-param logistic) or a fit fails to converge.
Sharp = critical-threshold (phase-transition-style) signature; gradual is still a law. ALLOWED
to be gradual or undetermined — reported as-is either way.

## Honesty / determinism
Zero edits to Phase 2-5 source. New harness `phase10_grid.py` (reuses quant_noise,
dynamiccache_to_pages, _inject_pages, _decode_under_cache, normalized_match, LARGE_PROBES,
predict_head_dim/predict_kv_width) + CPU tests only. GPU sweep foreground TWICE, sequential
load/unload; per-model rows byte-identical. Report: results/REPORT_phase10_grid.md.
