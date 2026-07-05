# PRE-REGISTRATION — Phase 10 step (5) / 9.5 redundancy-floor law

> WRITTEN BEFORE THE GPU RUN. Metric, stress level, floor, seeds, primary predictor, predicted
> pass/fail model lists, and verdict line FIXED here. Nothing tuned to win. ALLOWED to FAIL —
> a wrong prediction is the finding. Honesty spine S9 unchanged. Law derivation:
> `proofs/redundancy-floor-law.md`.

## Metric (FIXED)
For each model: prefill all-but-last prompt token → KV pages (`dynamiccache_to_pages`). Stress
= `lossy_tier.quant_noise(page, level, seed+layer)` applied to EVERY page (lossy-compression
proxy). Forward the last token through the corrupted cache and greedy-decode
(`phase9_cw._decode_under_cache`); score with `eval_set.normalized_match`.
- `clean_acc` = accuracy with clean pages injected (control; must be > 0 for retention to be
  meaningful — reported alongside).
- `retention` = mean over seeds of (corrupt_acc / clean_acc).
- `tolerant` (PASS the floor) ⇔ `retention ≥ FLOOR`.

FIXED knobs: `level = 0.20` (Phase 8.5 crossover), `FLOOR = 0.70`, `seeds = [0, 1, 2]`,
probes = the 8 `CW_PROBES`. Deterministic (seeded quant_noise, do_sample=False).

## Models (FIXED, VRAM+id verified 2026-07-04)
| name | model_id | head_dim | KV-width |
|------|----------|----------|----------|
| qwen0.5b | Qwen/Qwen2.5-0.5B-Instruct | 64  | 128 |
| qwen1.5b | Qwen/Qwen2.5-1.5B-Instruct | 128 | 256 |
| tinyllama | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 64 | 256 |

## Primary predictor (FIXED) = H1 head_dim law: tolerant ⇔ head_dim ≥ 128
- predicted PASS = [qwen1.5b]
- predicted FAIL = [qwen0.5b, tinyllama]

Alternative recorded (H2 KV-width law, tolerant ⇔ width ≥ 256): PASS = [qwen1.5b, tinyllama],
FAIL = [qwen0.5b]. H1 and H2 differ ONLY on tinyllama → it is the discriminator.

## Verdict line (FIXED)
```
FLOOR_LAW: predicted=<H1 pass-list> observed=<measured pass-list> match=<bool>
```
`match` = (observed tolerant-set == H1 predicted tolerant-set). match=False if tinyllama (or
any model) lands opposite to H1 → H1 falsified, H2 (or neither) is the finding. Reported as-is.
Runtime f-string; test asserts the LINE EXISTS, never a value.

## Predicted outcome (pre-registered)
Under H1: observed pass = [qwen1.5b] only → match=True. The strong alternative outcome is
tinyllama PASS (retention ≥ 0.70) → observed pass = [qwen1.5b, tinyllama] → match=False → the
KV-width/GQA law (H2) is supported over head_dim. Either is a real result.

## Honesty / determinism
Zero edits to Phase 2–5 source (new harness `phase10_floor_law.py` + CPU tests only; reuses
`quant_noise`, `dynamiccache_to_pages`, `_decode_under_cache`, `normalized_match`, `CW_PROBES`).
clean_acc control per model. GPU run foreground TWICE; retention rows must match.
Report: `results/REPORT_phase10_floor_law.md`.
