# PRE-REGISTRATION v2 — Phase 10 step (5) / 9.5: redundancy-floor law

> WRITTEN BEFORE THE GPU RUN. Predictors, pass-lists, stress level, floor, seeds, inclusion
> rule, prompt-formatting policy, and verdict line are FIXED here so nothing is tuned to win.
> Supersedes the referenced v1 (never materialized as a file) for the prompt-template and
> inclusion policy; the H1/H2 predictors and their pass-lists are UNCHANGED. Honesty spine S9
> unchanged. ALLOWED to FAIL. Deterministic (no RNG beyond the fixed seed loop).

## Claim under test
KV self-healing tolerance is predicted by KV redundancy. Two candidate predictors that agree
on the Phase-8.5 anchors but DISAGREE on TinyLlama (the discriminator):
- **H1 (PRIMARY) head_dim law**: tolerant <=> head_dim >= 128.
- **H2 (alternative) KV-width law**: tolerant <=> n_kv_heads * head_dim >= 256.
A wrong prediction is the finding, reported as-is. Proof: proofs/redundancy-floor-law.md.

## Models + fixed architecture facts
- qwen0.5b  Qwen/Qwen2.5-0.5B-Instruct — head_dim=64,  n_kv=2, KV-width=128.
- qwen1.5b  Qwen/Qwen2.5-1.5B-Instruct — head_dim=128, n_kv=2, KV-width=256.
- tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0 — head_dim=64, n_kv=4, KV-width=256 (discriminator).

## Predictor pass-lists (FIXED, UNCHANGED from v1)
- **H1 predicted-tolerant = {qwen1.5b}** (only head_dim>=128).
- **H2 predicted-tolerant = {qwen1.5b, tinyllama}** (both KV-width>=256).
TinyLlama is the discriminator: H1 predicts FAIL, H2 predicts PASS.

## Stress + metric (FIXED)
- quant_noise at LEVEL=0.20 on every KV page (Phase 8.5 crossover).
- retention = mean_over_seeds(corrupt_acc) / clean_acc, SEEDS=(0,1,2).
- tolerant <=> retention >= FLOOR=0.70.

## Prompt-formatting policy (FIXED — this amendment, revised after measurement + HITL)
The CW probes are COMPLETION-style ("...Answer in one word:") and scored by a strict one-word
matcher (`eval_set.normalized_match`). A pre-run measurement showed that wrapping the prompt in
the model's chat template DEGRADES clean_acc on this eval for every model tested — qwen0.5b
1.00->0.75, qwen1.5b 1.00->1.00, tinyllama 0.50->0.00 — because chat/instruct models answer
conversationally ("Capital: Paris") and miss the bare one-word target. Therefore:
- **RAW prompting is the primary (and only-used) path** for the floor-law sweep. Applied
  identically to the clean control and every corrupt seed so the comparison is within-format.
- The chat-template path (`apply_chat_template(..., add_generation_prompt=True)`,
  `add_special_tokens=False`) is retained as a DOCUMENTED FALLBACK in `build_ids`
  (`use_chat_template=True`), for future non-completion eval sets; it is NOT invoked here.
This revision is recorded BEFORE the finalized double GPU run. It changes only the input
formatting; predictors, pass-lists, LEVEL, FLOOR, SEEDS, and the inclusion rule are UNCHANGED.

## Inclusion rule (FIXED — this amendment)
A model enters the H1-vs-H2 match comparison only if **clean_acc >= 0.90** (it must actually be
able to do the task before its noise-retention verdict is meaningful). Models below 0.90 are
EXCLUDED from `predicted`/`observed` and listed with their clean_acc. If TinyLlama (the
discriminator) is excluded, the report marks the discriminator UNAVAILABLE and the H1-vs-H2
question undetermined for this run — reported honestly, not forced.

## Verdict line (FIXED)
```
FLOOR_LAW: predicted=<H1 pass-list ∩ included> observed=<included & tolerant> match=<bool>
```
Runtime f-string; tests assert the LINE EXISTS, never a value. GPU run foreground TWICE; rows
byte-identical. Report: results/REPORT_phase10_floor_law.md.

## Honesty / determinism
Zero edits to Phase 2–5 source (reuses quant_noise, dynamiccache_to_pages, _decode_under_cache,
normalized_match, _inject_pages, CW_PROBES). New harness phase10_floor_law.py + CPU tests only.
