# Redundancy-floor law (Phase 10 step 5 / 9.5)

> Honest derivation SKETCH. Separates PROVEN (in code + measured) from OPEN. The law is a
> PREDICTOR; a wrong prediction is a real result (the law is wrong), reported as-is.

## Claim
A model's KV cache is "self-healing tolerant" — it survives lossy compression / partial
corruption with small task-accuracy loss, so redundancy-based recovery is worth its parity
cost — iff its per-token KV representation carries enough REDUNDANCY. Self-healing headroom
∝ KV redundancy.

## Motivation (the two measured anchor points, Phase 8.5)
| model | head_dim | n_kv_heads | KV-width = n_kv·head_dim | Phase 8.5 verdict |
|-------|----------|-----------|--------------------------|-------------------|
| Qwen2.5-0.5B | 64  | 2 | 128 | FAIL (not on compression frontier at noise=0.2; ΔNLL huge) |
| Qwen2.5-1.5B | 128 | 2 | 256 | PASS (frontier exists; ΔNLL≈+0.35 at 71% savings) |

Phase 8.5 attributed the 0.5B failure to `head_dim=64 (vs 128) → less KV redundancy → higher
sensitivity to quant noise`. That is candidate law **H1**. But the two anchors ALSO differ in
KV-width (128 vs 256), giving candidate law **H2**. On the anchors alone H1 and H2 are
confounded (both separate 0.5B from 1.5B).

## Two candidate predictors (both consistent with the anchors)
- **H1 (head_dim law):** tolerant ⇔ `head_dim ≥ 128`. Rationale: within a head, RoPE-rotated
  key subspace of dimension head_dim; larger head_dim ⇒ more spare dimensions to absorb
  quantization/erasure noise without moving the attention argmax ([coding-bounds]: an MDS-like
  distance grows with symbol dimension; [Gibbs]: softmax over higher-dim keys is smoother).
- **H2 (KV-width law):** tolerant ⇔ per-token KV-width `n_kv·head_dim ≥ 256`. Rationale: total
  redundant real-estate per token (across GQA KV heads) is what a page-level code has to work
  with; GQA replicates each KV head across a group, adding cross-head redundancy.

## Discriminating test (the pre-registered 3rd size)
`TinyLlama-1.1B-Chat-v1.0` (verified 2026-07-04: head_dim=64, n_kv_heads=4, KV-width=256,
22 layers, loads in fp16 on RTX 3050 with 1.22 GB headroom, DynamicCache API identical).
- H1 predicts **FAIL** (head_dim 64 < 128).
- H2 predicts **PASS** (KV-width 256 ≥ 256).
They DISAGREE → TinyLlama separates the two laws. Confound (recorded): TinyLlama is a different
family (Llama arch, different pretraining) — a cross-family point cannot fully isolate
head_dim from arch. Honest, not hidden.

## PROVEN vs OPEN
- PROVEN in code+tests: the retention metric is deterministic; recovery_on is bit-exact by
  construction (Cauchy-MDS, Phase 3) so the model-dependent quantity is DAMAGE tolerance
  (retention under fixed lossy stress), not recovery fidelity.
- OPEN: which predictor is right (H1 vs H2) is decided by the measured TinyLlama verdict; a
  single cross-family point is suggestive, not conclusive; the head_dim vs KV-width vs
  effective-rank vs GQA-group question is not fully identified with 3 points.

## Primary registration
PRIMARY predictor = **H1 (head_dim law)** — the project's own Phase 8.5 stated cause. It
predicts TinyLlama FAILS. If TinyLlama PASSES, H1 is falsified and H2 (KV-width) is the
supported law — that is a real result and is reported as the finding, not tuned away.
