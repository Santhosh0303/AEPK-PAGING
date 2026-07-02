# PREREG_phase9_baselines.md — Phase 9.2 Pre-Registration

**Purpose**: Config + success criterion for each competitor LOCKED BEFORE any result is
collected (BUILD_SPEC 9.2 + honesty spine S9 new rule).
Git commit hash of this file establishes the timestamp. Any post-hoc config change
to improve the verdict = honesty violation.

**Written**: 2026-07-02
**Owner**: Sonnet (real-model GPU; per PROGRESS.md Phase 9 standing assignment)

---

## 1. Method verification (paper + repo — not from memory)

### 1.1 KIVI

**Paper**: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
  Liu et al., ICML 2024. arXiv:2402.02750.
  Verified via: https://arxiv.org/abs/2402.02750 (abstract + title confirmed 2026-07-02)

**Repo**: https://github.com/jy-yuan/KIVI (ICML 2024 tag)
  Verified via: repo exists, README confirms k_bits/v_bits/group_size/residual_length
  interface (2026-07-02). quant/new_pack.py confirmed (2026-07-02).

**Method confirmed from code (new_pack.py, fetched 2026-07-02)**:
- K quantization: reshape K [B, nh, T, D] → [B, nh, T/group_size, group_size, D];
  compute min/max across the group_size (token) dimension; INT2 min-max per group×feature.
  Assertion: T % group_size == 0 (fails on T < group_size).
- V quantization: reshape V [B, nh, T, D] → [B, nh, T, D/group_size, group_size];
  compute min/max across the group_size (feature) dimension per token; INT2 min-max.
- Residual: last `residual_length` tokens stored in original dtype (fp16); NOT quantized.
- Custom CUDA kernels: required for throughput ONLY. Quantization math (min-max INT2)
  is reproducible in pure torch/numpy without the CUDA kernels. The CUDA kernels do not
  change the quantization grid, only execution speed.

**Reproducibility verdict: VERIFIED**
  The quantization math is faithfully reproducible in software. The CUDA kernels are for
  throughput; accuracy impact is determined by the quantization scheme alone.

**Short-prompt caveat (honest constraint)**:
  The official config uses group_size=32. For sequences T < 32, the assertion
  T % group_size == 0 fails. For our 100-probe accuracy eval (short prompts, typically
  7–25 tokens), K cannot be quantized. All tokens fall into the residual buffer (fp16).
  This is NOT a rigging issue — KIVI is designed for long-context inference. The
  short-prompt result is a fair measurement of KIVI at its official config on our task.
  To avoid a strawman, we also run a small-group config (see Section 2).

---

### 1.2 KVQuant

**Paper**: "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
  Quantization." Hooper et al., NeurIPS 2024. arXiv:2401.18079.
  Verified via: https://arxiv.org/abs/2401.18079 (2026-07-02).

**Repo**: https://github.com/SqueezeAILab/KVQuant (NeurIPS 2024)
  Verified via: https://github.com/squeezeailab/kvquant (2026-07-02).

**Method from paper**:
  (i) Per-Channel Key Quantization with non-uniform quantization (NUQ) centroids.
  (ii) Pre-RoPE Key Quantization (quantize K before rotary embeddings are applied).
  (iii) Non-Uniform KV Cache Quantization (NUQ): fits quantization centroids via
       calibration dataset — NOT min-max; requires per-channel centroid fitting.
  (iv) Per-Vector Dense-and-Sparse Quantization (optional).

**Why UNVERIFIED on this rig**:
  1. NUQ requires a calibration dataset to fit per-channel quantization centroids.
     Without calibration, the method degrades to uniform quantization — which is NOT
     KVQuant and would be a strawman version (weaker than the paper claims).
  2. Pre-RoPE quantization requires intercepting K before the rotary embedding is
     applied. This requires model-specific hooks not available for Qwen2.5 in
     transformers 5.12.1 without custom patching.
  3. The official repo targets specific LLaMA/Mistral architectures with custom CUDA
     kernels; installation on Windows/Python 3.12 is not feasible.
  4. No faithfully-calibrated KVQuant implementation exists for Qwen2.5-1.5B-Instruct
     that can be run on this rig.

**Reproducibility verdict: UNVERIFIED: KVQuant**
  Excluded from dominance claim per BUILD_SPEC 9.2 and honesty spine S9.
  Dominance verdict must NOT reference KVQuant.

---

### 1.3 SnapKV

**Paper**: "SnapKV: LLM Knows What You are Looking for Before Generation."
  Li et al., 2024. arXiv:2404.14469.
  Verified via: https://arxiv.org/abs/2404.14469 (2026-07-02).

**Repo**: https://github.com/FasterDecoding/SnapKV
  Verified via: repo confirmed (2026-07-02). README states tested with
  transformers==4.37.0, supports Llama/Mistral/Mixtral only.

**Method confirmed from paper**:
  During prefill:
  1. Compute attention weights for the FULL prompt with `output_attentions=True`.
  2. For each head, pool attention weights of the last `window_size` tokens
     (observation window) across the window dimension → importance score per KV position.
     Pooling: apply average pooling (kernel_size = window_size for simplicity; paper
     uses kernel_size=5 pooling along seq dim for local context aggregation).
  3. For each attention head (or KV head group in GQA), select top-k KV positions
     by importance score. keep_ratio = k / (seq_len - window_size).
  4. Reconstruct KV cache with only the top-k positions + the window positions.
  5. Discard all other positions (set to zero or remove from cache).

**Pre-verification result (run 2026-07-02)**:
  `output_attentions=True` with `attn_implementation="eager"` on Qwen2.5-1.5B-Instruct
  (transformers 5.12.1): PASSES. Returns 28 layers, shape [1, 12, seq_len, seq_len].
  Model has 12 query heads (Q-heads), 2 KV heads (GQA ratio 6:1).

**Reproduction plan**:
  - Load Qwen2.5-1.5B-Instruct with `attn_implementation="eager"`.
  - Run forward pass with `output_attentions=True`.
  - For each layer: aggregate attention weights of last `window_size` tokens;
    pool per KV group (6 Q-heads → 1 KV head); select top-k KV positions.
  - Reconstruct KV cache (zero out non-selected positions; window positions kept).
  - Official SnapKV repo not used (Qwen2.5 not supported, wrong transformers version).
    Implementation from paper specification (arXiv:2404.14469 Section 3) only.

**Eager-attention caveat**:
  The default model uses SDPA attention which does NOT return attention weights.
  We must load with `attn_implementation="eager"`, which changes the attention
  computation backend. B0_eager (clean accuracy with eager attention) will be
  measured alongside SnapKV results. If B0_eager ≈ B0_sdpa (0.330 from Phase 9.1),
  the SnapKV accuracy comparison to AEPK is valid. If B0_eager differs significantly,
  it will be flagged as a confound in the report.

**Reproducibility verdict: VERIFIED (with eager-attention caveat above)**
  Algorithm reproducible from paper spec. Pre-verification confirms attention weights
  accessible on this rig. Official repo's Qwen2.5 + transformers 5.x incompatibility
  means independent paper-spec implementation is the correct approach.

---

## 2. Exact configs to run

All methods use the fixed 9.1 accuracy metric:
  - 100 probes (30 EVAL_PROBES + 70 allenai/sciq@validation rows 0-69)
  - Identical to `phase9_accuracy.py::build_extended_eval_set()`
  - normalized_match scoring
  - Manual greedy decode (same as B0/B3 in 9.1 — no `generate(ids, pkv)`)
  - No-damage control row: each method at "no compression" config must equal B0 accuracy

### 2.1 KIVI configs

| Name | k_bits | v_bits | group_size | residual_length | Expected behavior |
|------|--------|--------|------------|-----------------|-------------------|
| KIVI-2-official | 2 | 2 | 32 | 32 | Short prompts: no K compression (T<32); falls back to fp16 |
| KIVI-2-small | 2 | 2 | 4 | 0 | Short prompts: K quantized (T≥4 OK); no residual |
| KIVI-4-official | 4 | 4 | 32 | 32 | Same short-prompt limitation; 4-bit grid, less distortion |
| KIVI-fp16-control | 16 | 16 | — | — | No-op control: must equal B0 accuracy (regression lock) |

Rationale for small-group config: the official config (group_size=32) is designed for
long-context. Running only the official config on short prompts = strawman (KIVI can
never compress short prompts with group_size=32). The small-group config (group_size=4)
measures KIVI's 2-bit quality on our actual eval set length. Both results are reported.

Storage formula per element (excluding quantization scale overhead):
  KIVI-2: 2 bits/elem (quantized) + 2×fp16 per group (scale+zero) amortized
         For K: scale overhead = 2 × 16 / group_size bits/elem = 1 bit/elem (at group_size=32)
         For V: scale overhead = 2 × 16 / group_size bits/elem = 8 bits/elem (at group_size=4)
         Total effective: ~3 bits/elem (group_size=32), ~10 bits/elem (group_size=4) — see harness
  Residual tokens stored in fp16 = 16 bits/elem; included in total bit count.

### 2.2 SnapKV configs

| Name | window_size | keep_ratio | Notes |
|------|-------------|------------|-------|
| SnapKV-r75 | 32 | 0.75 | Keep 75% of non-window positions |
| SnapKV-r50 | 32 | 0.50 | Keep 50% (paper default) |
| SnapKV-r25 | 32 | 0.25 | Keep 25%; more aggressive eviction |
| SnapKV-r100 | — | 1.00 | No-op control: no eviction; must equal B0_eager |

Note: for sequences shorter than window_size=32, ALL tokens are in the observation window;
no eviction occurs. SnapKV-r100 control verifies this falls back to B0 accuracy.

Storage formula: kept_tokens × (K+V) × num_kv_heads × head_dim × 16 bits (fp16, no quantization)
  keep_ratio=0.5 → 8 bits/elem effective (50% tokens × 16 bits each)

### 2.3 KVQuant

UNVERIFIED: KVQuant — excluded. No config run.

---

## 3. ISO-ACCURACY comparison protocol

**Reference point (AEPK from Phase 9.1)**:
  - B0_accuracy (sdpa) = 0.330 (clean, no compression) — 16 bits/elem
  - AEPK B3 at noise=0.2: accuracy = 0.324 ± 0.010 (mean ± 95% CI, 5 seeds)
    labeled "recovery-on, uninterpreted pending 9.3"
  - AEPK storage at noise=0.2: computed by harness for each probe's actual KV size
    (residency-coupled: RESIDENT + CODED + parity blocks, EVICTED = 0)
  - bits/elem for AEPK: derived from total_bits / (2 × seq_len × num_kv_heads × head_dim)

**Protocol**:
  1. Run all competitor configs on the 100-probe accuracy eval set.
  2. Record (accuracy, storage_bits_per_kv_element) for each config.
  3. Plot AEPK B3 points (noise ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.5}) and competitor points
     on a (accuracy, bits/elem) plane.
  4. ISO-ACCURACY crossing: for each accuracy level A, find which method achieves A with
     fewer bits/elem. This is the "fewer bits at same accuracy" question.
  5. The AEPK reference accuracy for the primary comparison is noise=0.2 (NLL crossover
     from Phase 8.2): accuracy=0.324.
  6. Report B0_eager for SnapKV comparisons to detect any eager-vs-sdpa confound.

**Storage normalization**:
  bits_per_kv_elem = total_stored_bits / (2 × seq_len × num_kv_heads × head_dim)
  where 2 = K+V, num_kv_heads=2, head_dim=128 for Qwen2.5-1.5B-Instruct.
  This normalizes across different sequence lengths.
  baseline fp16: 16 bits/elem; KIVI-2 target: ~2 bits/elem (long sequences).

---

## 4. Success criterion

Gate line:
  `BASELINE_DOMINANCE: {DOMINATES_ALL | DOMINATES_SOME | DOMINATED}`

Definitions (VERIFIED competitors only — KVQuant excluded):
  - DOMINATES_ALL: AEPK achieves accuracy ≥ competitor at STRICTLY fewer bits/elem
    for ALL verified competitors at the same accuracy level.
  - DOMINATES_SOME: AEPK achieves accuracy ≥ competitor at fewer bits/elem for
    SOME (not all) verified competitors.
  - DOMINATED: at least one verified competitor achieves the SAME accuracy as AEPK
    with fewer bits/elem.

Per-method sub-verdicts:
  - KIVI: AEPK_vs_KIVI: {AEPK_WINS | KIVI_WINS | TIED | KIVI_NOT_APPLICABLE}
    KIVI_NOT_APPLICABLE if official-config KIVI cannot compress our eval prompts
    (all prompts fall into residual buffer) — honest finding, not a rigged win.
  - SnapKV: AEPK_vs_SNAPKV: {AEPK_WINS | SNAPKV_WINS | TIED}

A strawman/weakened competitor = rigged PASS (forbidden by S9):
  - If KIVI cannot compress due to group_size > prompt length, this is reported honestly
    as "KIVI-not-applicable at short-prompt regime" — not counted as AEPK winning.
  - SnapKV eager-vs-sdpa confound: if B0_eager ≠ B0_sdpa by >5% relative, flag and
    normalize SnapKV results within the eager baseline.

No-damage control requirements (failure = abort, fix harness first):
  - KIVI-fp16-control accuracy == B0_accuracy (±0.01)
  - SnapKV-r100 accuracy == B0_eager_accuracy (±0.01)
  - These are regression locks, same pattern as 9.1 noise=0 control.

---

## 5. Output

Harness: `aepk_paging/harness/phase9_baselines.py`
Test: `tests/test_phase9_baselines.py`
Report: `results/REPORT_phase9_baselines_v2.md`

Report must include:
  - UNVERIFIED: KVQuant (with explanation)
  - AEPK_vs_KIVI sub-verdict
  - AEPK_vs_SNAPKV sub-verdict
  - BASELINE_DOMINANCE gate line
  - No-damage control rows (confirmed pass)
  - AEPK accuracy labeled "recovery-on, uninterpreted pending 9.3"

---

## 6. What is NOT in scope

- Tuning any Phase 2–5 constant to improve AEPK vs competitors.
- Using a weakened/misconfigured version of KIVI or SnapKV to inflate AEPK's advantage.
- Claiming KVQuant dominance (UNVERIFIED, excluded).
- Interpreting AEPK retention~1.0 at noise=0.5 as "AEPK survives noise"
  (uninterpreted pending 9.3; Phase 9.2 accuracy metric is labeled accordingly).
