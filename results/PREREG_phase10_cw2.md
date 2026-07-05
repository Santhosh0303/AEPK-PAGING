# PRE-REGISTRATION — Phase 10.2 CW-2 (natural-fault confident-wrong retest)

> WRITTEN BEFORE ANY GPU RUN (HITL directive #2). Fault grid, confident-wrong cell
> definition, thresholds, and predicted outcomes are FIXED here. Results collected after
> this commit. A strawman grid = a rigged test; this locks the grid so the verdict cannot
> be steered post hoc. Honesty spine S9 unchanged.

## Why CW-2 exists
Phase 9-CW swept only large-footprint STRUCTURED corruptions ({k_scale, v_scale, v_bias})
on SHORT one-word-answer probes and found NO confident-wrong cell (accuracy damage and
entropy rise were COUPLED → logprob catches everything). That negative is real but its
regime is narrow. CW-2 tests the TWO natural fault classes 9-CW never touched, which are
the strongest remaining candidates for a genuine confident-wrong cell:

1. **fp16 single-bit upsets** — the real hardware fault model on a non-ECC consumer GPU
   (RTX 3050 has no ECC DRAM). A single exponent-bit flip in one fp16 value: tiny footprint,
   can silently corrupt a retrieved fact. Candidate for "answer flips, entropy barely moves."
2. **Needle-page corruption on long context** — corrupt only the page(s) spanning a needle
   fact in a long passage; the model may confidently hallucinate a plausible substitute
   (the documented fluent-confident-wrong failure of long-context retrieval).

## Fault grid (FIXED)
### 10.2a — fp16 exponent bit-flips  → `CW2_BITFLIP`
- **Injector (NEW, numpy, CPU-testable):** `bitflip_fp16(page, n_flips, bit_selector, seed)`.
  View K (and/or V) fp32→fp16→uint16, flip `n_flips` bits chosen from `bit_selector`, cast
  back to fp32. Deterministic given seed. NOT the existing `lossy_tier.bit_flip` (that flips
  bits of int8-QUANTIZED pages; CW-2 needs raw-fp16 exponent flips). Build + CPU unit test
  (round-trip, exact flip count, determinism) BEFORE the GPU sweep.
- **bit_selector ∈ {exponent (fp16 bits 10–14), mantissa (bits 0–9), sign (bit 15)}** — swept.
- **n_flips ∈ {1, 3, 5}** per targeted page.
- **target_k ∈ {1, 3}** most-influential pages by `fp_key_norm_mean` (reuse 9-CW ranking).
- **tensor ∈ {K, V}** — swept (exponent flip in K vs V differ).
- **probes:** the 8 `CW_PROBES` (short factual) — clean_acc must reproduce 9-CW's 1.0 control.
- Grid size = 3 bit_selector × 3 n_flips × 2 target_k × 2 tensor = 36 cells (+ n_flips=0 control).

### 10.2b — needle-page corruption  → `CW2_NEEDLE`
- Reuse `phase9_3_lc.LONG_CONTEXT_PASSAGE` (~300 tok) + a needle QA probe whose answer lives
  at a KNOWN token span in the passage. Verify the needle-token→page mapping by tokenizer
  offsets BEFORE corrupting (verify-before-code); corrupt ONLY the page(s) covering the needle.
- **corruption ∈ {bitflip_fp16 exponent n=1, quant_noise 0.3, v_bias 8.0}** applied to the
  needle page only.
- **probes:** ≥8 needle QA items over the same passage (different asked facts).
- n_flips=0 / clean-needle control REQUIRED (must equal clean baseline).

## Confident-wrong cell definition (FIXED — same as 9-CW, do not relax)
A cell is `confident_wrong == YES` iff ALL THREE hold:
- `dacc ≤ −0.25` (accuracy genuinely broken), AND
- `blind == True`: `dentropy ≤ entropy_bar` where `entropy_bar = std(clean first-answer-token
  entropy across probes)` — i.e. the model's output entropy does NOT rise beyond clean spread,
  so logprob/confidence would NOT flag it, AND
- `flag_rate ≥ 0.5`: the calibrated content-agnostic physics fingerprints (phase9_cw
  `any_physics_flag`, post FLAW-A fix; NOT the degenerate detect.py path — directive #5) DO fire.

Confidence = REAL model output entropy via `token_entropy` on logits forwarded THROUGH the
corrupted cache (`_decode_under_cache`), never the toy readout. Answer token = first generated
token attends the corruption (the 9.1 prefix-bug guard is already in the harness).

## Verdict lines (FIXED, harness-computed runtime f-strings; tests assert LINE EXISTS only)
```
CW2_BITFLIP: confident_wrong_cells=<n> of <total>
CW2_NEEDLE:  confident_wrong_cells=<n> of <total>
```

## Predicted outcome (PRE-REGISTERED — recorded so a surprise is a real finding)
- **Primary prediction: `confident_wrong_cells = 0` for both** — coupling holds (extends 9-CW).
  Rationale: on this 1.5B model, corruption large enough to flip an answer perturbs the
  next-token distribution enough to raise entropy; single-bit exponent flips will either be
  absorbed (no accuracy change) or blow up the value (large entropy rise), not thread the needle.
- **Named upset that WOULD refute it (the one cell to watch):** a single **exponent** bit-flip
  (bit 14/13) in **V** of the **single** most-influential page that flips a needle answer while
  `dentropy ≤ entropy_bar`. If this cell fires → confident-wrong is REAL for KV → novelty regime
  found → escalate (do NOT stop). Reported honestly either way.
- Whatever the count, the sweep is kept in the paper: a 0/72 negative across natural fault
  classes is itself the first systematic confident-wrong test for KV corruption (publishable).

## Honesty / determinism
- Zero edits to Phase 2–5 source. New injector lives in the harness or a new module, unit-tested
  on CPU first. Nothing tuned to force YES; `dacc_min=0.25`, `entropy_bar=std(clean)`, and
  `flag_rate≥0.5` are FIXED here, not adjusted after seeing results.
- GPU sweep run foreground `-v -s`, TWICE, numbers must match (determinism gate).
- STOP-FOR-HITL after both verdict lines land (directive #10), before Phase 10.3 (9.6 live-heal).
