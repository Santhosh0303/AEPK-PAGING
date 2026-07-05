# PRE-REGISTRATION — Phase 10.3 / 9.6 MOVE A (live mid-generation self-heal)

> WRITTEN BEFORE THE GPU RUN. Fault (layer/magnitude/step), success criterion, and verdict
> line FIXED here so nothing is tuned to win. Honesty spine S9 unchanged.

## Claim under test
During REAL generation, inject a pre-registered fault into resident KV. Baseline (no heal)
garbles the answer. AEPK: a content-agnostic physics fingerprint LOCALIZES the corrupted
page(s); those locations are fed as ERASURES into the mixed RS decode
(`mixed_decode.rs_mixed_correct`, built on the confirmed `galois decode(erasures=)` API);
the page is restored BIT-EXACT and generation continues with ZERO recompute.

## Fault grid (FIXED)
- Model: Qwen/Qwen2.5-1.5B-Instruct fp16, CUDA (RTX 3050).
- Probes: the 8 short factual `CW_PROBES` (clean_acc must reproduce ~1.0 control).
- Coding: `ReedSolomonCode(t=3)` over each target page's fp16 bytes; store ONLY the 2t=6
  parity symbols per codeword (systematic → redundancy = parity only, genuine channel coding,
  not replication).
- Target: the single most-influential page by `fp_key_norm_mean` (the one that most changes
  the answer), corrupted in K.
- Fault magnitudes swept: n_bit_flips ∈ {0 (CONTROL), 8, 32} on the target page's fp16
  mantissa bits (mantissa → damaging but stays finite, so heal fidelity is what is measured,
  not a trivial NaN). The 0-flip CONTROL row MUST show baseline_acc == aepk_acc == clean_acc.
- Detect→locate: calibrated `phase9_cw` fingerprints flag the corrupted page; a flagged page's
  message symbols are supplied as located erasures. Unflagged-but-corrupt = blind (counts
  against heal, per directive #5).

## Success criterion (FIXED) & verdict line
`recovered == True` iff aepk_acc restores to clean_acc (bit-exact heal → identical logits →
identical answer) AND baseline_acc < clean_acc at that magnitude (fault actually bit).
```
LIVE_HEAL: baseline_acc=<x> aepk_acc=<y> recovered=<bool> decode_mode=erasure
```
ALLOWED to FAIL: if the detector misses the page (blind) or #corrupt > capacity, mixed decode
raises (fail-loud) → recovered=False, reported as-is. A control-row violation (0-flip not
equal to clean) aborts as a plumbing bug (the 9.1 pattern).

## Predicted outcome
recovered=True at both damaging magnitudes: the fingerprint flags a whole-page corruption
(structured, high recall per 9.3c-localization), so it is a pure located-erasure decode →
bit-exact restore → answer identical to clean. decode_mode=erasure. The mixed-decode's 2x
error-regime headroom is proven separately at symbol level in `tests/test_mixed_decode.py`.

## AMENDMENT (2026-07-04, post run-A, documented not hidden)
The originally pre-registered fault (mantissa fp16 bit-flips, n∈{0,8,32}) was **inert** on real
KV: run A measured baseline_acc=1.000 at n=32 (no accuracy damage — mantissa flips absorbed)
and flagged_rate=0.00 (physics fingerprint never fired — broadband-subtle, matches 9.3c-
localization). It therefore exercised NEITHER detection NOR healing → an uninformative null.
Fault changed to **structured key-scale** on the target page K, k_factor ∈ {1.0 (CONTROL,
identity), 2.0, 4.0}: k_scale both breaks accuracy AND is localized by the fingerprint
(9.3c-localization recall 0.83), so it actually tests the detect→locate→erasure-heal path.
This is a corrected fault choice justified by the run-A evidence above, NOT a tuning of any
threshold/constant to move the verdict; the success criterion and verdict line are UNCHANGED.
The inert bit-flip run-A numbers are recorded here as the reason for the change.

## Honesty / determinism
Zero edits to Phase 2–5 source (new files only). Verdict line runtime f-string; test asserts
LINE EXISTS, never value. GPU run foreground twice; data rows must match. Explicit fault=0
control row included (review rule). Reports: `results/REPORT_phase10_liveheal.md`.
