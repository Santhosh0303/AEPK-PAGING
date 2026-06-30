# Content-Agnostic Detector: What Is (and Isn't) Guaranteed

> Proof **sketch**, provisional. States the exact guarantee and its null space honestly.
> Cites [Gibbs], [Kolmogorov].

## Detector
For a page, a fingerprint `g(·)` is a cheap scalar physics statistic of `(K,V)` — attention mass
(top-fraction of the softmax=Boltzmann distribution, [Gibbs]) or the K/V norm ratio. With a stored
clean baseline `g₀`, flag corruption iff `|g(page) − g₀| > τ`.

## Claim (conditional guarantee)
**Detection holds for any corruption whose fingerprint shift exceeds tolerance.** Formally, for
corruption `Δ` mapping `(K,V) → (K',V')`:

```
flagged  ⟺  |g(K',V') − g₀| > τ.
```

So detection is **guaranteed** for the class `{Δ : |g∘Δ − g₀| > τ}` and is **content-agnostic** (it reads
K,V statistics, never token semantics — this is what catches "confident-wrong" corruption that model
logprob cannot: `test_confident_wrong_*`).

## The null space (honest limitation)
There is **no universal** detection guarantee. Any corruption that preserves the fingerprint —
`g(K',V') ≈ g₀` while `(K',V') ≠ (K,V)` — is **invisible** to that detector (its image lies in the
**null space** of `g`). A scalar `g` has a large null space by construction.

**Mitigations (and why they are partial):**
1. **Multiple independent invariants** (attention mass ⟂ norm ratio ⊕ future cross-layer) raise the
   effective rank of the fingerprint map ⇒ shrink the joint null space. Still finite-rank ⇒ still a null
   space.
2. **The channel code** (`coding.py`) catches *coded* corruption the physics fingerprint may miss, and the
   fingerprint catches *uncoded* drift the code is blind to (`test_invariants_flag_uncoded_quant_noise_*`).
   The two are **complementary**, neither complete.

## Proven vs. open
- **Proven (tested):** the conditional guarantee (flag ⟺ shift > τ); clean pages do not false-positive
  (zero deviation at baseline); real-size corruption is flagged.
- **Honest gap:** `τ` is currently hand-set. A derived threshold (from a target false-positive rate via
  the baseline's fingerprint variance) and a measured **ROC** on real-model corruption are open
  (improvement #2 — methodology fixable in sim, real numbers need Phase 7).
- **Honest gap:** characterizing the null space ([Kolmogorov]: corruption that preserves the minimal
  sufficient statistic of the attention relation is *undetectable by any content-agnostic g* and may also
  be *harmless* to output — the "non-problem" question, resolved only on a real model) is open.
