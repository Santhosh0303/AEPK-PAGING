# Joint Source–Channel Bound for the KV Cache

> Proof **sketch**, provisional. States the bound the implementation realizes and is honest
> about what is proven vs. open. Cites [Shannon], [coding-bounds].

## Setup
A KV page is (a) lossily quantized — a **source-coding** step at rate `R` bits/symbol with
rate–distortion `D_src(R)` ([Shannon] source theorem; cf. [TurboQuant] within 2.7× the Shannon
limit) — then (b) carried through a **noisy channel** (bit-flip / quant-noise / eviction) and
protected by an **error-correcting code** ([Shannon] channel theorem; bounds [coding-bounds]).

The implementation realizes (b) with `RS(255, 255−2t)` over GF(2^8) (`ReedSolomonCode`) for the
**error** regime and a Cauchy-MDS group (`CauchyReedSolomonGroup`) for the **erasure** regime.

## Claim (concatenation bound)
Let `e` = number of corrupted symbols per codeword, `t = (n−k)/2` the RS capacity, and
`p_fail = P(e > t)` under the channel. End-to-end expected distortion obeys

```
E[D] ≤ D_src(R)            (always — the lossy floor)
     + p_fail · D_loss      (only when the channel exceeds the code)
```

**Proof sketch.** For `e ≤ t`, RS decoding is *exact* (recovers the quantized symbols bit-for-bit;
verified: within-bound tests). So conditioned on `e ≤ t`, channel-induced distortion is **zero** and
total distortion equals the quantization floor `D_src(R)`. The residual term is the probability the
channel exceeds the code times the loss incurred there. For the **erasure** regime the bound is
sharp and deterministic: `≤ r` erasures ⇒ exact recovery (Cauchy MDS ⇒ every `k×k` survivor
submatrix is invertible, [coding-bounds] Singleton); `> r` ⇒ `UncorrectableError` (no silent loss).

## Proven vs. open
- **Proven (in code + tests):** within-bound exactness ⇒ the conditional-zero-channel-distortion
  step; erasure side is tight (MDS).
- **Honest gap:** this is a **concatenation** (separate source then channel code), not the *joint*
  source–channel optimum. Shannon **separation** is asymptotically optimal but has a finite-blocklength
  gap; the tight joint optimum for KV (jointly designing the quantizer and the code) is **open**.
- **Honest gap:** beyond `t`, RS may **mis-correct silently** (no fail-loud guarantee for the error
  regime — see `test_rs_beyond_error_bound_never_silently_returns_correct`). `p_fail · D_loss` is
  therefore a *real* term, and is exactly why the channel code must be paired with the content-agnostic
  detector (see `detector-guarantee.md`).
