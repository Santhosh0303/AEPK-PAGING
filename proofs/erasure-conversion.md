# Erasure-Conversion — Physics Detection Doubles the Provable Healing Bound

> Proof **sketch**, provisional. States the coding-theory bound the reframed novelty rests on,
> separates PROVEN (pure MDS algebra) from OPEN (what the harness must still verify/measure).
> Cites [coding-bounds] (Singleton/MDS), [Shannon]. Companion to `conserved-invariant-recovery.md`
> and `detector-guarantee.md`.

## Why this proof exists
The original novelty claim ("content-agnostic physics detection uniquely beats the model's own
logprob") is a documented NEGATIVE (Phase 9-CW: accuracy damage and confidence loss are COUPLED for
structured KV corruption on this model → logprob already catches what physics catches). This proof
carries the surviving, provable, unclaimed novelty instead:

**A content-agnostic physics detector that LOCALIZES corruption converts unknown-location errors into
known-location erasures. Under any MDS code this exactly DOUBLES the number of corrupted pages a fixed
parity budget can repair — and removes Reed-Solomon's silent-mis-correction failure mode.**

The value of detection is therefore NOT "alarm uniqueness vs logprob" (logprob signals *that* something
is wrong). It is **localization for repair** (logprob never signals *which* page): logprob cannot drive
healing; a syndrome + physics fingerprint can.

## Setup
Let a parity group be an `[n, k, d]` code over GF(2^8) with `r = n − k` parity symbols. The realizations
in `coding.py`:
- `CauchyReedSolomonGroup(num_parity=r)` — systematic MDS `[k+r, k]`, `erasure_recovery_bound = r`
  (`encode_rs_erasure_group` / `recover_rs_erasure`, verified `coding.py:308,333`).
- `ReedSolomonCode(t)` — full-length `RS(255, 255−2t)`, corrects up to `t` unknown-location symbol
  errors per codeword (`coding.py:392`).

MDS ⇒ minimum distance `d = n − k + 1 = r + 1` ([coding-bounds], Singleton met with equality).

## Theorem (PROVEN — standard MDS algebra)
For an `[n, k, d]` MDS code with `r = d − 1` parity symbols:

```
erasures correctable  (known location)   =  d − 1        =  r
errors    correctable  (unknown location) =  ⌊(d−1)/2⌋   =  ⌊r/2⌋
mixed:  e errors + s erasures correctable ⟺  2e + s ≤ d − 1 = r
```

Proof: the `2e + s ≤ d − 1` bound is the classical MDS decoding guarantee — a bounded-distance decoder
resolves any error/erasure pattern with `2e + s < d`; MDS attains `d = r + 1`. Each *located* symbol
costs 1 unit of the `d−1` budget; each *unlocated* symbol costs 2 (one to find it, one to fix it). ∎

**Corollary (the conversion gain).** Take a purely-error pattern of `m` corrupted pages. A blind
decoder repairs `m ≤ ⌊r/2⌋`. A detector that supplies each corrupted page's location turns all `m`
into erasures, repairing `m ≤ r`. At a fixed parity budget `r`:

```
capacity_with_detection / capacity_without  =  r / ⌊r/2⌋  →  2   (r even; = 2 − 1/⌊r/2⌋ for r odd)
```

Predicted gain **= 2×** (pre-registered per HITL directive #2). This is the `ERASURE_CONVERSION` verdict
the harness will measure.

## Corollary (silent-mis-correction removed)
`coding.py` records the honest failure mode (improvement #1 note, `PROGRESS.md`): RS error decoding
*beyond* `t` may FAIL loud OR **silently return wrong symbols** — no fail-loud guarantee in the error
regime. Erasure decoding does NOT share this: `recover_rs_erasure` inverts a Cauchy submatrix that is
invertible for any `≤ r` erasures (`coding.py:287` guarantee) → bit-exact or explicit `UncorrectableError`,
never a silent miss. Converting errors→erasures therefore moves the repair off the only code path that can
lie. This is a *qualitative* safety gain on top of the 2× *quantitative* capacity gain.

## Effective vs ideal gain (honest ceiling — folds detector recall in)
The 2× is the IDEAL (perfect localization). A real detector has recall `ρ < 1` (Phase 9-CW-localization:
`ρ` up to 1.00 for structured corruption, 0.22–0.41 for broadband `quant_noise`) and FPR `φ` (measured
0.000 at every threshold on fp16 round-trip → clean headroom). Only *flagged* corrupted pages become
erasures; an unflagged corrupted page (`1 − ρ`) stays an unlocated error and still costs 2 units. So the
harness MUST report the **recall-folded effective gain**, not the ideal 2× (HITL directive #5):

```
effective erasures  =  ρ · m          (flagged corrupt pages → located)
residual errors      =  (1 − ρ) · m    (missed → still unlocated, cost 2 each)
correctable ⟺  2·(1−ρ)·m + ρ·m ≤ r    ⟺  m ≤ r / (2 − ρ)
effective_gain(ρ)   =  (r / (2 − ρ)) / ⌊r/2⌋   →   2      as ρ → 1
                                              →   1      as ρ → 0   (no localization, no gain)
```

FPR `φ > 0` would waste budget flagging clean pages as erasures; the measured `φ = 0` means this term is
currently zero but the harness must re-check it, not assume it.

## Proven vs open
- **PROVEN (this document):** the MDS `2e + s ≤ d − 1` bound; the ideal 2× conversion gain; the
  recall-folded `r/(2−ρ)` ceiling; the qualitative removal of silent mis-correction (erasure path is
  Cauchy-invertible, `coding.py:287`).
- **OPEN (harness must verify/measure — steps 1→3 of the path, NOT claimed here):**
  1. **API precondition.** Feeding detector-supplied positions as *erasures* into a mixed error/erasure
     decode is standard for RS, but the installed `galois` decode-with-erasure-locations API must be
     opened and confirmed before the harness codes against it (verify-before-code, AGENTS §1.2). The
     current `ReedSolomonCode.correct_array` takes NO location input — a mixed-mode call is a NEW code
     path to build, not an existing one to reuse. Do not claim the code already realizes conversion.
  2. **Measured effective gain.** `ERASURE_CONVERSION: parity=<r> errors_corrected=<e> erasures_corrected=<k>
     gain=<x>` — harness-computed on real Qwen2.5-1.5B KV, with `ρ` from the calibrated phase9_cw
     fingerprints folded in. Measured gain reported even if below the predicted 2×.
  3. **Regime dependence.** The gain is real only where recall is high (structured corruption). For
     broadband `quant_noise` (`ρ ≈ 0.3`) the effective gain collapses toward 1× — report per corruption
     class, never a single headline number.

## Empirical confirmation on the installed library (read-only probe, 2026-07-03)
`galois.ReedSolomon.decode(codeword, erasures=<bool mask>, errors=True)` — verified signature;
`erasures` is a boolean location mask (the API precondition of OPEN #1 EXISTS). On `RS(15,11)`,
`r = 4` parity, GF(2^8):
- **2 blind errors** (no mask): corrected, `n_err=2`. (2e = 4 ≤ r.)
- **The SAME 4 corrupted symbols as KNOWN erasures** (mask supplied): corrected, `n_err=0`.
  (s = 4 ≤ r.) → located capacity (4) = 2× blind capacity (2). **The 2× is real on the library.**
- **4 blind errors** (no mask): returned the WRONG message but reported `n_err=1` (NOT −1) →
  **silent mis-correction empirically demonstrated.** Supplying the locations (turning the same 4
  into erasures) both corrects them AND avoids the silent lie. This is the qualitative-safety claim,
  confirmed — not merely argued.

This downgrades OPEN #1 from "verify the API exists" to "build the KVPage-level mixed-mode call on
top of the confirmed `erasures=` API" (still a NEW code path, not a claim that the current
`ReedSolomonCode.correct_array` already does it).

## Positioning (vs prior art)
GhostServe / RAID-for-KV obtain erasure locations from a *hardware* failure signal (a device died — the
location is free). In-memory bit-rot / quant corruption emits **no hardware signal**; the location is
unknown → it is an *error*, not an erasure, and prior art cannot repair it at the 2× rate. The
content-agnostic physics detector is what manufactures the missing location signal from the live data
itself. That is the unclaimed synthesis: **detection-as-erasure-locator over the lossy paged KV channel.**
