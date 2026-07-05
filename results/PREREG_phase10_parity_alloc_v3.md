# PRE-REGISTRATION AMENDMENT v3 — Phase 10 step (6): physics-free parity baseline

> WRITTEN BEFORE THE GPU RERUN. Adds a physics-free control policy to the parity-allocation
> harness. Does NOT touch the pre-registered kT=1.0 headline path (allocate / critical_set /
> gibbs_weights / the PARITY_ALLOC verdict) or the v2 kT-sensitivity section. Honesty spine S9
> unchanged. Deterministic (no RNG). ALLOWED answer: the policies coincide.

## Question (pre-registered, honest)
Does the free-energy formalism (Gibbs attention-mass as a Boltzmann utility at kT=1) beat a
physics-free heuristic anywhere on these probes for PARITY ALLOCATION? Or is the thermodynamic
vocabulary decorative for this task?

## topk_norm policy (FIXED)
`topk_norm` protects the **k pages with the highest RAW attention_mass**, selected by a plain
magnitude sort — NO softmax, NO temperature, NO Boltzmann/free-energy framing. The count is
matched to the free-energy policy: **k = |critical set|** chosen by the thermo policy on the
same probe. Deterministic tie-break: mass descending, then page index ascending.
Parity placement: a block on every group intersecting the top-k set (same grouping/NUM_PARITY
as thermo). Protection read off placement via `recoverable_critical` (same honest accounting).

## Comparison (FIXED)
Per probe and aggregated across the 8 CW_PROBES:
1. **Parity bits**: topk_cost vs thermo_cost.
2. **Protected-set identity**: the topk-protected set vs the thermo-protected set, AND the
   selected page sets (topk vs critical). `sets_identical` is True only if they match on every
   probe.

## Verdict line (FIXED)
```
BASELINE_PARITY: thermo_cost=<a> topk_cost=<b> sets_identical=<bool>
```
Runtime f-string; the CPU test asserts the LINE EXISTS and exercises the policy via a mutation
test, never a hard-coded cost value.

## Pre-registered expectation + ALLOWED outcome
Because softmax is order-preserving in attention_mass, the smallest set reaching mass_target of
Gibbs mass is exactly the top-|crit| pages by raw mass. So topk_norm and thermo are expected to
select the IDENTICAL set at the IDENTICAL cost — i.e. `thermo_cost == topk_cost` and
`sets_identical == True`. **If so, the report must state that the thermodynamic vocabulary is
DECORATIVE for allocation on this workload, and that the physics claim rests on steps 5/7, not
on parity allocation.** This tie/loss is reported as-is, not tuned away.

## Honesty / determinism
Zero edits to Phase 2-5 source and zero edits to the kT=1 headline path. New functions
`topk_norm_set` / `allocate_topk` + a CPU mutation test only. GPU rerun foreground TWICE; rows
and the BASELINE_PARITY line byte-identical. Report: results/REPORT_phase10_parity_alloc.md.
