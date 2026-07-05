# PRE-REGISTRATION AMENDMENT v2 — Phase 10 step (4): kT-sensitivity of the critical set

> WRITTEN BEFORE THE GPU RERUN. This amendment adds a DESCRIPTIVE robustness section only.
> It does NOT change the pre-registered kT=1.0 headline (verdict line, cost metric, iso
> criterion, predicted outcome) fixed in `PREREG_phase10_parity_alloc.md` — those remain
> exactly as pre-registered. Honesty spine S9 unchanged. Deterministic (no RNG).

## Motivation (artifact check)
`gibbs_weights` applies `softmax(attention_mass)` at kT=1.0, but `attention_mass` is a raw
mean per-token key-norm of order O(5–50). At kT=1.0 the softmax is near one-hot, so the
pre-registered critical set can collapse to a single page (crit_pages≈1). That small critical
set may be a temperature ARTIFACT rather than a property of the mass distribution. This
amendment tests whether the crit-set size is temperature-robust.

## Sweep (FIXED, descriptive only)
For each probe, recompute the critical set (same `mass_target=0.5`, same greedy rule) at three
pre-registered temperatures, derived from that probe's own page masses `m`:
- `kT=1`           — the pre-registered headline temperature (unchanged).
- `mean|Δmass|`    — mean absolute successive difference of the sorted masses (typical mass gap).
- `std(mass)`      — standard deviation of the masses (distribution spread).
Degenerate temperatures (`≤ 0`) are dropped and noted, never silently skipped.

## Reported (FIXED)
A `## kT sensitivity` section in `results/REPORT_phase10_parity_alloc.md` with one row per
temperature label: the temperature label and the mean critical-set size across the 8 probes.
Plus a runtime caveat line stating whether the kT=1.0 crit-set size is temperature-ROBUST
(crit-set size stable across temperatures) or an ARTIFACT (grows materially at higher kT).

## Does NOT change
The `PARITY_ALLOC:` verdict line, `uniform_cost`/`thermo_cost`, `iso_protection`, the
predicted outcome, and the ALLOWED-to-FAIL clause are all unchanged. Sensitivity is additive.
GPU rerun foreground TWICE; per-probe rows AND sensitivity rows must match byte-identical.
