# PRE-REGISTRATION v2 — Phase 10.3 / 9.6 live-heal: page-selection CONTROL arm

> WRITTEN BEFORE THE GPU RUN. The v1 headline path (top-influence page corruption →
> detect → erasure-heal, KSCALE_GRID=(1.0, 2.0, 4.0), GROUP_SIZE=4, NUM_PARITY=1,
> CW_PROBES, LIVE_HEAL verdict line) is UNCHANGED. This amendment ADDS a control arm
> answering: is the "top-influence page" (highest fp_key_norm_mean) selection actually
> load-bearing, or would corrupting ANY page break accuracy the same way?

## Control arm (FIXED)
- Same fault: `corrupt_k_scale` at the SAME factors 2.0 and 4.0 (no new magnitudes).
- Target: the LOWEST-fp_key_norm_mean page (order[-1] of the same ranking the headline
  arm uses at order[0]) — the other end of the influence proxy.
- Measure: baseline_acc ONLY (no heal arm — the question is about damage, not recovery).
- Model/probes/decode identical to v1 (Qwen2.5-1.5B fp16, CW_PROBES, _decode_under_cache).

## Fixed prediction & verdict rule (FIXED BEFORE RUN, sign allowed to be wrong)
- If low-mass-page corruption ALSO drives baseline_acc to ~0, the top-influence selection
  is NOT load-bearing (the influence proxy is unvalidated — recorded as a finding, not
  reframed).
- If the low-mass arm stays near clean, the selection is validated.
- Binary rule for the verdict line, evaluated at the STRONGEST factor (4.0):
  `selection_load_bearing = (low_baseline >= clean_acc - 0.125)`
  (0.125 = one probe on the 8-probe CW set; "near clean" = within one probe).
- Verdict line (runtime f-string; tests assert line-exists, never value):
  `HEAL_CONTROL: top_baseline=<a> low_baseline=<b> selection_load_bearing=<bool>`
  where top_baseline = headline-arm baseline_acc at factor 4.0, low_baseline =
  control-arm baseline_acc at factor 4.0.
- EITHER outcome reported as-is.

## Hygiene (this amendment, code-only)
- The stale `n_flips={nf}` print variable in `__main__` renamed to `k_factor` (the fault
  has been k_scale since the v1 amendment; the print name is a leftover).

## Honesty / determinism
Zero edits to Phase 2-5 source. Control arm is a NEW function; the headline path is not
edited. GPU run TWICE; report byte-identical. Report: results/REPORT_phase10_liveheal.md
gains a "Control arm" table + the HEAL_CONTROL line.
