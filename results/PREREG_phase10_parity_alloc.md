# PRE-REGISTRATION — Phase 10 step (4): thermodynamic parity allocation

> WRITTEN BEFORE THE GPU RUN. Policy definitions, protection metric, iso-protection
> criterion, cost metric, verdict line, and predicted outcome are FIXED here so nothing
> is tuned to win. Honesty spine S9 unchanged. ALLOWED to FAIL.

## Claim under test
The free-energy residency law (Gibbs attention-mass as a Boltzmann utility weight, Phase 5)
should govern not only tiering but PARITY ALLOCATION: parity protection is a scarce bit
budget, and it should be spent on the pages carrying the most Gibbs attention-mass. A
content-agnostic *uniform* allocation must protect every page equally (it cannot tell which
pages matter); a *thermodynamic* allocation places parity only where the attention-mass is,
and so buys the SAME protection of the critical set for FEWER parity bits.

## Definitions (FIXED)
- Model: Qwen/Qwen2.5-1.5B-Instruct fp16, CUDA (RTX 3050). Probes: the 8 short factual
  `CW_PROBES`. Pages: `dynamiccache_to_pages` returns ONE page per layer (28 pages),
  `attention_mass` = mean per-token key-norm (the adapter's Gibbs proxy, verified
  real_model_adapter.py:50).
- Gibbs weight per page: `w_i = softmax(attention_mass)_i` (Boltzmann over pages, temperature
  kT=1.0 — same convention as `residency.TierCostModel`). Total mass normalized to 1.
- Parity groups: sibling layer-pages chunked by `GROUP_SIZE=4` in layer order (same grouping
  the live-heal harness uses) → ceil(28/4)=7 groups. Each parity block protects its group to
  ONE erasure (`num_parity=1`, Cauchy-MDS, Phase 3).
- Critical set C = the smallest set of pages whose summed Gibbs weight `w` reaches
  `mass_target=0.5` of the total (greedy by descending mass). These are the pages the
  free-energy law says are worth protecting.
- Protection goal (identical for both policies): every page in C must be single-erasure
  recoverable, i.e. sit in a group that carries a parity block.
- **Uniform policy** (content-agnostic): cannot identify C, so to guarantee C is protected it
  must place a parity block on EVERY group → `uniform_blocks = G` (=7).
- **Thermo policy** (Gibbs-ranked): places a parity block only on groups that intersect C →
  `thermo_blocks = |{groups ∩ C}|`.
- Cost metric = parity bits = `blocks × bits_per_parity_page`, where `bits_per_parity_page` =
  `TierCostModel().resident_bits(page)` of a representative real page (all real pages share
  shape). Reported as raw parity BITS so the overhead is honest, not a page count.
- `iso_protection` = (set of C-pages that are recoverable under thermo) == (set under uniform).
  True by construction iff every critical page's group got a block under BOTH policies — the
  runtime check confirms it rather than assuming it.

## Verdict line (FIXED)
```
PARITY_ALLOC: uniform_cost=<a> thermo_cost=<b> iso_protection=<bool>
```
Aggregated across the 8 probes by summing per-probe parity bits (deterministic; no RNG, so no
seed loop — documented, not silently dropped). Runtime f-string; test asserts the LINE EXISTS,
never a value.

## Predicted outcome (pre-registered)
`thermo_cost < uniform_cost` and `iso_protection == True`: attention-mass on real KV is
concentrated (a minority of layers carry most key-norm mass), so C spans FEWER than all 7
groups → thermo skips the low-mass groups' parity while still covering every critical page.
Expected order-of-magnitude: C covers ~2–4 of 7 groups → thermo ≈ 0.3–0.6× uniform bits.

**ALLOWED to FAIL:** if real attention-mass is DIFFUSE (critical set at mass_target=0.5 spreads
across all 7 groups), then `thermo_blocks == G` → `thermo_cost == uniform_cost`, and the law
buys nothing here. That is a real result (the free-energy law does not help when mass is
uniform) and is reported as-is, not tuned away by lowering mass_target.

## Honesty / determinism
Zero edits to Phase 2–5 source (new harness `phase10_parity_alloc.py` + CPU tests only; reuses
`residency.TierCostModel.resident_bits`, `real_model_adapter.dynamiccache_to_pages`,
`phase9_cw.CW_PROBES`). Fault=0 analogue = the CONTROL check that `iso_protection` holds (both
policies protect the identical critical set) — if it ever fails, the run aborts as a coverage
bug. GPU run foreground TWICE; per-probe rows must match byte-identical.
Report: `results/REPORT_phase10_parity_alloc.md`.
