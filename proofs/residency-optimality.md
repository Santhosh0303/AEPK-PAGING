# Free-Energy Residency: Optimality Conditions

> Proof **sketch**, provisional. Honest about when the greedy policy is globally optimal vs. only
> locally. Cites [Gibbs], [Landauer], [Kolmogorov], [coding-bounds].

## Objective
Each page `i` may sit in a tier `τ ∈ {RESIDENT, CODED, EVICTED}` with free energy
`F_i(τ) = E_i(τ) − w·kT·S_i(τ)` ([Gibbs]: `S` = Shannon entropy of the softmax=Boltzmann attention
distribution; [Landauer]: EVICTED carries the `erased_bits·kT·ln2` erase floor). Storage bits per
tier are `b_i(RESIDENT) > b_i(CODED) > 0 = b_i(EVICTED)`. Goal:

```
minimize  Σ_i F_i(τ_i)   s.t.   Σ_i b_i(τ_i) ≤ B   and   (eviction ≤ reconstruction capacity)
```

The capacity constraint (Phase 6.2) couples eviction to the erasure bound `r` ([coding-bounds]).

## Claim
The greedy budgeted settling (`ResidencyManager.plan`) — repeatedly take the feasible promotion with
the largest free-energy drop per added bit — is **globally optimal** when the per-page promotion gains
are **submodular** (diminishing returns: gain of EVICTED→CODED ≥ gain of CODED→RESIDENT per bit), and a
**(1−1/e)-approximation** otherwise.

**Proof sketch.** Without the capacity constraint, the problem is a 0/1 multiple-choice knapsack over a
per-page chain EVICTED⊑CODED⊑RESIDENT. If each page's `(−F)` vs. `bits` profile is concave
(submodular promotions), the continuous relaxation has an integral greedy optimum and exchange
arguments give global optimality. The capacity constraint is a **matroid** (partition matroid on parity
groups: ≤ r EVICTED per group); greedy maximization of a submodular gain over a matroid is the classic
`(1−1/e)` guarantee, and is exact when the gain is additive within the budget. **Well-posedness:** the
[Landauer] floor lower-bounds `F` ⇒ the objective is bounded below ⇒ the minimization is well-defined.

## Proven vs. open
- **Proven (tested):** determinism; **monotonicity in attention mass** (raising mass never demotes —
  `test_residency_policy_is_deterministic_and_monotonic_in_attention_mass`); settling free energy is
  **non-increasing** by construction (only accept drops > 0); tier ordering robust across ±50% weights.
- **Honest gap:** **global** optimality requires the submodularity/concavity assumption above. For
  arbitrary weight settings the tier profile need not be concave, so greedy is an **approximation**, not
  proven optimal. The exact min-free-energy assignment under a hard budget is NP-hard in general
  (knapsack) — we claim a well-characterized greedy, not the global optimum.
- **Honest gap:** `S_i` ([Kolmogorov]-adjacent: entropy of the retained relational state) is the
  *attention* entropy, a proxy for the irreducible sufficient statistic — not the true minimal
  sufficient statistic. Tightening this proxy is open.
