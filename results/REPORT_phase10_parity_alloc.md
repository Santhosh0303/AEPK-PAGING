# REPORT_phase10_parity_alloc.md — Phase 10 step (4) thermodynamic parity allocation

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). The free-energy law (Gibbs attention-mass as a Boltzmann utility, Phase 5) allocates a scarce parity-bit budget. Two policies protect the IDENTICAL critical set (smallest set reaching 0.50 of Gibbs mass) to one erasure each: uniform (a parity block on every group of 4 sibling layer-pages) vs thermo (a block only on groups holding a critical page). Cost = parity bits = blocks x resident_bits(page). Deterministic (no RNG).

| probe | pages | groups | crit_pages | uniform_blocks | thermo_blocks | iso |
|-------|-------|--------|-----------|----------------|---------------|-----|
| What is the capital of France? A | 28 | 7 | 1 | 7 | 1 | True |
| What is the capital of Japan? An | 28 | 7 | 1 | 7 | 1 | True |
| What is the capital of Italy? An | 28 | 7 | 1 | 7 | 1 | True |
| What is the capital of Egypt? An | 28 | 7 | 1 | 7 | 1 | True |
| What is 7 plus 5? Answer with a  | 28 | 7 | 1 | 7 | 1 | True |
| What color is the sky on a clear | 28 | 7 | 1 | 7 | 1 | True |
| What is the capital of Spain? An | 28 | 7 | 1 | 7 | 1 | True |
| What planet do we live on? One w | 28 | 7 | 1 | 7 | 1 | True |

## Interpretation
iso must be True on every row: both policies protect the identical critical set (the CONTROL / plumbing check). thermo_blocks <= uniform_blocks because real attention-mass is concentrated in a minority of layers, so the critical set touches fewer than all groups. Aggregate parity cost: thermo is 0.143x uniform. If mass were diffuse the critical set would span every group and the two costs would coincide (the law buys nothing there) — reported as-is, not tuned away.

PARITY_ALLOC: uniform_cost=10092544 thermo_cost=1441792 iso_protection=True

## kT sensitivity (PREREG amendment v2 — descriptive artifact check)
`gibbs_weights` softmaxes raw key-norms (O(5-50)); at the pre-registered kT=1.0 the distribution is near one-hot, so the critical set can collapse to ~1 page. This section recomputes the mean critical-set size (across the 8 probes, same mass_target=0.50) at swept temperatures. The kT=1.0 HEADLINE above is unchanged; these rows are additive.

| kT label | mean kT | mean crit_pages |
|----------|---------|-----------------|
| kT=1 | 1.000 | 1.00 |
| mean|dmass| | 40.170 | 1.00 |
| std(mass) | 200.783 | 1.00 |

KT_SENSITIVITY: crit_pages=1.00 at kT=1 vs up to 1.00 at higher kT -> temperature-ROBUST.

## Physics-free baseline (PREREG v3 — does the free-energy vocabulary earn its name?)
topk_norm protects the k highest RAW attention_mass pages directly (no Gibbs / softmax / Boltzmann / free-energy framing), with k = |critical set| chosen by the thermo policy (matched protection count). Question: does the free-energy formalism beat this physics-free heuristic anywhere on these probes?

Answer: NO — topk_norm and thermo select the IDENTICAL protected set for the identical parity cost on every probe (softmax is order-preserving in attention_mass, so the smallest set reaching 0.50 of Gibbs mass IS the top-k by raw mass). The thermodynamic vocabulary is DECORATIVE for allocation on this workload; the physics claim rests on steps 5/7, not here.

BASELINE_PARITY: thermo_cost=1441792 topk_cost=1441792 sets_identical=True
