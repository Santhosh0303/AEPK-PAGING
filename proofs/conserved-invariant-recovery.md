# Conserved-Invariant Recovery (the §9 Covariant Law)

> Proof **sketch**, provisional. Separates the *proven* covariant corner from the *open* cross-weight
> frontier. Cites [general-covariance], [RoPE], [Kolmogorov], [rel-rep], [platonic].

## Frame view
A KV page is tensor components in a frame induced by (weights, position field). The **covariant object**
is the attention relation `A = softmax(QKᵀ)`, **invariant under any joint orthogonal rotation**
`Q→QO, K→KO` (since `Q O (K O)ᵀ = Q Oᵀ⁻¹... = QKᵀ` for orthogonal `O`). So `A` is frame-covariant; raw
`K,V` are one coordinate embedding ([general-covariance]). Store the invariant + a frame descriptor;
any concrete `K,V` is a frame-projection.

## Theorem (position covariance — PROVEN, closed-form)
RoPE applies a block-diagonal rotation `R(p)` to `K` at position `p` ([RoPE]). For positions `p, p+Δ`:

```
K_{p+Δ} = R(Δ) · K_p ,      R(Δ) ∈ SO(2)^{d/2} ,      R(Δ)⁻¹ = R(−Δ) = R(Δ)ᵀ.
```

Therefore re-basing a cached page to a new position is an **exact, lossless group action** — recover
`K_{p+Δ}` from `K_p` with **zero distortion**, no recompute. This is the Phase-7 brick
(`7.5`: assert bit-level agreement vs. recompute). Precision re-basing is the analogous (lossy) transcode
already realized by the CODED tier.

## Recovery bound (unified)
Let `T_frame` be the frame map between a page's stored frame and the target frame. Recovery error is

```
‖K_target − T_frame(K_stored)‖  =  0           if T_frame is a known group action (position: exact;
                                                 precision: exact up to the declared quantization),
                                ≤ code residual  for same-frame channel damage (joint-source-channel.md),
                                =  UNBOUNDED      if T_frame is unknown (cross-weight).
```

Same-frame self-healing is thus the special case `T_frame = identity` composed with the channel decoder.

## Proven vs. open
- **Proven:** position covariance is an exact group action (closed form, [RoPE]); the invariance of `A`
  under joint orthogonal frame change (algebra above); same-frame recovery inherits the coding bounds.
- **Open (honest, do NOT claim):** **cross-weight** `T_frame` has **no closed form**. [rel-rep]/[platonic]
  give *evidence* that aligned representations exist and a low-complexity map *may* be learnable, but
  there is **no guarantee** the conserved relational invariant ([Kolmogorov] sufficient statistic of `A`)
  is **sufficient** to regenerate usable target-frame `K,V` keeping generation coherent. Different
  head-count / head-dim / RoPE-base / GQA-grouping make `T_frame` non-dimension-preserving. Establishing
  (or refuting) cross-weight recovery is **post-Phase-7 research** — see THESIS_DOSSIER §9 for the six
  reasons it is out of Phase-7 scope.
