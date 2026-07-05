"""Phase 10.2 CW-2 — raw-fp16 single-bit-upset injector (natural non-ECC DRAM fault model).

NOT `lossy_tier.bit_flip`: that flips bits of int8-QUANTIZED pages. CW-2 needs the real
hardware fault — a bit-flip in the stored fp16 KV value itself. On an RTX 3050 (no ECC DRAM)
this is the physically realistic corruption: a cosmic-ray/charge-leak single-event upset in
one fp16 element. The IEEE-754 half layout (uint16 view):

    bit 15      = sign
    bits 14..10 = exponent (5 bits)   <- an exponent flip is the large-magnitude, high-impact upset
    bits  9..0  = mantissa (10 bits)  <- a mantissa flip is a small perturbation

Deterministic given (n_flips, region, seed, tensor). Phase 2-5 source untouched (new file).
"""

from __future__ import annotations

import numpy as np

from aepk_paging.kv_page import KVPage

_REGION_BITS = {
    "sign": (15,),
    "exponent": (10, 11, 12, 13, 14),
    "mantissa": tuple(range(0, 10)),
}


def _flip_fp16(arr: np.ndarray, n_flips: int, region: str, rng: np.random.Generator) -> np.ndarray:
    """Flip exactly n_flips bits (each at a distinct element, one bit in `region`) of arr.
    arr is any-shape float; it is cast fp32->fp16->uint16, bit-flipped, and returned as fp32."""
    if region not in _REGION_BITS:
        raise ValueError(f"region must be one of {sorted(_REGION_BITS)}")
    half = np.asarray(arr, dtype=np.float32).astype(np.float16)
    flat = half.reshape(-1).view(np.uint16).copy()
    if n_flips < 0:
        raise ValueError("n_flips must be >= 0")
    if n_flips > flat.size:
        raise ValueError("n_flips exceeds element count")
    if n_flips == 0:
        return flat.view(np.float16).reshape(half.shape).astype(np.float32)
    bits = _REGION_BITS[region]
    # distinct elements so flips cannot cancel; one region-bit chosen per element
    elems = rng.choice(flat.size, size=n_flips, replace=False)
    for e in elems:
        b = bits[int(rng.integers(0, len(bits)))]
        flat[e] ^= np.uint16(1 << b)
    return flat.view(np.float16).reshape(half.shape).astype(np.float32)


def bitflip_fp16(page: KVPage, n_flips: int, region: str, seed: int, tensor: str = "K") -> KVPage:
    """Return a new KVPage with n_flips fp16 single-bit upsets in the chosen tensor ('K'|'V').

    The clean baseline is fp32 KV that came from an fp16 model, so the fp32->fp16 recast is
    itself lossless on those values (they are exactly representable); the ONLY change is the
    injected flips. n_flips=0 is therefore a bit-exact no-op control (asserted in tests)."""
    if tensor not in ("K", "V"):
        raise ValueError("tensor must be 'K' or 'V'")
    rng = np.random.default_rng(seed)
    K = np.asarray(page.K, dtype=np.float32)
    V = np.asarray(page.V, dtype=np.float32)
    if tensor == "K":
        K = _flip_fp16(K, n_flips, region, rng)
    else:
        V = _flip_fp16(V, n_flips, region, rng)
    return KVPage(
        page_id=page.page_id,
        layer=page.layer,
        token_range=page.token_range,
        K=K,
        V=V,
        precision_tag=f"{page.precision_tag}+fp16flip[{tensor},{region},n{n_flips}]",
        attention_mass=page.attention_mass,
    )
