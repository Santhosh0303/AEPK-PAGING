"""Phase 10.3 — KVPage-level MIXED error/erasure Reed-Solomon decode.

Realizes the erasure-conversion reframe (proofs/erasure-conversion.md) on the CONFIRMED
galois API `ReedSolomon.decode(codeword, erasures=<bool mask>, errors=True)`. This is the
NEW code path flagged as OPEN #1 in the proof — it is built HERE, on top of the frozen
Phase-3 `ReedSolomonCode`, NOT by editing `coding.py` (`correct_array` still takes no
positions; this module adds the located-error path beside it).

The MDS bound (proven, coding-bounds Singleton): per RS(255, 255-2t) codeword,
    2*e + s <= 2t          (e = unlocated errors, s = located erasures)
So a content-agnostic physics detector that supplies s located symbol positions lets the
same 2t parity correct up to 2t located erasures (vs only t blind errors) — the 2x gain,
with NO silent mis-correction (erasure positions are inverted deterministically).

Systematic layout (verified 2026-07-04): message symbols occupy the first k=255-2t columns
of each codeword; a flat message-symbol index i maps to (block=i//k, col=i%k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from aepk_paging.coding import ReedSolomonCode, ReedSolomonCodewords, UncorrectableError, _gf


@dataclass(frozen=True)
class MixedDecodeResult:
    recovered: np.ndarray          # same shape/dtype as the original encoded values
    n_blind_errors: int            # unlocated symbol errors the decoder still had to find
    n_located_erasures: int        # symbols the detector located (fed as erasures)


def _erasure_mask(codewords_shape, k: int, located_symbols: Iterable[int]) -> np.ndarray:
    """Boolean mask (num_blocks, 255) marking located message symbols as erasures."""
    mask = np.zeros(codewords_shape, dtype=bool)
    nblocks = codewords_shape[0]
    for i in located_symbols:
        b, col = divmod(int(i), k)
        if 0 <= b < nblocks and 0 <= col < k:      # parity/pad columns are not located here
            mask[b, col] = True
    return mask


def rs_mixed_correct(
    code: ReedSolomonCode,
    corrupted: ReedSolomonCodewords,
    located_symbols: Iterable[int],
) -> MixedDecodeResult:
    """Decode with detector-supplied erasure locations. `located_symbols` are flat indices
    into the ORIGINAL uint8 value stream (message symbols) known to be corrupt. Raises
    UncorrectableError (fail-loud) when a codeword exceeds 2e+s<=2t — never a silent miss
    on the located symbols."""
    GF = _gf()
    mask = _erasure_mask(corrupted.codewords.shape, code.k, located_symbols)
    decoded, n_err = code._rs.decode(
        GF(corrupted.codewords.astype(int)), erasures=mask, errors=True
    )
    n_err = np.atleast_1d(np.asarray(n_err))
    if np.any(n_err < 0):
        raise UncorrectableError("mixed RS decode failed: 2e+s exceeds 2t in some codeword")
    flat = np.asarray(decoded, dtype=np.uint8).reshape(-1)[: corrupted.original_len]
    recovered = flat.view(corrupted.dtype).reshape(corrupted.shape)
    return MixedDecodeResult(
        recovered=recovered,
        n_blind_errors=int(n_err.sum()),
        n_located_erasures=int(mask.sum()),
    )
