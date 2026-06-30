"""Coding layer for erasure and error regimes.

Two strengths live here:

* Toy / legacy (kept for the foundational tests): one XOR parity block recovers
  one known erasure; SECDED Hamming(8,4) corrects one bit error and detects two
  per 8-bit codeword. Bounds cited from [coding-bounds] (Singleton / Hamming).
* Real / parametric (improvement #1): Reed-Solomon over GF(2^8). A
  Cauchy-Reed-Solomon group recovers up to ``r`` fully EVICTED pages (Singleton /
  MDS, [coding-bounds]); a full-length ``RS(255, 255-2t)`` corrects up to ``t``
  unknown-location symbol errors per codeword. Both fail loud beyond their bound
  (``UncorrectableError``) — never a silent mis-correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.lossy_tier import QuantizedArray, QuantizedPage


class UncorrectableError(RuntimeError):
    pass


# --- Reed-Solomon field (GF(2^8)); imported lazily so the toy tier needs no galois ---
_GF = None


def _gf():
    global _GF
    if _GF is None:
        import galois  # pinned dependency; see pyproject

        _GF = galois.GF(2**8)
    return _GF


@dataclass(frozen=True)
class ErasureParityGroup:
    pages: tuple[KVPage, ...]
    parity_K: np.ndarray
    parity_V: np.ndarray

    @property
    def erasure_recovery_bound(self) -> int:
        return 1


def encode_erasure_group(pages: Sequence[KVPage]) -> ErasureParityGroup:
    if len(pages) < 2:
        raise ValueError("erasure group needs at least two pages")
    _assert_compatible_pages(pages)
    parity_K = _xor_arrays([page.K for page in pages])
    parity_V = _xor_arrays([page.V for page in pages])
    return ErasureParityGroup(pages=tuple(pages), parity_K=parity_K, parity_V=parity_V)


def recover_erasure(group: ErasureParityGroup, missing_page_ids: Iterable[Hashable]) -> KVPage:
    missing_ids = set(missing_page_ids)
    if len(missing_ids) != 1:
        raise UncorrectableError("one parity block recovers exactly one known erasure")
    missing_id = next(iter(missing_ids))
    pages_by_id = {page.page_id: page for page in group.pages}
    if missing_id not in pages_by_id:
        raise KeyError("missing page is not in this erasure group")
    survivors = [page for page in group.pages if page.page_id != missing_id]
    missing_template = pages_by_id[missing_id]
    recovered_K = _xor_arrays([group.parity_K, *(page.K for page in survivors)]).view(
        missing_template.K.dtype
    )
    recovered_V = _xor_arrays([group.parity_V, *(page.V for page in survivors)]).view(
        missing_template.V.dtype
    )
    return KVPage(
        page_id=missing_template.page_id,
        layer=missing_template.layer,
        token_range=missing_template.token_range,
        K=recovered_K.reshape(missing_template.K.shape),
        V=recovered_V.reshape(missing_template.V.shape),
        precision_tag=missing_template.precision_tag,
        attention_mass=missing_template.attention_mass,
    )


@dataclass(frozen=True)
class SyndromeReport:
    suspect_ids: tuple[Hashable, ...]
    uncorrectable_ids: tuple[Hashable, ...]


class HammingSECDEDCode:
    correction_bound_t = 1
    detection_bound = 2

    def encode(self, pages: Sequence[QuantizedPage]) -> tuple[QuantizedPage, ...]:
        return tuple(_encode_quantized_page(page) for page in pages)

    def detect(self, pages: Sequence[QuantizedPage]) -> SyndromeReport:
        suspect_ids: list[Hashable] = []
        uncorrectable_ids: list[Hashable] = []
        for page in pages:
            report = _decode_quantized_page(page)
            if report.corrected_error_count:
                suspect_ids.append(page.page_id)
            if report.uncorrectable:
                uncorrectable_ids.append(page.page_id)
        return SyndromeReport(tuple(suspect_ids), tuple(uncorrectable_ids))

    def correct(self, pages: Sequence[QuantizedPage]) -> dict[Hashable, QuantizedPage]:
        corrected: dict[Hashable, QuantizedPage] = {}
        for page in pages:
            report = _decode_quantized_page(page)
            if report.uncorrectable:
                raise UncorrectableError("detected beyond-bound bit corruption")
            corrected[page.page_id] = QuantizedPage(
                page_id=page.page_id,
                layer=page.layer,
                token_range=page.token_range,
                K=QuantizedArray(
                    values=report.K_values,
                    scale=page.K.scale,
                    bit_width=page.K.bit_width,
                    mse=page.K.mse,
                ),
                V=QuantizedArray(
                    values=report.V_values,
                    scale=page.V.scale,
                    bit_width=page.V.bit_width,
                    mse=page.V.mse,
                ),
                precision_tag=page.precision_tag.replace("secded-", "", 1),
                attention_mass=page.attention_mass,
                distortion_mse=page.distortion_mse,
            )
        return corrected


@dataclass(frozen=True)
class _DecodeReport:
    K_values: np.ndarray
    V_values: np.ndarray
    corrected_error_count: int
    uncorrectable: bool


def _assert_compatible_pages(pages: Sequence[KVPage]) -> None:
    first = pages[0]
    for page in pages[1:]:
        if page.K.shape != first.K.shape or page.V.shape != first.V.shape:
            raise ValueError("all pages in an erasure group must share K/V shapes")
        if page.K.dtype != first.K.dtype or page.V.dtype != first.V.dtype:
            raise ValueError("all pages in an erasure group must share K/V dtypes")


def _xor_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    result = np.ascontiguousarray(arrays[0]).view(np.uint8).copy()
    for array in arrays[1:]:
        result = np.bitwise_xor(result, np.ascontiguousarray(array).view(np.uint8))
    return result


def _encode_quantized_page(page: QuantizedPage) -> QuantizedPage:
    return QuantizedPage(
        page_id=page.page_id,
        layer=page.layer,
        token_range=page.token_range,
        K=QuantizedArray(
            values=_encode_bytes(page.K.values),
            scale=page.K.scale,
            bit_width=page.K.bit_width,
            mse=page.K.mse,
        ),
        V=QuantizedArray(
            values=_encode_bytes(page.V.values),
            scale=page.V.scale,
            bit_width=page.V.bit_width,
            mse=page.V.mse,
        ),
        precision_tag=f"secded-{page.precision_tag}",
        attention_mass=page.attention_mass,
        distortion_mse=page.distortion_mse,
    )


def _decode_quantized_page(page: QuantizedPage) -> _DecodeReport:
    K_values, K_errors, K_uncorrectable = _decode_bytes(page.K.values)
    V_values, V_errors, V_uncorrectable = _decode_bytes(page.V.values)
    return _DecodeReport(
        K_values=K_values,
        V_values=V_values,
        corrected_error_count=K_errors + V_errors,
        uncorrectable=K_uncorrectable or V_uncorrectable,
    )


def _encode_bytes(values: np.ndarray) -> np.ndarray:
    byte_values = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
    codewords = []
    for byte in byte_values:
        low = int(byte) & 0x0F
        high = (int(byte) >> 4) & 0x0F
        codewords.append(_encode_nibble(low))
        codewords.append(_encode_nibble(high))
    return np.array(codewords, dtype=np.uint8).reshape(values.shape + (2,))


def _decode_bytes(encoded: np.ndarray) -> tuple[np.ndarray, int, bool]:
    original_shape = encoded.shape[:-1]
    codewords = np.ascontiguousarray(encoded).view(np.uint8).reshape(-1)
    if len(codewords) % 2 != 0:
        return np.array([], dtype=np.int8), 0, True
    decoded = []
    corrected_errors = 0
    uncorrectable = False
    for index in range(0, len(codewords), 2):
        low, low_errors, low_bad = _decode_codeword(int(codewords[index]))
        high, high_errors, high_bad = _decode_codeword(int(codewords[index + 1]))
        decoded.append(np.uint8(low | (high << 4)))
        corrected_errors += low_errors + high_errors
        uncorrectable = uncorrectable or low_bad or high_bad
    return np.array(decoded, dtype=np.uint8).view(np.int8).reshape(original_shape), corrected_errors, uncorrectable


def _encode_nibble(nibble: int) -> np.uint8:
    d1 = (nibble >> 0) & 1
    d2 = (nibble >> 1) & 1
    d3 = (nibble >> 2) & 1
    d4 = (nibble >> 3) & 1
    bits = [0] * 8
    bits[2] = d1
    bits[4] = d2
    bits[5] = d3
    bits[6] = d4
    bits[0] = bits[2] ^ bits[4] ^ bits[6]
    bits[1] = bits[2] ^ bits[5] ^ bits[6]
    bits[3] = bits[4] ^ bits[5] ^ bits[6]
    bits[7] = bits[0] ^ bits[1] ^ bits[2] ^ bits[3] ^ bits[4] ^ bits[5] ^ bits[6]
    return np.uint8(sum(bit << offset for offset, bit in enumerate(bits)))


def _decode_codeword(codeword: int) -> tuple[int, int, bool]:
    bits = [(codeword >> offset) & 1 for offset in range(8)]
    s1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6]
    s2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6]
    s4 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6]
    syndrome = s1 | (s2 << 1) | (s4 << 2)
    overall = bits[0] ^ bits[1] ^ bits[2] ^ bits[3] ^ bits[4] ^ bits[5] ^ bits[6] ^ bits[7]
    corrected_errors = 0
    if syndrome and overall:
        bits[syndrome - 1] ^= 1
        corrected_errors = 1
    elif not syndrome and overall:
        bits[7] ^= 1
        corrected_errors = 1
    elif syndrome and not overall:
        return 0, 0, True
    nibble = bits[2] | (bits[4] << 1) | (bits[5] << 2) | (bits[6] << 3)
    return nibble, corrected_errors, False


# ============================================================================
# Real / parametric Reed-Solomon (improvement #1) — replaces the toy strength.
# ============================================================================


@dataclass(frozen=True)
class CauchyReedSolomonGroup:
    """Systematic MDS erasure group over GF(2^8): recover up to ``num_parity``
    fully EVICTED pages from the survivors + parity ([coding-bounds] Singleton/MDS)."""

    pages: tuple[KVPage, ...]
    parity_bytes: np.ndarray            # (num_parity, byte_len) uint8
    k_byte_len: int                     # bytes in the K half of each page row
    page_ids: tuple[Hashable, ...]
    num_parity: int

    @property
    def erasure_recovery_bound(self) -> int:
        return self.num_parity


def _cauchy_generator(k: int, r: int):
    """Systematic (k+r, k) MDS generator [I_k ; Cauchy(r,k)] over GF(2^8).
    Cauchy guarantees every square submatrix is invertible -> any <= r erasures recover."""
    GF = _gf()
    if k + r > 255:
        raise ValueError("Cauchy-RS over GF(2^8) needs k + num_parity <= 255")
    xs = GF(np.arange(1, 1 + r, dtype=int))
    ys = GF(np.arange(1 + r, 1 + r + k, dtype=int))
    cauchy = GF(np.zeros((r, k), dtype=int))
    for i in range(r):
        for j in range(k):
            cauchy[i, j] = GF(1) / (xs[i] + ys[j])
    return np.vstack([GF(np.eye(k, dtype=int)), cauchy])


def _page_row_bytes(page: KVPage) -> tuple[np.ndarray, int]:
    k_bytes = np.ascontiguousarray(page.K).view(np.uint8).reshape(-1)
    v_bytes = np.ascontiguousarray(page.V).view(np.uint8).reshape(-1)
    return np.concatenate([k_bytes, v_bytes]), int(k_bytes.size)


def encode_rs_erasure_group(pages: Sequence[KVPage], num_parity: int) -> CauchyReedSolomonGroup:
    if len(pages) < 1:
        raise ValueError("erasure group needs at least one page")
    if num_parity < 1:
        raise ValueError("num_parity must be >= 1")
    _assert_compatible_pages(list(pages))
    GF = _gf()
    rows = []
    k_byte_len = None
    for page in pages:
        row, klen = _page_row_bytes(page)
        rows.append(row)
        k_byte_len = klen if k_byte_len is None else k_byte_len
    data = GF(np.array(rows, dtype=int))                       # (k, L)
    generator = _cauchy_generator(len(pages), num_parity)
    parity = generator[len(pages):, :] @ data                  # (r, L)
    return CauchyReedSolomonGroup(
        pages=tuple(pages),
        parity_bytes=np.asarray(parity, dtype=np.uint8),
        k_byte_len=int(k_byte_len),
        page_ids=tuple(page.page_id for page in pages),
        num_parity=num_parity,
    )


def recover_rs_erasure(group: CauchyReedSolomonGroup, missing_page_ids: Iterable[Hashable]) -> dict[Hashable, KVPage]:
    missing = [pid for pid in group.page_ids if pid in set(missing_page_ids)]
    if len(missing) > group.num_parity:
        raise UncorrectableError(
            f"{len(missing)} erasures exceed parity bound {group.num_parity}"
        )
    if not missing:
        return {}
    GF = _gf()
    k = len(group.pages)
    generator = _cauchy_generator(k, group.num_parity)
    missing_set = set(missing)
    survivor_rows: list[int] = []
    survivor_bytes: list[np.ndarray] = []
    for idx, page in enumerate(group.pages):                   # surviving data rows
        if page.page_id not in missing_set:
            row, _ = _page_row_bytes(page)
            survivor_rows.append(idx)
            survivor_bytes.append(row)
    for j in range(group.num_parity):                          # parity rows
        survivor_rows.append(k + j)
        survivor_bytes.append(group.parity_bytes[j])
        if len(survivor_rows) == k:
            break
    if len(survivor_rows) < k:
        raise UncorrectableError("not enough survivors to invert the generator")
    sub = generator[survivor_rows[:k], :]
    recv = GF(np.array(survivor_bytes[:k], dtype=int))
    data = np.linalg.inv(sub) @ recv                            # (k, L) recovered
    recovered: dict[Hashable, KVPage] = {}
    for idx, page in enumerate(group.pages):
        if page.page_id in missing_set:
            row = np.asarray(data[idx], dtype=np.uint8)
            recovered[page.page_id] = _row_to_page(row, page, group.k_byte_len)
    return recovered


def _row_to_page(row: np.ndarray, template: KVPage, k_byte_len: int) -> KVPage:
    k_flat = row[:k_byte_len].view(template.K.dtype).reshape(template.K.shape)
    v_flat = row[k_byte_len:].view(template.V.dtype).reshape(template.V.shape)
    return KVPage(
        page_id=template.page_id,
        layer=template.layer,
        token_range=template.token_range,
        K=k_flat,
        V=v_flat,
        precision_tag=template.precision_tag,
        attention_mass=template.attention_mass,
    )


@dataclass(frozen=True)
class ReedSolomonCodewords:
    codewords: np.ndarray               # (num_blocks, 255) uint8
    original_len: int                   # symbols before padding
    shape: tuple[int, ...]
    dtype: np.dtype


class ReedSolomonCode:
    """Full-length RS(255, 255-2t) over GF(2^8): corrects up to ``t`` unknown-location
    symbol errors per codeword ([coding-bounds]); beyond -> UncorrectableError."""

    def __init__(self, t: int = 2) -> None:
        if t < 1 or 2 * t >= 255:
            raise ValueError("t must satisfy 1 <= t and 2t < 255")
        import galois

        self.t = t
        self.n = 255
        self.k = 255 - 2 * t
        self._rs = galois.ReedSolomon(self.n, self.k, field=_gf())

    def encode_array(self, values: np.ndarray) -> ReedSolomonCodewords:
        GF = _gf()
        flat = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
        original_len = int(flat.size)
        pad = (-original_len) % self.k
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
        messages = GF(padded.reshape(-1, self.k).astype(int))
        codewords = np.asarray(self._rs.encode(messages), dtype=np.uint8)
        return ReedSolomonCodewords(codewords, original_len, values.shape, values.dtype)

    def correct_array(self, corrupted: ReedSolomonCodewords) -> tuple[np.ndarray, int]:
        GF = _gf()
        decoded, n_errors = self._rs.decode(
            GF(corrupted.codewords.astype(int)), errors=True
        )
        n_errors = np.atleast_1d(np.asarray(n_errors))
        if np.any(n_errors < 0):
            raise UncorrectableError("RS decode failed: errors exceed t per codeword")
        flat = np.asarray(decoded, dtype=np.uint8).reshape(-1)[: corrupted.original_len]
        recovered = flat.view(corrupted.dtype).reshape(corrupted.shape)
        return recovered, int(n_errors.sum())
