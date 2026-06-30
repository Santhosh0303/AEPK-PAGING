import numpy as np
import pytest

from aepk_paging.coding import (
    HammingSECDEDCode,
    ReedSolomonCode,
    ReedSolomonCodewords,
    UncorrectableError,
    encode_erasure_group,
    encode_rs_erasure_group,
    recover_erasure,
    recover_rs_erasure,
)
from aepk_paging.kv_page import KVPage
from aepk_paging.lossy_tier import bit_flip, quantize_page


def float_page(page_id: int) -> KVPage:
    rng = np.random.default_rng(200 + page_id)
    return KVPage(
        page_id=page_id,
        layer=0,
        token_range=(page_id * 8, page_id * 8 + 8),
        K=rng.normal(loc=0.0, scale=1.0, size=(8, 4)).astype(np.float32),
        V=rng.normal(loc=0.0, scale=1.0, size=(8, 4)).astype(np.float32),
        precision_tag="float32",
        attention_mass=1.0,
    )


def test_erasure_parity_reconstructs_one_evicted_page_bit_exact() -> None:
    pages = [float_page(0), float_page(1), float_page(2)]
    group = encode_erasure_group(pages)

    recovered = recover_erasure(group, missing_page_ids=[1])

    assert group.erasure_recovery_bound == 1
    assert np.array_equal(recovered.K, pages[1].K)
    assert np.array_equal(recovered.V, pages[1].V)


def test_erasure_parity_beyond_single_parity_bound_fails_loud() -> None:
    pages = [float_page(0), float_page(1), float_page(2)]
    group = encode_erasure_group(pages)

    with pytest.raises(UncorrectableError):
        recover_erasure(group, missing_page_ids=[0, 1])


def test_secded_corrects_unknown_location_bit_flip_within_hamming_bound() -> None:
    quantized = quantize_page(float_page(0), bit_width=8)
    code = HammingSECDEDCode()
    encoded = code.encode([quantized])[0]

    corrupted = bit_flip(encoded, p=0.001, seed=0)
    report = code.detect([corrupted])
    corrected = code.correct([corrupted])[quantized.page_id]

    assert code.correction_bound_t == 1
    assert code.detection_bound == 2
    assert report.suspect_ids == (quantized.page_id,)
    assert report.uncorrectable_ids == ()
    assert np.array_equal(corrected.K.values, quantized.K.values)
    assert np.array_equal(corrected.V.values, quantized.V.values)


def test_secded_beyond_hamming_bound_detects_and_flags_without_miscorrection() -> None:
    quantized = quantize_page(float_page(0), bit_width=8)
    code = HammingSECDEDCode()
    encoded = code.encode([quantized])[0]

    corrupted = bit_flip(encoded, p=0.002, seed=42)
    report = code.detect([corrupted])

    assert report.uncorrectable_ids == (quantized.page_id,)
    with pytest.raises(UncorrectableError):
        code.correct([corrupted])


# --- Real / parametric Reed-Solomon (improvement #1) ---


def test_rs_corrects_multi_symbol_errors_within_bound() -> None:
    values = quantize_page(float_page(0), bit_width=8).K.values
    code = ReedSolomonCode(t=3)
    enc = code.encode_array(values)
    cw = enc.codewords.copy()
    cw[0, [5, 50, 200]] ^= np.array([0x7F, 0x33, 0x9A], dtype=np.uint8)  # 3 symbol errors <= t
    recovered, n_errors = code.correct_array(
        ReedSolomonCodewords(cw, enc.original_len, enc.shape, enc.dtype)
    )

    assert code.t == 3 and code.k == 255 - 6
    assert np.array_equal(recovered, values)
    assert n_errors >= 1


def test_rs_beyond_error_bound_never_silently_returns_correct() -> None:
    # Honest property: RS corrects <= t. Beyond t it may FAIL (UncorrectableError)
    # OR silently mis-correct to a wrong codeword -- it never silently returns the
    # TRUE values claiming success. (This is why the error code must be paired with
    # the Phase-4 detection invariants, which catch mis-corrections the code won't.)
    values = quantize_page(float_page(0), bit_width=8).K.values
    code = ReedSolomonCode(t=2)
    enc = code.encode_array(values)
    cw = enc.codewords.copy()
    cw[0, :40] ^= np.uint8(0xFF)  # ~40 symbol errors >> t=2

    raised = False
    recovered = None
    try:
        recovered, _ = code.correct_array(
            ReedSolomonCodewords(cw, enc.original_len, enc.shape, enc.dtype)
        )
    except UncorrectableError:
        raised = True

    assert raised or not np.array_equal(recovered, values)


def test_cauchy_rs_recovers_multiple_evicted_pages_bit_exact() -> None:
    pages = [float_page(i) for i in range(4)]
    group = encode_rs_erasure_group(pages, num_parity=3)

    recovered = recover_rs_erasure(group, missing_page_ids=[0, 2, 3])  # 3 erasures = bound

    assert group.erasure_recovery_bound == 3
    for pid in (0, 2, 3):
        assert np.array_equal(recovered[pid].K, pages[pid].K)
        assert np.array_equal(recovered[pid].V, pages[pid].V)


def test_cauchy_rs_fails_loud_beyond_erasure_bound() -> None:
    pages = [float_page(i) for i in range(4)]
    group = encode_rs_erasure_group(pages, num_parity=2)

    with pytest.raises(UncorrectableError):
        recover_rs_erasure(group, missing_page_ids=[0, 1, 2])  # 3 > 2 parity
