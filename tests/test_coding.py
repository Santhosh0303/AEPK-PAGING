import numpy as np
import pytest

from aepk_paging.coding import (
    HammingSECDEDCode,
    UncorrectableError,
    encode_erasure_group,
    recover_erasure,
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
