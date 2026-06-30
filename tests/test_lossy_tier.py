import numpy as np
import pytest

from aepk_paging.kv_page import KVPage, ResidencyTier
from aepk_paging.lossy_tier import (
    Channel,
    MissingPageError,
    bit_flip,
    page_mse,
    quant_noise,
    quantize_page,
)


def normal_float_page(page_id: int = 0) -> KVPage:
    rng = np.random.default_rng(100 + page_id)
    return KVPage(
        page_id=page_id,
        layer=0,
        token_range=(page_id * 16, page_id * 16 + 16),
        K=rng.normal(loc=0.0, scale=1.0, size=(16, 8)).astype(np.float32),
        V=rng.normal(loc=0.0, scale=1.0, size=(16, 8)).astype(np.float32),
        precision_tag="float32",
        attention_mass=1.0,
    )


def test_int8_and_int4_quantization_record_real_float_distortion() -> None:
    page = normal_float_page()

    int8_page = quantize_page(page, bit_width=8)
    int4_page = quantize_page(page, bit_width=4)

    assert int8_page.precision_tag == "int8"
    assert int4_page.precision_tag == "int4"
    assert int8_page.distortion_mse > 0.0
    assert int4_page.distortion_mse > int8_page.distortion_mse
    assert page_mse(page, int8_page.dequantize()) == pytest.approx(int8_page.distortion_mse)


def test_bit_flip_is_reproducible_under_fixed_seed() -> None:
    quantized = quantize_page(normal_float_page(), bit_width=8)

    first = bit_flip(quantized, p=0.2, seed=7)
    second = bit_flip(quantized, p=0.2, seed=7)

    assert np.array_equal(first.K.values, second.K.values)
    assert np.array_equal(first.V.values, second.V.values)


def test_quant_noise_is_reproducible_under_fixed_seed() -> None:
    page = normal_float_page()

    first, first_mse = quant_noise(page, level=0.25, seed=9)
    second, second_mse = quant_noise(page, level=0.25, seed=9)

    assert np.array_equal(first.K, second.K)
    assert np.array_equal(first.V, second.V)
    assert first_mse == pytest.approx(second_mse)


def test_quant_noise_distortion_increases_monotonically_with_level() -> None:
    page = normal_float_page()
    levels = [0.0, 0.05, 0.25, 0.75]

    distortions = [quant_noise(page, level=level, seed=11)[1] for level in levels]

    assert distortions[0] == 0.0
    assert all(left < right for left, right in zip(distortions, distortions[1:]))


def test_channel_quant_noise_is_reproducible_under_fixed_seed() -> None:
    pages = [normal_float_page(0), normal_float_page(1)]
    channel = Channel()

    first = channel.apply(pages, "quant_noise", level=0.1, seed=3)
    second = channel.apply(pages, "quant_noise", level=0.1, seed=3)

    for page in pages:
        assert np.array_equal(first.fetch(page.page_id).K, second.fetch(page.page_id).K)
        assert first.distortions[page.page_id] == pytest.approx(second.distortions[page.page_id])


def test_forced_evict_reports_missing_never_zero_page() -> None:
    page = normal_float_page()
    result = Channel().apply([page], "forced_evict", page_ids=[page.page_id], seed=0)

    assert result.tiers[page.page_id] is ResidencyTier.EVICTED
    with pytest.raises(MissingPageError):
        result.fetch(page.page_id)
