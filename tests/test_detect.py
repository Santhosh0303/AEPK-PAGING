import numpy as np

from aepk_paging.coding import HammingSECDEDCode
from aepk_paging.detect import (
    attention_mass,
    attention_mass_detector,
    confidence_proxy,
    norm_consistency_detector,
    norm_ratio,
)
from aepk_paging.kv_page import KVPage
from aepk_paging.lossy_tier import quant_noise, quantize_page


def clean_page() -> KVPage:
    rng = np.random.default_rng(400)
    K = rng.normal(loc=0.0, scale=1.0, size=(32, 8)).astype(np.float32)
    V = (K * np.float32(0.75) + rng.normal(loc=0.0, scale=0.05, size=(32, 8))).astype(
        np.float32
    )
    base = KVPage(
        page_id="p0",
        layer=0,
        token_range=(0, 32),
        K=K,
        V=V,
        precision_tag="float32",
        attention_mass=0.0,
    )
    return KVPage(
        page_id=base.page_id,
        layer=base.layer,
        token_range=base.token_range,
        K=base.K,
        V=base.V,
        precision_tag=base.precision_tag,
        attention_mass=attention_mass(base),
    )


def test_clean_pages_pass_physics_invariants() -> None:
    page = clean_page()

    mass = attention_mass_detector(page, tolerance=0.01)
    norm = norm_consistency_detector(page, expected_ratio=norm_ratio(page), tolerance=0.01)

    assert not mass.flag
    assert mass.deviation == 0.0
    assert not norm.flag
    assert norm.deviation == 0.0


def test_invariants_flag_uncoded_quant_noise_that_secded_syndrome_cannot_see() -> None:
    page = clean_page()
    corrupted, _ = quant_noise(page, level=0.8, seed=17)
    code = HammingSECDEDCode()
    encoded_after_corruption = code.encode([quantize_page(corrupted, bit_width=8)])[0]
    syndrome_report = code.detect([encoded_after_corruption])

    mass = attention_mass_detector(corrupted, expected_mass=page.attention_mass, tolerance=0.01)
    norm = norm_consistency_detector(corrupted, expected_ratio=norm_ratio(page), tolerance=0.01)

    assert syndrome_report.suspect_ids == ()
    assert syndrome_report.uncorrectable_ids == ()
    assert mass.flag
    assert mass.deviation > mass.tolerance
    assert norm.flag
    assert norm.deviation > norm.tolerance


def test_confident_wrong_confidence_blind_but_invariant_catches() -> None:
    page = clean_page()
    corrupted, _ = quant_noise(page, level=0.8, seed=17)

    low_surprise_logits = np.array([12.0, -2.0, -4.0], dtype=np.float32)
    confidence = confidence_proxy(low_surprise_logits, surprise_threshold=0.25)
    invariant = attention_mass_detector(
        corrupted,
        expected_mass=page.attention_mass,
        tolerance=0.01,
    )

    assert not confidence.flag
    assert confidence.deviation < confidence.tolerance
    assert invariant.flag
    assert invariant.deviation > invariant.tolerance
