"""Lossy tier and seedable channel-noise injectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

import numpy as np

from aepk_paging.kv_page import KVPage, ResidencyTier


@dataclass(frozen=True)
class QuantizedArray:
    values: np.ndarray
    scale: float
    bit_width: int
    mse: float

    def dequantize(self) -> np.ndarray:
        return self.values.astype(np.float32) * np.float32(self.scale)


@dataclass(frozen=True)
class QuantizedPage:
    page_id: Hashable
    layer: int
    token_range: tuple[int, int]
    K: QuantizedArray
    V: QuantizedArray
    precision_tag: str
    attention_mass: float
    distortion_mse: float

    def dequantize(self) -> KVPage:
        return KVPage(
            page_id=self.page_id,
            layer=self.layer,
            token_range=self.token_range,
            K=self.K.dequantize(),
            V=self.V.dequantize(),
            precision_tag=f"dequantized-{self.precision_tag}",
            attention_mass=self.attention_mass,
        )


@dataclass(frozen=True)
class ChannelResult:
    pages: Mapping[Hashable, KVPage]
    tiers: Mapping[Hashable, ResidencyTier]
    distortions: Mapping[Hashable, float]

    def fetch(self, page_id: Hashable) -> KVPage:
        if self.tiers[page_id] is ResidencyTier.EVICTED:
            raise MissingPageError(f"page {page_id!r} is EVICTED")
        return self.pages[page_id]


class MissingPageError(KeyError):
    pass


def quantize_page(page: KVPage, bit_width: int) -> QuantizedPage:
    if bit_width not in (8, 4):
        raise ValueError("bit_width must be 8 or 4")
    quantized_K = _quantize_array(page.K, bit_width)
    quantized_V = _quantize_array(page.V, bit_width)
    distortion_mse = float((quantized_K.mse + quantized_V.mse) / 2.0)
    return QuantizedPage(
        page_id=page.page_id,
        layer=page.layer,
        token_range=page.token_range,
        K=quantized_K,
        V=quantized_V,
        precision_tag=f"int{bit_width}",
        attention_mass=page.attention_mass,
        distortion_mse=distortion_mse,
    )


def bit_flip(page: QuantizedPage, p: float, seed: int) -> QuantizedPage:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    rng = np.random.default_rng(seed)
    flipped_K = _bit_flip_array(page.K.values, p, rng)
    flipped_V = _bit_flip_array(page.V.values, p, rng)
    return _replace_quantized_values(page, flipped_K, flipped_V)


def quant_noise(page: KVPage, level: float, seed: int) -> tuple[KVPage, float]:
    if level < 0.0:
        raise ValueError("level must be non-negative")
    rng = np.random.default_rng(seed)
    K_noise = rng.normal(loc=0.0, scale=1.0, size=page.K.shape).astype(np.float32)
    V_noise = rng.normal(loc=0.0, scale=1.0, size=page.V.shape).astype(np.float32)
    damaged = KVPage(
        page_id=page.page_id,
        layer=page.layer,
        token_range=page.token_range,
        K=page.K.astype(np.float32) + np.float32(level) * K_noise,
        V=page.V.astype(np.float32) + np.float32(level) * V_noise,
        precision_tag=f"{page.precision_tag}+quant_noise",
        attention_mass=page.attention_mass,
    )
    return damaged, page_mse(page, damaged)


def forced_evict(page_ids: Iterable[Hashable]) -> set[Hashable]:
    return set(page_ids)


class Channel:
    def apply(
        self,
        pages: Iterable[KVPage],
        corruption: str,
        *,
        seed: int,
        level: float = 0.0,
        page_ids: Iterable[Hashable] = (),
    ) -> ChannelResult:
        page_map = {page.page_id: page for page in pages}
        tiers = {page_id: ResidencyTier.RESIDENT for page_id in page_map}
        distortions = {page_id: 0.0 for page_id in page_map}

        if corruption == "quant_noise":
            damaged_pages: dict[Hashable, KVPage] = {}
            for offset, page in enumerate(page_map.values()):
                damaged, mse = quant_noise(page, level=level, seed=seed + offset)
                damaged_pages[page.page_id] = damaged
                distortions[page.page_id] = mse
            page_map = damaged_pages
        elif corruption == "forced_evict":
            for page_id in forced_evict(page_ids):
                if page_id in tiers:
                    tiers[page_id] = ResidencyTier.EVICTED
        elif corruption == "none":
            pass
        else:
            raise ValueError("unknown corruption")

        return ChannelResult(pages=page_map, tiers=tiers, distortions=distortions)


def page_mse(original: KVPage, damaged: KVPage) -> float:
    k_mse = np.mean((original.K.astype(np.float32) - damaged.K.astype(np.float32)) ** 2)
    v_mse = np.mean((original.V.astype(np.float32) - damaged.V.astype(np.float32)) ** 2)
    return float((k_mse + v_mse) / 2.0)


def _quantize_array(values: np.ndarray, bit_width: int) -> QuantizedArray:
    float_values = np.asarray(values, dtype=np.float32)
    max_abs = float(np.max(np.abs(float_values)))
    levels = (2 ** (bit_width - 1)) - 1
    scale = max_abs / float(levels) if max_abs > 0.0 else 1.0
    quantized = np.clip(np.round(float_values / np.float32(scale)), -levels, levels).astype(np.int8)
    dequantized = quantized.astype(np.float32) * np.float32(scale)
    mse = float(np.mean((float_values - dequantized) ** 2))
    return QuantizedArray(values=quantized, scale=scale, bit_width=bit_width, mse=mse)


def _bit_flip_array(values: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    bytes_view = np.ascontiguousarray(values).view(np.uint8)
    bits = np.unpackbits(bytes_view)
    flip_mask = rng.random(size=bits.shape) < p
    damaged_bits = np.bitwise_xor(bits, flip_mask.astype(np.uint8))
    damaged = np.packbits(damaged_bits).reshape(bytes_view.shape)
    return damaged.view(values.dtype).reshape(values.shape)


def _replace_quantized_values(page: QuantizedPage, K: np.ndarray, V: np.ndarray) -> QuantizedPage:
    damaged_K = QuantizedArray(
        values=K,
        scale=page.K.scale,
        bit_width=page.K.bit_width,
        mse=float("nan"),
    )
    damaged_V = QuantizedArray(
        values=V,
        scale=page.V.scale,
        bit_width=page.V.bit_width,
        mse=float("nan"),
    )
    return QuantizedPage(
        page_id=page.page_id,
        layer=page.layer,
        token_range=page.token_range,
        K=damaged_K,
        V=damaged_V,
        precision_tag=page.precision_tag,
        attention_mass=page.attention_mass,
        distortion_mse=float("nan"),
    )
