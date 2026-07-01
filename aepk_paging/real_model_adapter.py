"""
Task 7.2 — adapter between transformers DynamicCache and KVPage.

Verified API (transformers 5.12.1, 2026-07-01):
  past_key_values: DynamicCache
  past_key_values.layers: list[DynamicLayer]   # len = num_layers
  layer.keys:   Tensor[batch, num_kv_heads, seq_len, head_dim]  float16
  layer.values: Tensor[batch, num_kv_heads, seq_len, head_dim]  float16

Storage convention in KVPage:
  K, V stored as float32 ndarray of shape [seq_len, num_kv_heads, head_dim]
  precision_tag = "real_fp16"
  token_range = (0, seq_len)
  attention_mass = mean L2 norm of K vectors (per-head mean, then mean across heads)

Round-trip guarantee:
  fp16 -> float32 is lossless (float32 is a superset of float16).
  float32 -> float16 restores exact bits (values came from fp16, so representable).
  torch.equal(original_fp16, roundtripped_fp16) == True
"""

from __future__ import annotations

import numpy as np
import torch

from .kv_page import KVPage


def dynamiccache_to_pages(pkv, batch_idx: int = 0) -> list[KVPage]:
    """
    Extract every layer from a DynamicCache as KVPage objects.

    Returns one KVPage per layer (len == num_layers).
    K/V shape stored: [seq_len, num_kv_heads, head_dim] float32.
    """
    pages: list[KVPage] = []
    for layer_idx, layer in enumerate(pkv.layers):
        # layer.keys: [batch, num_kv_heads, seq_len, head_dim]
        k_t = layer.keys[batch_idx]   # [num_kv_heads, seq_len, head_dim]
        v_t = layer.values[batch_idx]  # same

        # [seq_len, num_kv_heads, head_dim] — shape-preserving, contiguous
        k_np = k_t.permute(1, 0, 2).contiguous().cpu().float().numpy()
        v_np = v_t.permute(1, 0, 2).contiguous().cpu().float().numpy()

        seq_len = k_np.shape[0]

        # attention_mass: mean L2-norm across all K vectors
        norms = np.linalg.norm(k_np.reshape(seq_len, -1), axis=1)  # [seq_len]
        attn_mass = float(norms.mean())
        if not np.isfinite(attn_mass) or attn_mass < 0.0:
            attn_mass = 0.0

        pages.append(KVPage(
            page_id=("real", layer_idx),
            layer=layer_idx,
            token_range=(0, seq_len),
            K=k_np,
            V=v_np,
            precision_tag="real_fp16",
            attention_mass=attn_mass,
        ))
    return pages


def pages_to_kv_tensors(
    page: KVPage,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct [1, num_kv_heads, seq_len, head_dim] K/V tensors from a KVPage.

    page.K shape: [seq_len, num_kv_heads, head_dim] float32
    Returns tensors with the requested dtype (fp16 round-trip is bit-exact).
    """
    k = torch.from_numpy(page.K.copy()).to(device=device, dtype=torch.float32)
    v = torch.from_numpy(page.V.copy()).to(device=device, dtype=torch.float32)

    seq_len, num_kv_heads, head_dim = k.shape
    # [seq_len, nh, hd] -> [nh, seq_len, hd] -> [1, nh, seq_len, hd]
    k = k.permute(1, 0, 2).contiguous().unsqueeze(0).to(dtype)
    v = v.permute(1, 0, 2).contiguous().unsqueeze(0).to(dtype)
    return k, v
