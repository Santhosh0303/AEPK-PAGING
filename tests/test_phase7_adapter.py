"""
Phase 7.2 — acceptance test: KVPage <-> DynamicCache round-trip BIT-EXACT.

Requires: torch + transformers (Phase 7 only).
Skipped automatically if torch/transformers not installed (shouldn't happen in Phase 7 env).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM
from aepk_paging.real_model_adapter import dynamiccache_to_pages, pages_to_kv_tensors

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@pytest.fixture(scope="module")
def model_and_cache():
    """Load model once per module; run one forward pass; return (model, pkv, layer0_K)."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE,
    )
    model.eval()
    inp = tok("The quick brown fox", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inp, use_cache=True)
    return model, out.past_key_values, tok


class TestRoundTrip:
    """7.2 — KVPage <-> DynamicCache adapter round-trip."""

    def test_pages_extracted_all_layers(self, model_and_cache):
        _, pkv, _ = model_and_cache
        pages = dynamiccache_to_pages(pkv)
        assert len(pages) == len(pkv.layers), "one page per layer"

    def test_page_fields_valid(self, model_and_cache):
        _, pkv, _ = model_and_cache
        pages = dynamiccache_to_pages(pkv)
        for i, page in enumerate(pages):
            assert page.layer == i
            assert page.token_range[0] == 0
            assert page.token_range[1] > 0
            assert page.K.shape == page.V.shape
            assert page.K.ndim == 3  # [seq_len, num_kv_heads, head_dim]
            assert page.precision_tag == "real_fp16"
            assert page.attention_mass >= 0.0

    def test_kv_roundtrip_bit_exact_all_layers(self, model_and_cache):
        """Core 7.2 gate: every layer K and V round-trip bit-exact."""
        _, pkv, _ = model_and_cache
        pages = dynamiccache_to_pages(pkv)

        for layer_idx, (layer, page) in enumerate(zip(pkv.layers, pages)):
            orig_k = layer.keys[0]   # [num_kv_heads, seq_len, head_dim]
            orig_v = layer.values[0]

            rec_k, rec_v = pages_to_kv_tensors(page, dtype=DTYPE, device=DEVICE)
            rec_k = rec_k.squeeze(0)  # [num_kv_heads, seq_len, head_dim]
            rec_v = rec_v.squeeze(0)

            assert torch.equal(orig_k, rec_k), (
                f"Layer {layer_idx} K not bit-exact: "
                f"max_diff={( orig_k.float() - rec_k.float()).abs().max().item()}"
            )
            assert torch.equal(orig_v, rec_v), (
                f"Layer {layer_idx} V not bit-exact: "
                f"max_diff={(orig_v.float() - rec_v.float()).abs().max().item()}"
            )

    def test_page_shape_matches_model_config(self, model_and_cache):
        """Shape sanity: seq_len * num_kv_heads * head_dim consistent."""
        _, pkv, _ = model_and_cache
        pages = dynamiccache_to_pages(pkv)
        layer0 = pkv.layers[0]
        _, num_kv_heads, seq_len, head_dim = layer0.keys.shape  # [batch, nh, seq, hd]
        page0 = pages[0]
        assert page0.K.shape == (seq_len, num_kv_heads, head_dim)
