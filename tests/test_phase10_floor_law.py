"""CPU tests for Phase 10 step (5) / 9.5 redundancy-floor law.

Exercises the deterministic predictor + retention math + report emission (no model). The
GPU sweep is in phase10_floor_law.__main__. Tests assert structure/classification, never a
hard-coded winning retention number.
"""

from aepk_paging.harness.phase10_floor_law import (
    predict_head_dim, predict_kv_width, retention, build_ids,
    write_floor_law_report, FLOOR, INCLUSION_CLEAN_ACC,
)


class _FakeTok:
    """Records which formatting path build_ids took. Has a chat_template so the default-raw
    policy is exercised even when a template EXISTS."""
    chat_template = "dummy-template"

    def __init__(self):
        self.raw = False
        self.tpl = False

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        self.raw = True
        self.last_text = text

        class _Enc:
            input_ids = "IDS"
            def to(self, device):
                return self
        return _Enc()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.tpl = True
        return "TEMPLATED:" + messages[0]["content"]


def test_build_ids_defaults_to_raw_even_with_chat_template():
    tok = _FakeTok()
    build_ids(tok, "hello", "cpu")
    assert tok.raw is True and tok.tpl is False        # RAW primary (HITL / PREREG v2 revised)
    assert tok.last_text == "hello"                    # unwrapped prompt


def test_build_ids_uses_template_only_when_explicitly_requested():
    tok = _FakeTok()
    build_ids(tok, "hello", "cpu", use_chat_template=True)
    assert tok.tpl is True                             # documented fallback path
    assert tok.last_text.startswith("TEMPLATED:")


def test_predictors_on_anchors_and_discriminator():
    # H1 head_dim law
    assert predict_head_dim(128) is True          # qwen1.5b
    assert predict_head_dim(64) is False           # qwen0.5b, tinyllama
    # H2 KV-width law
    assert predict_kv_width(2, 128) is True         # qwen1.5b width 256
    assert predict_kv_width(2, 64) is False          # qwen0.5b width 128
    # discriminator: tinyllama head_dim=64 (H1 FAIL) but width=256 (H2 PASS)
    assert predict_head_dim(64) is False and predict_kv_width(4, 64) is True


def test_retention_math():
    assert abs(retention(1.0, [0.8, 0.6, 0.7]) - 0.7) < 1e-9
    assert abs(retention(0.5, [0.5, 0.5]) - 1.0) < 1e-9
    import math
    assert math.isnan(retention(0.0, [0.0]))       # undefined when clean_acc==0


def test_tolerant_threshold_via_report(tmp_path):
    # rows: (name, model_id, head_dim, n_kv, clean_acc, retention, tolerant)
    # all >= inclusion threshold so the discriminator is included (PREREG v2).
    rows = [
        ("qwen0.5b", "id", 64, 2, 0.9, 0.40, False),
        ("qwen1.5b", "id", 128, 2, 0.9, 0.95, True),
        ("tinyllama", "id", 64, 4, 0.9, 0.90, True),   # discriminator INCLUDED, lands PASS
    ]
    p = tmp_path / "rep.md"
    predicted, observed, match = write_floor_law_report(rows, path=str(p))
    assert predicted == ["qwen1.5b"]                 # H1 predicts only qwen1.5b tolerant
    assert observed == ["qwen1.5b", "tinyllama"]     # tinyllama observed tolerant
    assert match is False                            # H1 falsified in this fixture
    text = p.read_text(encoding="utf-8")
    assert "FLOOR_LAW: predicted=" in text and "observed=" in text and "match=" in text
    assert "INCLUDED" in text                        # discriminator available


def test_report_match_true_when_h1_holds(tmp_path):
    rows = [
        ("qwen0.5b", "id", 64, 2, 0.9, 0.40, False),
        ("qwen1.5b", "id", 128, 2, 0.9, 0.95, True),
        ("tinyllama", "id", 64, 4, 0.9, 0.40, False),  # H1 correct: tinyllama fails
    ]
    p = tmp_path / "rep.md"
    predicted, observed, match = write_floor_law_report(rows, path=str(p))
    assert predicted == observed == ["qwen1.5b"]
    assert match is True


def test_inclusion_excludes_low_clean_acc_discriminator(tmp_path):
    # PREREG v2 inclusion rule: a below-threshold discriminator is EXCLUDED and marked
    # UNAVAILABLE; the match is computed over the included anchors only, not forced.
    assert INCLUSION_CLEAN_ACC == 0.90
    rows = [
        ("qwen0.5b", "id", 64, 2, 0.95, 0.40, False),
        ("qwen1.5b", "id", 128, 2, 0.95, 0.95, True),
        ("tinyllama", "id", 64, 4, 0.50, 0.90, True),  # below threshold -> excluded
    ]
    p = tmp_path / "rep.md"
    predicted, observed, match = write_floor_law_report(rows, path=str(p))
    assert "tinyllama" not in observed and "tinyllama" not in predicted
    assert predicted == observed == ["qwen1.5b"]
    assert match is True                             # anchors agree; discriminator not counted
    text = p.read_text(encoding="utf-8")
    assert "EXCLUDED" in text and "UNAVAILABLE" in text


def test_floor_constant_is_referenced():
    assert 0.0 < FLOOR <= 1.0
