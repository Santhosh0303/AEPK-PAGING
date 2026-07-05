"""CPU tests for Phase 10 step (5) factorial grid (no model).

Exercises the deterministic verdict math: answer normalization, the FLOOR_LAW_GRID H1/H2/neither
logic, and the logistic-vs-linear TRANSITION selection. Tests assert structure/classification,
never a hard-coded GPU retention value.
"""

import pytest

from aepk_paging.harness.phase10_grid import (
    normalize_answer, floor_law_grid_verdict, transition_verdict,
    arch, MIN_CLEAN_CORRECT, FLOOR, LEVEL, SEEDS,
)


def test_normalize_answer_strips_chat_artifact():
    assert normalize_answer(" 4.Human: what next").strip() == "4."
    assert normalize_answer(" the sun.Assistant: more").strip() == "the sun."
    assert normalize_answer("Paris") == "Paris"                    # untouched when clean


# --- FLOOR_LAW_GRID verdict (rows: name, head_dim, n_kv, retention, tolerant) ---

def _rows(tol_set):
    # A: hd128/kvw256 (H1 yes, H2 yes); B: hd64/kvw128 (both no); C: hd64/kvw256 (H1 no, H2 yes)
    spec = {"A": (128, 2), "B": (64, 2), "C": (64, 4)}
    return [(n, hd, nkv, 0.9 if n in tol_set else 0.1, n in tol_set)
            for n, (hd, nkv) in spec.items()]


def test_grid_verdict_h1():
    pH1, pH2, obs, v = floor_law_grid_verdict(_rows({"A"}))
    assert pH1 == ["A"] and pH2 == ["A", "C"] and obs == ["A"] and v == "H1"


def test_grid_verdict_h2():
    pH1, pH2, obs, v = floor_law_grid_verdict(_rows({"A", "C"}))
    assert pH2 == ["A", "C"] and obs == ["A", "C"] and v == "H2"


def test_grid_verdict_neither():
    pH1, pH2, obs, v = floor_law_grid_verdict(_rows({"A", "B"}))
    assert obs == ["A", "B"] and v == "neither"


def test_grid_verdict_indistinguishable():
    # Only models where H1_pred == H2_pred (no discriminator like C): predictions coincide,
    # observation matches -> the laws cannot be separated on this set.
    rows = [r for r in _rows({"A"}) if r[0] in ("A", "B")]      # A: both True, B: both False
    pH1, pH2, obs, v = floor_law_grid_verdict(rows)
    assert pH1 == pH2 == obs == ["A"] and v == "indistinguishable"


def test_grid_report_indistinguishable_gate_line(tmp_path):
    from aepk_paging.harness.phase10_grid import write_grid_report
    # 8-field rows: (name, family, head_dim, n_kv, n_cc, retention, tolerant, status)
    rows = [
        ("A", "fam", 128, 2, 50, 0.9, True,  "included"),
        ("B", "fam", 64,  2, 40, 0.1, False, "included"),
        ("C", "fam", 64,  4, 10, 0.1, False, "excluded(N_cc=10<30)"),   # H2-contradicting row
    ]
    p = tmp_path / "rep.md"
    write_grid_report(rows, path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "FLOOR_LAW_GRID:" in text and "verdict=indistinguishable" in text
    assert "EXPLORATORY ONLY" in text                           # line exists, value not asserted


# --- TRANSITION fit selection ---

def test_transition_undetermined_too_few_points():
    form, det = transition_verdict([1, 2, 3], [0.1, 0.5, 0.9])
    assert form == "undetermined" and det["n"] == 3


def test_transition_gradual_on_linear_data():
    pytest.importorskip("scipy")
    xs = [100, 200, 300, 400, 500, 600]
    ys = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]              # perfectly linear
    form, det = transition_verdict(xs, ys)
    assert form in ("gradual", "undetermined")             # never 'sharp' on a straight line
    assert "aic_linear" in det


def test_transition_sharp_on_step_data():
    pytest.importorskip("scipy")
    xs = [1, 2, 3, 4, 5, 6]
    ys = [0.02, 0.03, 0.05, 0.92, 0.95, 0.96]              # abrupt threshold
    form, det = transition_verdict(xs, ys)
    assert form == "sharp"
    assert det["aic_logistic"] + 2 < det["aic_linear"]


def test_constants_sane():
    assert MIN_CLEAN_CORRECT >= 30
    assert 0.0 < FLOOR <= 1.0 and 0.0 < LEVEL < 1.0 and tuple(SEEDS) == (0, 1, 2)


class _Cfg:
    hidden_size = 1024
    num_attention_heads = 16
    num_key_value_heads = 4


def test_arch_reads_config():
    hd, nkv = arch(_Cfg())
    assert hd == 64 and nkv == 4                            # 1024/16=64, gqa kv=4