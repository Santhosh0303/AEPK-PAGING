"""CPU tests for Phase 10 step (6) / 9.4 statistics.

Exercises the deterministic crossover-interpolation + CI math + report emission (no model).
Tests assert structure/interpolation correctness, never a hard-coded GPU number.
"""

import math

from aepk_paging.harness.phase10_stats import (
    crossover_level, ci95, write_stats_report, FLOOR,
)
from aepk_paging.harness.eval_set_large import (
    LARGE_PROBES, MIN_PROBES, answer_leaks, _answers,
)


def test_crossover_interpolation_midpoint():
    # retention 0.8 at L=0.2, 0.6 at L=0.3, FLOOR=0.7 -> crossover halfway = 0.25
    lv = (0.1, 0.2, 0.3, 0.4)
    rt = [0.95, 0.80, 0.60, 0.40]
    x = crossover_level(lv, rt, floor=0.70)
    assert abs(x - 0.25) < 1e-9


def test_crossover_right_censored():
    lv = (0.1, 0.2, 0.3)
    rt = [0.99, 0.95, 0.90]                 # never below FLOOR
    assert crossover_level(lv, rt, floor=0.70) == 0.3


def test_crossover_left_censored():
    lv = (0.1, 0.2, 0.3)
    rt = [0.50, 0.40, 0.30]                 # already below at first level
    assert crossover_level(lv, rt, floor=0.70) == 0.1


def test_ci95_basic():
    mu, ci = ci95([0.4, 0.4, 0.4, 0.4, 0.4])
    assert abs(mu - 0.4) < 1e-9 and ci == 0.0        # zero variance -> zero CI
    mu2, ci2 = ci95([0.3, 0.5])
    assert abs(mu2 - 0.4) < 1e-9 and ci2 > 0.0
    m1, c1 = ci95([0.4])
    assert m1 == 0.4 and c1 == 0.0                    # n=1 -> ci 0


def test_verdict_line_exists(tmp_path):
    seeds = (0, 1, 2, 3, 4)
    levels = (0.1, 0.2, 0.3, 0.4)
    grid = {s: [0.95, 0.80, 0.60, 0.40] for s in seeds}
    crossovers = [crossover_level(levels, grid[s]) for s in seeds]
    p = tmp_path / "rep.md"
    mu, ci, n = write_stats_report(1.0, crossovers, grid, seeds=seeds, levels=levels, path=str(p),
                                   n_cc=50)
    assert n == 5
    text = p.read_text(encoding="utf-8")
    assert "STATS: crossover=" in text and "seeds=" in text
    assert "N_cc=" in text                    # PREREG v3 inclusion note (line exists, no value)


def test_floor_referenced():
    assert 0.0 < FLOOR <= 1.0


# --- Step 4: >=100-probe large eval set --------------------------------------

def test_large_set_has_at_least_100_probes():
    # >=100 -> accuracy granularity <= 1/100 (ACCEPT).
    assert len(LARGE_PROBES) >= MIN_PROBES >= 100
    assert 1.0 / len(LARGE_PROBES) <= 1.0 / 100


def test_large_set_no_duplicate_prompts():
    prompts = [p["prompt"] for p in LARGE_PROBES]
    assert len(set(prompts)) == len(prompts)


def test_large_set_format():
    for p in LARGE_PROBES:
        assert isinstance(p["prompt"], str) and p["prompt"].strip()
        assert isinstance(p["expected"], str) and p["expected"].strip()
        assert isinstance(p["alternatives"], list)          # normalized to a list
        assert all(isinstance(a, str) for a in p["alternatives"])


def test_large_set_no_answer_leakage():
    # No probe may contain its own gold answer (or an alternative) verbatim in the prompt.
    leaked = [p["prompt"][:50] for p in LARGE_PROBES
              if answer_leaks(p["prompt"], _answers(p))]
    assert leaked == []


def test_answer_leaks_detector():
    # sanity of the leakage detector itself (word-boundary, case-insensitive).
    assert answer_leaks("The capital is Paris here.", ["Paris"]) is True
    assert answer_leaks("What is the capital of France?", ["Paris"]) is False
    assert answer_leaks("Parisian food.", ["Paris"]) is False   # boundary: no substring hit
