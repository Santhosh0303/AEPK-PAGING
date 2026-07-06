"""CPU tests for Phase 10 step (16) — EASY probe pool + combined grid pool.

Assert structure/count/leakage/format on the deterministic hand-authored set (no model,
no network). Mirrors the step-4 large-set tests. The combined pool feeds the step-17
powered grid rerun; granularity <= 1/200 is the ACCEPT bar.
"""

import subprocess
import sys

from aepk_paging.harness.eval_set_easy import (
    get_easy_probes, get_combined_probes, MIN_EASY, MIN_COMBINED,
)
from aepk_paging.harness.eval_set_large import answer_leaks, _answers


def test_easy_set_has_at_least_100_probes():
    easy = get_easy_probes()
    assert len(easy) >= MIN_EASY >= 100


def test_combined_pool_at_least_200_and_granularity():
    # >= 200 combined -> accuracy granularity <= 1/200 (ACCEPT).
    pool = get_combined_probes()
    assert len(pool) >= MIN_COMBINED >= 200
    assert 1.0 / len(pool) <= 1.0 / 200


def test_easy_set_no_duplicate_prompts():
    prompts = [p["prompt"] for p in get_easy_probes()]
    assert len(set(prompts)) == len(prompts)


def test_combined_pool_no_duplicate_prompts():
    prompts = [p["prompt"] for p in get_combined_probes()]
    assert len(set(prompts)) == len(prompts)


def test_easy_set_format():
    for p in get_easy_probes():
        assert isinstance(p["prompt"], str) and p["prompt"].strip()
        assert isinstance(p["expected"], str) and p["expected"].strip()
        assert isinstance(p["alternatives"], list)          # normalized to a list
        assert all(isinstance(a, str) for a in p["alternatives"])


def test_easy_set_no_answer_leakage():
    # No probe may contain its own gold answer (or an alternative) verbatim in the prompt.
    leaked = [p["prompt"][:50] for p in get_easy_probes()
              if answer_leaks(p["prompt"], _answers(p))]
    assert leaked == []


def test_combined_pool_no_answer_leakage():
    leaked = [p["prompt"][:50] for p in get_combined_probes()
              if answer_leaks(p["prompt"], _answers(p))]
    assert leaked == []


def test_combined_superset_of_easy_and_large():
    # Every easy probe (post-filter) survives into the combined pool; the pool is
    # strictly larger than either input (LARGE=105 hard + >=100 easy, minus dup collisions).
    easy_prompts = {p["prompt"] for p in get_easy_probes()}
    pool_prompts = {p["prompt"] for p in get_combined_probes()}
    assert easy_prompts <= pool_prompts
    assert len(pool_prompts) > len(easy_prompts)


def test_import_is_lazy_no_torch():
    # Importing the module (in a CLEAN interpreter — the shared pytest process may already
    # hold torch/caches) must not pull in torch or trigger a probe build (test economy).
    code = (
        "import sys, aepk_paging.harness.eval_set_easy as m; "
        "assert 'torch' not in sys.modules, 'torch imported at import time'; "
        "assert m._easy_cache is None and m._combined_cache is None, 'built at import time'; "
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
