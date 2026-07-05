"""Phase 10 step (4) — >=100-probe evaluation set for the redundancy-floor statistics.

Combines the Phase 9.3a long-context probe set (`build_lc_probe_set`: 30 curated EVAL_PROBES +
70 allenai/sciq rows, each prefixed with the shared LONG_CONTEXT_PASSAGE) with the short-factual
`CW_PROBES`. Every probe carries `expected` and an `alternatives` list (normalized to a list,
possibly empty). Probes whose gold answer leaks verbatim (word-boundary, case-insensitive) into
their own prompt are DROPPED, so accuracy measures parametric recall under KV noise rather than
reading the answer straight off the context passage.

Step 8 (test economy): the 70 sciq rows are VENDORED at tests/fixtures/sciq70.json (provenance
in the fixture header: allenai/sciq@validation rows 0-69) — no dataset/network access at import
or test time. Construction is LAZY: `get_large_probes()` builds once and caches; the legacy
`LARGE_PROBES` module attribute still works via module `__getattr__`. Nothing heavy (torch,
harness chain) is imported at module import time, and the >=100 assert runs only on first build.

Honesty spine S9: zero edits to Phase 2-5 source — this only REUSES `build_lc_probe_set` and
`CW_PROBES`. Deterministic (fixed construction, no RNG). >=100 probes after leakage filtering,
giving accuracy granularity <= 1/100.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MIN_PROBES = 100

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sciq70.json"

_cache: list[dict] | None = None


def _answers(probe: dict) -> list[str]:
    """Gold answer + alternatives as a flat list of non-empty strings."""
    return [a for a in [probe.get("expected"), *(probe.get("alternatives") or [])] if a]


def answer_leaks(prompt: str, answers: list[str]) -> bool:
    """True if any gold answer appears as a whole word (case-insensitive) in the prompt —
    i.e. the probe is answerable by copying from context, not by recall."""
    p = prompt.lower()
    return any(re.search(rf"\b{re.escape(a.strip().lower())}\b", p) for a in answers if a.strip())


def _normalize(probe: dict) -> dict:
    """Ensure a probe has a string prompt/expected and an `alternatives` LIST."""
    e = dict(probe)
    e["alternatives"] = list(e.get("alternatives") or [])
    return e


def _sciq_probes_from_fixture() -> list[dict]:
    """70 vendored allenai/sciq@validation rows 0-69 from tests/fixtures/sciq70.json.

    Same prompt formatting as phase9_accuracy._sciq_probes, so the base probe list is
    identical to build_extended_eval_set() — but read from the committed fixture, never
    from HF/datasets."""
    data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    probes = [
        {"prompt": f"{r['question']} Answer with one or two words:", "expected": r["answer"]}
        for r in data["probes"]
    ]
    assert len(probes) == 70, f"sciq fixture has {len(probes)} rows, expected 70"
    return probes


def build_large_eval_set() -> list[dict]:
    """>=100 leakage-free probes: long-context (9.3a) + short-factual (CW), deterministic.

    Heavy imports are deferred to call time so importing this module stays cheap and
    touches no network."""
    from aepk_paging.harness.eval_set import EVAL_PROBES
    from aepk_paging.harness.phase9_3_lc import build_lc_probe_set
    from aepk_paging.harness.phase9_cw import CW_PROBES

    base = list(EVAL_PROBES) + _sciq_probes_from_fixture()
    raw = [_normalize(p) for p in (build_lc_probe_set(base) + list(CW_PROBES))]
    clean = [p for p in raw if not answer_leaks(p["prompt"], _answers(p))]
    assert len(clean) >= MIN_PROBES, (
        f"only {len(clean)} probes survive leakage filter (< {MIN_PROBES}); "
        "extend the short-factual set before running stats."
    )
    return clean


def get_large_probes() -> list[dict]:
    """Cached accessor — builds the probe set once, on first use."""
    global _cache
    if _cache is None:
        _cache = build_large_eval_set()
    return _cache


def __getattr__(name: str):
    # Legacy import path: `from ... import LARGE_PROBES` still works, now lazily.
    if name == "LARGE_PROBES":
        return get_large_probes()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    ps = get_large_probes()
    print(f"LARGE_PROBES: n={len(ps)} unique_prompts={len(set(p['prompt'] for p in ps))} "
          f"granularity={1/len(ps):.4f}")
    print("  sample:", {k: (v[:40] if isinstance(v, str) else v) for k, v in ps[0].items()})
