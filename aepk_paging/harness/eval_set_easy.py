"""Phase 10 step (16) — EASY short-factual probe pool + combined grid pool.

The step-4 large pool (`eval_set_large`, 105 probes) is HARD by design (a factual
question after a 1783-char distractor passage + SciQ), so the strongest grid model
scores clean_acc ~0.43 and only 2/8 models clear the N_clean_correct >= 30 inclusion
gate — the powered law/transition test is under-powered. This module adds ~100 EASY
short-factual probes (CW_PROBES style: one-word capital / color / arithmetic / planet /
misc answers with alternatives) so more grid models reach >= 30 clean-correct probes.

`get_combined_probes()` returns the grid pool the step-17 sweep runs on:
LARGE_PROBES + EASY_PROBES, deduplicated by prompt (LARGE wins on collision) and passed
through the SAME leakage filter as `eval_set_large` (a probe is dropped when its gold
answer or an alternative appears as a whole word in its own prompt — so accuracy measures
parametric recall, not context copying).

Honesty spine S9: zero edits to Phase 2-5 source. REUSES `eval_set_large`'s leakage
helpers (`answer_leaks`, `_answers`, `_normalize`) rather than re-implementing them.
Deterministic (fixed hand-authored list, no RNG). Construction is LAZY and does NO
import-time work: `get_easy_probes()` / `get_combined_probes()` build once and cache; the
legacy `EASY_PROBES` / `COMBINED_PROBES` attributes still resolve via module `__getattr__`.
Nothing heavy (torch, harness chain) is imported at module import time.
"""

from __future__ import annotations

MIN_EASY = 100          # >= 100 easy probes survive the leakage filter
MIN_COMBINED = 200      # >= 200 combined -> accuracy granularity <= 1/200 on the pool

_easy_cache: list[dict] | None = None
_combined_cache: list[dict] | None = None


# ---------------------------------------------------------------------------
# Hand-authored EASY probes. One-word (or short) answers, leakage-free by
# construction: the gold answer never appears in its own prompt. Number answers
# carry a spelled-out alternative so the matcher accepts either form.
# ---------------------------------------------------------------------------

def _cap(country: str, city: str, alts: list[str] | None = None) -> dict:
    return {"prompt": f"What is the capital city of {country}? One word:",
            "expected": city, "alternatives": alts or []}


def _num(question: str, digits: str, word: str) -> dict:
    return {"prompt": f"{question} Answer with a number:",
            "expected": digits, "alternatives": [word]}


_CAPITALS = [
    _cap("Germany", "Berlin"), _cap("Russia", "Moscow"), _cap("China", "Beijing"),
    _cap("Canada", "Ottawa"), _cap("Australia", "Canberra"), _cap("Greece", "Athens"),
    _cap("Portugal", "Lisbon"), _cap("Norway", "Oslo"), _cap("Sweden", "Stockholm"),
    _cap("Finland", "Helsinki"), _cap("Denmark", "Copenhagen"), _cap("Poland", "Warsaw"),
    _cap("Austria", "Vienna"), _cap("Ireland", "Dublin"), _cap("Turkey", "Ankara"),
    _cap("India", "Delhi", ["New Delhi"]), _cap("Brazil", "Brasilia", ["Brasília"]),
    _cap("Cuba", "Havana"), _cap("Peru", "Lima"), _cap("Chile", "Santiago"),
    _cap("Kenya", "Nairobi"), _cap("Thailand", "Bangkok"), _cap("Iran", "Tehran"),
    _cap("Iraq", "Baghdad"), _cap("Netherlands", "Amsterdam"), _cap("Belgium", "Brussels"),
    _cap("Switzerland", "Bern"), _cap("Hungary", "Budapest"), _cap("Ukraine", "Kyiv", ["Kiev"]),
    _cap("South Korea", "Seoul"), _cap("Vietnam", "Hanoi"), _cap("Indonesia", "Jakarta"),
    _cap("Philippines", "Manila"), _cap("Morocco", "Rabat"), _cap("Nigeria", "Abuja"),
    _cap("Colombia", "Bogota", ["Bogotá"]), _cap("Czechia", "Prague"),
]

_COLORS = [
    {"prompt": "What color is healthy grass? One word:", "expected": "green"},
    {"prompt": "What color is a ripe banana? One word:", "expected": "yellow"},
    {"prompt": "What color is fresh blood? One word:", "expected": "red"},
    {"prompt": "What color is freshly fallen snow? One word:", "expected": "white"},
    {"prompt": "What color is a lump of coal? One word:", "expected": "black"},
    {"prompt": "What color is a ripe lemon? One word:", "expected": "yellow"},
    {"prompt": "What color is a ripe tomato? One word:", "expected": "red"},
    {"prompt": "What color is the ocean on a clear day? One word:", "expected": "blue"},
    {"prompt": "What color is a carrot? One word:", "expected": "orange"},
    {"prompt": "What color is a ripe strawberry? One word:", "expected": "red"},
    {"prompt": "What color is a glass of milk? One word:", "expected": "white"},
    {"prompt": "What color is an emerald gemstone? One word:", "expected": "green"},
    {"prompt": "What color is a ruby gemstone? One word:", "expected": "red"},
    {"prompt": "What color is a sapphire gemstone? One word:", "expected": "blue"},
]

_ARITH = [
    _num("What is 2 plus 2?", "4", "four"),
    _num("What is 3 plus 4?", "7", "seven"),
    _num("What is 5 plus 6?", "11", "eleven"),
    _num("What is 8 plus 4?", "12", "twelve"),
    _num("What is 9 plus 6?", "15", "fifteen"),
    _num("What is 10 plus 10?", "20", "twenty"),
    _num("What is 6 plus 6?", "12", "twelve"),
    _num("What is 4 plus 5?", "9", "nine"),
    _num("What is 3 plus 3?", "6", "six"),
    _num("What is 8 minus 5?", "3", "three"),
    _num("What is 9 minus 4?", "5", "five"),
    _num("What is 10 minus 2?", "8", "eight"),
    _num("What is 12 minus 4?", "8", "eight"),
    _num("What is 7 minus 6?", "1", "one"),
    _num("What is 2 times 3?", "6", "six"),
    _num("What is 3 times 3?", "9", "nine"),
    _num("What is 4 times 2?", "8", "eight"),
    _num("What is 5 times 2?", "10", "ten"),
    _num("What is 6 times 2?", "12", "twelve"),
    _num("What is 10 times 10?", "100", "one hundred"),
    _num("What is 2 times 5?", "10", "ten"),
    _num("What is 5 times 5?", "25", "twenty-five"),
    _num("What is 4 times 4?", "16", "sixteen"),
    _num("What is 3 times 4?", "12", "twelve"),
]

_PLANETS = [
    {"prompt": "Which planet is the largest in our solar system? One word:", "expected": "Jupiter"},
    {"prompt": "Which planet is known as the Red Planet? One word:", "expected": "Mars"},
    {"prompt": "Which planet is closest to the sun? One word:", "expected": "Mercury"},
    {"prompt": "Which planet is famous for its bright rings? One word:", "expected": "Saturn"},
    {"prompt": "Which planet is home to all humans? One word:", "expected": "Earth"},
    {"prompt": "Which planet is the hottest in our solar system? One word:", "expected": "Venus"},
    {"prompt": "What star sits at the center of our solar system? One word:", "expected": "Sun"},
]

_MISC = [
    _num("How many days are in a week?", "7", "seven"),
    _num("How many months are in a year?", "12", "twelve"),
    _num("How many legs does a spider have?", "8", "eight"),
    _num("How many legs does an insect have?", "6", "six"),
    _num("How many sides does a triangle have?", "3", "three"),
    _num("How many sides does a square have?", "4", "four"),
    _num("How many colors are in a rainbow?", "7", "seven"),
    _num("How many continents are there on our world?", "7", "seven"),
    _num("How many primary colors are there?", "3", "three"),
    _num("How many wheels does a typical car have?", "4", "four"),
    _num("How many hours are in a day?", "24", "twenty-four"),
    _num("How many letters are in the English alphabet?", "26", "twenty-six"),
    _num("What is the freezing point of water in Celsius?", "0", "zero"),
    _num("What is the boiling point of water in Celsius?", "100", "one hundred"),
    {"prompt": "What gas do humans breathe in to survive? One word:", "expected": "oxygen"},
    {"prompt": "What is the common name for H2O? One word:", "expected": "water"},
    {"prompt": "What is the largest ocean on our world? One word:", "expected": "Pacific"},
    {"prompt": "What is the largest mammal alive today? One word:", "expected": "whale",
     "alternatives": ["blue whale"]},
    {"prompt": "What is the fastest land animal? One word:", "expected": "cheetah"},
    {"prompt": "Which big cat is called the king of the jungle? One word:", "expected": "lion"},
    {"prompt": "What is the opposite of hot? One word:", "expected": "cold"},
    {"prompt": "What is the opposite of up? One word:", "expected": "down"},
    {"prompt": "What is the opposite of black? One word:", "expected": "white"},
    {"prompt": "What is the opposite of day? One word:", "expected": "night"},
    {"prompt": "What do you call frozen water? One word:", "expected": "ice"},
    {"prompt": "What is the chemical symbol for gold? One word:", "expected": "Au"},
    {"prompt": "What is the chemical symbol for oxygen? One letter:", "expected": "O"},
    {"prompt": "What is the tallest land animal? One word:", "expected": "giraffe"},
]

EASY_PROBES_RAW = _CAPITALS + _COLORS + _ARITH + _PLANETS + _MISC


# ---------------------------------------------------------------------------
# Lazy, leakage-filtered accessors (heavy imports deferred to call time).
# ---------------------------------------------------------------------------

def build_easy_eval_set() -> list[dict]:
    """>= 100 leakage-free EASY probes, deterministic. Reuses eval_set_large's
    leakage filter (same rules); nothing heavy imported at module import time."""
    from aepk_paging.harness.eval_set_large import answer_leaks, _answers, _normalize

    raw = [_normalize(p) for p in EASY_PROBES_RAW]
    clean = [p for p in raw if not answer_leaks(p["prompt"], _answers(p))]
    assert len(clean) >= MIN_EASY, (
        f"only {len(clean)} easy probes survive the leakage filter (< {MIN_EASY}); "
        "extend EASY_PROBES_RAW."
    )
    return clean


def build_combined_grid_pool() -> list[dict]:
    """The step-17 grid pool: LARGE_PROBES + EASY_PROBES, deduped by prompt (LARGE
    wins on collision) and leakage-filtered under the shared rules. >= 200 probes."""
    from aepk_paging.harness.eval_set_large import answer_leaks, _answers, get_large_probes

    seen: set[str] = set()
    out: list[dict] = []
    for p in list(get_large_probes()) + get_easy_probes():
        pr = p["prompt"]
        if pr in seen:
            continue
        if answer_leaks(pr, _answers(p)):     # belt-and-braces (both inputs pre-filtered)
            continue
        seen.add(pr)
        out.append(p)
    assert len(out) >= MIN_COMBINED, (
        f"only {len(out)} combined probes (< {MIN_COMBINED}); granularity would exceed 1/200."
    )
    return out


def get_easy_probes() -> list[dict]:
    """Cached accessor — builds the easy probe set once, on first use."""
    global _easy_cache
    if _easy_cache is None:
        _easy_cache = build_easy_eval_set()
    return _easy_cache


def get_combined_probes() -> list[dict]:
    """Cached accessor — builds the combined grid pool once, on first use."""
    global _combined_cache
    if _combined_cache is None:
        _combined_cache = build_combined_grid_pool()
    return _combined_cache


def __getattr__(name: str):
    # Legacy import paths still work, now lazily.
    if name == "EASY_PROBES":
        return get_easy_probes()
    if name == "COMBINED_PROBES":
        return get_combined_probes()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    easy = get_easy_probes()
    combined = get_combined_probes()
    print(f"EASY_PROBES: n={len(easy)} unique={len(set(p['prompt'] for p in easy))}")
    print(f"COMBINED_PROBES: n={len(combined)} granularity={1/len(combined):.4f}")
