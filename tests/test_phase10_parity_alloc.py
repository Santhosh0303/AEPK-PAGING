"""CPU tests for Phase 10 step (4) thermodynamic parity allocation.

Exercises the deterministic allocation math on synthetic KVPages (no model). Confirms:
concentrated Gibbs mass -> thermo cheaper than uniform (iso-protected); diffuse mass -> they
coincide (the ALLOWED-to-FAIL case); the verdict LINE EXISTS. Tests assert structure/inequality,
never a hard-coded winning number.
"""

import numpy as np

from aepk_paging.kv_page import KVPage
from aepk_paging.harness.phase10_parity_alloc import (
    gibbs_weights, parity_groups, critical_set, allocate, recoverable_critical,
    sweep_temperatures, crit_sizes_by_temperature,
    topk_norm_set, allocate_topk,
    write_parity_alloc_report, GROUP_SIZE, NUM_PARITY,
)


def _page(layer: int, mass: float, T: int = 8, H: int = 2, D: int = 4) -> KVPage:
    rng = np.random.default_rng(layer)
    K = rng.normal(size=(T, H, D)).astype(np.float32)
    V = rng.normal(size=(T, H, D)).astype(np.float32)
    return KVPage(("real", layer), layer, (0, T), K, V, "real_fp16", float(mass))


def _pages(masses):
    return [_page(i, m) for i, m in enumerate(masses)]


def test_gibbs_weights_normalized():
    w = gibbs_weights(_pages([1.0, 2.0, 3.0, 4.0]))
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert np.all(np.diff(w) > 0)          # monotone in attention_mass


def test_parity_groups_cover_all():
    groups = parity_groups(28, GROUP_SIZE)
    flat = [i for g in groups for i in g]
    assert flat == list(range(28))
    assert len(groups) == (28 + GROUP_SIZE - 1) // GROUP_SIZE


def test_concentrated_mass_thermo_cheaper():
    # one dominant layer carries almost all mass -> critical set is tiny -> thermo touches
    # far fewer groups than uniform.
    pages = _pages([50.0] + [0.1] * 27)     # 28 pages, 7 groups
    ub, tb, iso, bpb = allocate(pages)
    assert iso is True                      # both protect the identical critical set (CONTROL)
    assert tb < ub                          # thermo strictly cheaper
    assert bpb > 0


def test_diffuse_mass_costs_coincide():
    # When the critical set spans EVERY parity group, thermo cannot skip any group -> costs
    # coincide (ALLOWED-to-FAIL case, reported as-is). Equal masses + a high coverage target
    # forces the critical set to reach into all 7 groups.
    pages = _pages([1.0] * 28)
    ub, tb, iso, bpb = allocate(pages, mass_target=0.99)
    assert iso is True
    assert tb == ub


def test_critical_set_reaches_target():
    pages = _pages([10.0, 5.0, 1.0, 1.0])
    crit = critical_set(pages, mass_target=0.5)
    w = gibbs_weights(pages)
    assert sum(w[i] for i in crit) >= 0.5
    assert 0 in crit                        # highest-mass page always included


def test_verdict_line_exists(tmp_path):
    pages = _pages([50.0] + [0.1] * 27)
    ub, tb, iso, bpb = allocate(pages)
    rows = [("probe", len(pages), len(parity_groups(len(pages))),
             len(critical_set(pages)), ub, tb, iso)]
    p = tmp_path / "rep.md"
    write_parity_alloc_report(rows, ub * bpb, tb * bpb, iso, path=str(p))
    text = p.read_text(encoding="utf-8")
    assert "PARITY_ALLOC: uniform_cost=" in text
    assert "thermo_cost=" in text and "iso_protection=" in text


def test_mutation_broken_allocator_fails_control():
    # A correct thermo placement protects the whole critical set (iso=True). A DELIBERATELY
    # broken allocator that drops the parity block from one critical-touching group must leave
    # a critical page unrecoverable -> protected set != crit -> the control fails. Proves
    # iso_protection is not tautological.
    pages = _pages([50.0, 40.0, 0.1, 0.1] + [0.1] * 24)   # 28 pages, 7 groups
    groups = parity_groups(len(pages))
    crit = critical_set(pages)
    thermo_group_idx = {g for g, grp in enumerate(groups) if crit.intersection(grp)}
    assert thermo_group_idx                               # crit touches >=1 group

    good = {g: NUM_PARITY for g in thermo_group_idx}
    assert recoverable_critical(crit, groups, good) == crit    # correct: full protection

    dropped = next(iter(thermo_group_idx))
    broken = {g: NUM_PARITY for g in thermo_group_idx if g != dropped}
    assert recoverable_critical(crit, groups, broken) != crit  # broken: control fails

    # sanity: the honest allocate() still passes the control on the same input
    _, _, iso, _ = allocate(pages)
    assert iso is True


def test_sweep_temperatures_includes_headline_and_positive():
    pages = _pages([50.0, 40.0, 1.0, 1.0] + [0.5] * 24)
    temps = sweep_temperatures(pages)
    labels = [lbl for lbl, _ in temps]
    assert "kT=1" in labels and dict(temps)["kT=1"] == 1.0
    assert len(temps) >= 2                      # at least kT=1 + one non-degenerate temp
    assert all(t > 0 for _, t in temps)


def test_higher_kT_does_not_shrink_crit_set():
    # flattening the softmax (higher kT) can only keep or grow the critical set needed to reach
    # the same mass_target -> monotone non-decreasing in kT. This is the artifact-check property.
    pages = _pages([50.0] + [0.1] * 27)
    sizes = crit_sizes_by_temperature(pages, [("kT=1", 1.0), ("hot", 50.0)])
    by = {lbl: sz for lbl, _, sz in sizes}
    assert by["hot"] >= by["kT=1"]
    assert by["kT=1"] >= 1


def test_report_contains_sensitivity_table(tmp_path):
    pages = _pages([50.0] + [0.1] * 27)
    ub, tb, iso, bpb = allocate(pages)
    rows = [("probe", len(pages), len(parity_groups(len(pages))),
             len(critical_set(pages)), ub, tb, iso)]
    sens = [(lbl, t, sz) for lbl, t, sz in
            crit_sizes_by_temperature(pages, sweep_temperatures(pages))]
    assert len(sens) >= 2                        # >=2 temperatures reported
    p = tmp_path / "rep.md"
    write_parity_alloc_report(rows, ub * bpb, tb * bpb, iso, path=str(p), sensitivity=sens)
    text = p.read_text(encoding="utf-8")
    assert "## kT sensitivity" in text
    assert "| kT label | mean kT | mean crit_pages |" in text
    assert "KT_SENSITIVITY: crit_pages=" in text
    assert ("temperature-ROBUST" in text) or ("ARTIFACT" in text)


def test_topk_norm_selects_highest_mass():
    # physics-free: top-k by RAW attention_mass, deterministic tie-break.
    pages = _pages([5.0, 50.0, 1.0, 30.0, 0.1])
    assert topk_norm_set(pages, 2) == {1, 3}          # masses 50, 30
    assert topk_norm_set(pages, 1) == {1}


def test_topk_norm_coincides_with_thermo():
    # Softmax is order-preserving in attention_mass, so top-|crit| by raw mass == critical set;
    # topk_norm and thermo must select the identical set for the identical parity cost.
    pages = _pages([50.0, 40.0, 0.1, 0.1] + [0.1] * 24)
    crit = critical_set(pages)
    assert topk_norm_set(pages, len(crit)) == crit
    tk_blocks, tk_prot, th_prot, sets_id, bpb = allocate_topk(pages)
    _, thermo_blocks, _, _ = allocate(pages)
    assert sets_id is True                            # physics-free set == free-energy set
    assert tk_prot == th_prot                         # identical protected set
    assert tk_blocks == thermo_blocks                 # identical parity cost -> vocabulary decorative
    assert bpb > 0


def test_topk_mutation_broken_baseline_differs():
    # A DELIBERATELY broken baseline (protect the LOWEST-mass pages) must NOT coincide with the
    # free-energy set — proves sets_identical is a real check, not tautological.
    pages = _pages([50.0, 40.0, 0.1, 0.1] + [0.1] * 24)
    crit = critical_set(pages)
    order_low = sorted(range(len(pages)), key=lambda i: (float(pages[i].attention_mass), i))
    broken = set(order_low[:len(crit)])               # lowest-mass instead of highest
    assert broken != crit                             # control can fail


def test_baseline_parity_line_in_report(tmp_path):
    pages = _pages([50.0] + [0.1] * 27)
    ub, tb, iso, bpb = allocate(pages)
    tk_blocks, _, _, sets_id, _ = allocate_topk(pages)
    rows = [("probe", len(pages), len(parity_groups(len(pages))),
             len(critical_set(pages)), ub, tb, iso)]
    p = tmp_path / "rep.md"
    write_parity_alloc_report(rows, ub * bpb, tb * bpb, iso, path=str(p),
                              baseline=(tk_blocks * bpb, sets_id))
    text = p.read_text(encoding="utf-8")
    assert "BASELINE_PARITY: thermo_cost=" in text
    assert "topk_cost=" in text and "sets_identical=" in text
    assert "DECORATIVE" in text                        # coincidence -> decorative-vocabulary caveat


def test_allocate_deterministic():
    pages = _pages([50.0] + [0.1] * 27)
    a = allocate(pages)
    b = allocate(pages)
    assert a == b
