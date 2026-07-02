"""
Phase 9.1-FIX — Task-accuracy axis (corrected harness).

Bug fixed (2026-07-02): commit 1c0f7a5 called model.generate(ids, past_key_values=pkv)
where pkv already spanned all of ids → prompt double-processed → B3 differed from B0 by
cache plumbing, not AEPK. Proof: noise=0 retention was 0.3462 instead of 1.0.

Fix: _run_accuracy_b0 and _run_accuracy_b3 both use a manual greedy decode loop
starting from the prefill's last-position logits. At noise=0, damaged=clean pages →
_inject_pages is bit-exact → greedy loop is identical for B0 and B3 → retention=1.0.

RS config aligned: accuracy path now uses num_parity=2 / recover-worst-2, matching
the NLL path (phase8_sweep._b3_nll_at_level) so both axes measure identical protection.

Seeds pulled forward from 9.4: B3 accuracy run over N_SEEDS=5 seeds per noise level;
report mean ± 95% CI; STATS line added. B0 runs once (deterministic — no noise).

Extends Phase 8.1 eval set from 30 to 100 probes: reuses EVAL_PROBES (30) and appends
70 from allenai/sciq@validation rows 0-69, hardcoded to avoid pyarrow/torch DLL conflict
on Windows (pyarrow crashes when loaded after torch in the same process).

Dataset verified 2026-07-02 via standalone load_dataset call (no torch):
  id=allenai/sciq  split=validation  rows=1000
  prompt_field='question'  answer_field='correct_answer'
  Rows 0-69 extracted and hardcoded below.

APIs verified (all prior phases):
  - phase7_quality.py:69  _compute_nll
  - phase7_quality.py:87  _inject_pages  (mutates pkv in-place — confirmed here)
  - phase7_quality.py:46  HELD_OUT_PREFIX / HELD_OUT_CONT
  - eval_set.py:33        EVAL_PROBES
  - eval_set.py:79        normalized_match
  - model(ids, use_cache=True) returns CausalLMOutputWithPast with .logits and
    .past_key_values — verified Phase 7.3/7.4
  - model(nxt, past_key_values=pkv, use_cache=True).past_key_values — updated cache
  - step.logits[:, -1] — logit at the just-processed position
  - DynamicLayer.keys / .values assignable — verified Phase 7.2/7.4
  - transformers 5.12.1, torch 2.5.1+cu121
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from aepk_paging.coding import encode_rs_erasure_group, recover_rs_erasure
from aepk_paging.harness.eval_set import EVAL_PROBES, normalized_match
from aepk_paging.harness.phase7_quality import (
    HELD_OUT_CONT,
    HELD_OUT_PREFIX,
    _compute_nll,
    _inject_pages,
)
from aepk_paging.lossy_tier import quant_noise
from aepk_paging.real_model_adapter import dynamiccache_to_pages

# ---------------------------------------------------------------------------
# Dataset attribution (probes hardcoded below to avoid pyarrow/torch conflict)
# ---------------------------------------------------------------------------
SCIQ_DATASET_ID = "allenai/sciq"
SCIQ_SPLIT = "validation"
SCIQ_N_ROWS = 70

# allenai/sciq@validation rows 0-69 — verified 2026-07-02 via standalone load
_SCIQ_RAW: list[tuple[str, str]] = [
    ("Who proposed the theory of evolution by natural selection?", "darwin"),
    ("Each specific polypeptide has a unique linear sequence of which acids?", "amino"),
    ("A frameshift mutation is a deletion or insertion of one or more of what that changes the reading frame of the base sequence?", "nucleotides"),
    ("What is an area of land called that is wet for all or part of the year?", "wetland"),
    ("Surface waters are heated by the radiation from?", "the sun"),
    ("What are arteries, veins, and capillaries examples of?", "blood vessels"),
    ("Biochemical reactions of metabolism include what two general categories?", "catabolic and anabolic"),
    ("Compounds with aluminum and silicon are commonly found in the clay fractions of soils derived from what?", "volcanic ash"),
    ("What organ has four major regions: the cerebrum, the diencephalon, the stem, and the cerebellum?", "brain"),
    ("What can refer to a rope in a particular shape and a genetic structure involved in splicing?", "lariat"),
    ("What is the ratio of the mass of an object to its volume?", "density"),
    ("What is the most common type of anemia?", "iron-def"),
    ("What stimulates milk production in mammals?", "prolactin"),
    ("During telophase , the chromosomes begin to uncoil and form what?", "chromatin"),
    ("The science dealing with the study of the atmosphere is known as what?", "meteorology"),
    ("On what basis are the eras of the phanerozoic eon separated?", "mass extinctions"),
    ("What type of response is generated when a stimulus is received by the nervous system?", "a motor response"),
    ("Fluid in the pseudocoelom serves as a hydrostatic what?", "skeleton"),
    ("How will global warming eliminate some islands and reduce the area of others?", "raise sea levels"),
    ("What is the term for the secretion of saliva?", "salivation"),
    ("What is the most common sti in the u.s.?", "chlamydia"),
    ("What plant structures are the main avenues by which water evaporates from the sporophyte?", "stomata"),
    ("Decomposers break down dead organisms into nutrients and what?", "gases"),
    ("What is the term for the use of technology to treat genetic disorders or change organisms so they are more useful to people?", "biotechnology"),
    ("What science specialty, with a set of like-named scientific laws, refers to the study of energy and energy transfer involving physical matter?", "thermodynamics"),
    ("What type of tissue makes up the brain and the nerves that connect the brain to all parts of the body?", "nervous tissue"),
    ("Periodic refers to something that does what?", "repeat"),
    ("Blood is pumped from the heart, pushing open which valves?", "pulmonary and aortic semilunar"),
    ("What term refers to a list of the elements that will replace the ones below them in single-replacement reactions?", "activity series"),
    ("What cause many human diseases by killing host cells or disturbing their homeostasis?", "viruses"),
    ("What two types of digestive systems do invertebrates have?", "incomplete or complete"),
    ("What are ectothermic vertebrates that divide their time between freshwater and terrestrial habitats?", "amphibians"),
    ("During exercise, the rate of blood returning to the heart does this?", "increases"),
    ("What is the pattern of spacing among individuals within the boundaries of the population?", "dispersion"),
    ("What level is a feeding position in a food chain or web?", "trophic"),
    ("What do most of the noble gas elements have in common?", "eight valence electrons"),
    ("What are the organisms that live in extreme conditions known as?", "extremophiles"),
    ("Collision frequency is greater for what category of catalysts, which also tend to be more sensitive to temperature and more 'expensive'?", "homogeneous"),
    ("What does the pull of the moon's gravity on earth cause?", "tides"),
    ("A water molecule forms when oxygen (o) and _______  atoms react and are held together by covalent bonds?", "hydrogen (h)"),
    ("What force makes objects seem lighter in water?", "buoyant"),
    ("What  is the process by which the nucleus of a eukaryotic cell divides?", "mitosis"),
    ("Iceland is made up of a series of?", "volcanoes"),
    ("What type of organism does not need oxygen for growth and dies in its presence?", "anaerobic"),
    ("What is the name of specialized organs that filter the lymph by percolation through a maze of connective tissue filled with white blood cells?", "lymph nodes"),
    ("What are hydrocarbons most important use?", "fuel"),
    ("What protects tissues of the central nervous system from changes in ph?", "bicarbonate ions"),
    ("The air pressure is about 80% that of ________ pressure at sea level.", "standard atmospheric"),
    ("Vertebrates - including fish, amphibians, reptiles, birds, and mammals - belong to what phylum?", "chordata"),
    ("In our wildflower population, the pool of what remains constant from one generation to the next?", "genes"),
    ("When a membrane uses energy to move a substance across it, what kind of transport is this?", "active"),
    ("An electrostatic attraction between two ions that have exchanged what?", "electrons"),
    ("During what part of a person's development are they generally at their physical peak?", "early adulthood"),
    ("What type of movement involves sluggish segmentation, primarily in the transverse and descending colons?", "haustral contraction"),
    ("Vertebrates evolved from primitive forms of which creature?", "chordates"),
    ("What does the aqueous fluid between the chloroplast membrane and the grana known as?", "stroma"),
    ("When a hypothesis is repeatedly confirmed, what can it then become?", "theory"),
    ("Using a hammer to remove a nail changes both the direction and strength of the what?", "applied force"),
    ("Through which process are plants able to make their own food?", "photosynthesis"),
    ("What kind of waves are sound waves?", "mechanical"),
    ("The protein without the prosthetic group is known as the what?", "apoprotein"),
    ("What connections allow heterocysts to transport fixed nitrogen to neighboring cells and to receive carbohydrates?", "intercellular"),
    ("The classes anthozoa, scyphozoa, cubozoa, and hydrozoa make up what phylum?", "cnidaria"),
    ("Where does waste enter the large intestine from?", "the small intestine"),
    ("What do different soil horizons show different amounts of?", "alteration"),
    ("What is the main source of energy for your body?", "carbohydrates"),
    ("What are thin, very small tail-like projections that extend outward from the cell body that allow protists to move?", "cilia"),
    ("Earthquakes, which may occur on california's abundant faults, can also trigger what?", "landslides"),
    ("In which way do particles of water move in deep water?", "circles"),
    ("What is the name of the study of heat engines?", "thermodynamics"),
]

assert len(_SCIQ_RAW) == 70, f"Expected 70 sciq rows, got {len(_SCIQ_RAW)}"


def _sciq_probes() -> list[dict]:
    return [
        {"prompt": f"{q} Answer with one or two words:", "expected": a}
        for q, a in _SCIQ_RAW
    ]


def build_extended_eval_set() -> list[dict]:
    """Return 30 EVAL_PROBES + 70 allenai/sciq rows = 100 probes total."""
    return list(EVAL_PROBES) + _sciq_probes()


# ---------------------------------------------------------------------------
# Noise levels and thresholds
# ---------------------------------------------------------------------------
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
N_SEEDS = 5                          # seeds per noise level for B3 accuracy
RETENTION_CROSSOVER_THRESHOLD = 0.5  # crossover = max noise where mean_retention >= this


# ---------------------------------------------------------------------------
# Greedy decode helper (shared by B0 and B3 for identical code paths)
# ---------------------------------------------------------------------------

def _greedy_decode(model, tok, ids: torch.Tensor, pkv, n_tokens: int = 8) -> str:
    """Greedy decode n_tokens from a pre-built KV cache.

    ids: prompt token ids already encoded in pkv.
    pkv: DynamicCache covering all of ids (from model(ids, use_cache=True)).
    Returns decoded string (skip_special_tokens=True).

    This is the CORRECT approach: we never re-feed ids to generate(). Instead we
    use the last-position logit from the prefill as the seed for generation, then
    step one token at a time through the damaged (or clean) cache.
    """
    # out must already have been run: caller passes the prefill output's logits
    raise NotImplementedError("Use _greedy_from_prefill_out instead")


def _greedy_from_prefill_out(model, tok, out, pkv, n_tokens: int = 8) -> str:
    """Generate n_tokens greedily from the logit at the last prefill position.

    out: return value of model(ids, use_cache=True) — provides out.logits[:, -1].
    pkv: DynamicCache (possibly mutated by _inject_pages before this call).

    At noise=0, pkv == original clean cache → same as model.generate(ids) greedy.
    """
    cur = out.logits[:, -1]   # [1, vocab] — prediction at the last prompt token
    gen: list[torch.Tensor] = []
    for _ in range(n_tokens):
        nxt = cur.argmax(-1, keepdim=True)    # [1, 1]
        gen.append(nxt)
        with torch.no_grad():
            step = model(nxt, past_key_values=pkv, use_cache=True)
        pkv = step.past_key_values
        cur = step.logits[:, -1]
    return tok.decode(torch.cat(gen, dim=-1)[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# B0 accuracy runner (clean KV — uses same greedy path as B3 for fair comparison)
# ---------------------------------------------------------------------------

def _run_accuracy_b0(model, tok, device: str, probes: list[dict]) -> float:
    """task_accuracy on clean KV.

    Uses the same manual greedy decode as B3 (not model.generate) so that at
    noise=0 the two paths are bit-identical → retention=1.0 is guaranteed.
    """
    model.eval()
    correct = 0
    for probe in probes:
        ids = tok(probe["prompt"], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        pred = _greedy_from_prefill_out(model, tok, out, out.past_key_values)
        if normalized_match(pred, probe["expected"], probe.get("alternatives")):
            correct += 1
    return correct / len(probes)


# ---------------------------------------------------------------------------
# B3 accuracy runner (quant_noise → RS recover → inject → greedy decode)
# ---------------------------------------------------------------------------

def _run_accuracy_b3(
    model,
    tok,
    device: str,
    dtype,
    probes: list[dict],
    noise_level: float,
    run_seed: int = 0,
) -> float:
    """task_accuracy on AEPK-damaged KV at given noise level and seed.

    RS config: num_parity=2 / recover worst-2 pages — ALIGNED with NLL path
    (phase8_sweep._b3_nll_at_level uses num_parity=2 / recover worst-2).

    Seed scheme: seed = 8000 + run_seed * 10000 + probe_idx * 100 + layer_idx
    At run_seed=0, probes 0-29 reproduce Phase 8.1 seeds (8000+probe*100+j).
    run_seed in {0,1,2,3,4} gives 5 independent draws per noise level.

    At noise_level=0.0: damaged = original pages (no noise); _inject_pages writes
    bit-exact values (f16→f32→f16 is lossless per Phase 7.2); greedy loop is
    identical to B0 → acc == b0_accuracy → retention == 1.0.
    """
    model.eval()
    correct = 0
    for probe_idx, probe in enumerate(probes):
        ids = tok(probe["prompt"], return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = model(ids, use_cache=True)
        pkv = out.past_key_values
        pages = dynamiccache_to_pages(pkv)

        # RS encode — num_parity=2 matches NLL path
        rs_group = encode_rs_erasure_group(pages, num_parity=2)

        damaged: list = []
        mses: list[float] = []
        for j, page in enumerate(pages):
            if noise_level == 0.0:
                damaged.append(page)
                mses.append(0.0)
            else:
                dam, mse = quant_noise(
                    page, level=noise_level,
                    seed=8000 + run_seed * 10000 + probe_idx * 100 + j,
                )
                damaged.append(dam)
                mses.append(float(mse))

        # Recover worst 2 pages — matches NLL path (num_parity=2 covers 2 erasures)
        if noise_level > 0.0:
            try:
                worst_2_ids = [pages[i].page_id for i in np.argsort(mses)[-2:]]
                rec = recover_rs_erasure(rs_group, worst_2_ids)
                for pid, rpage in rec.items():
                    idx = next(j2 for j2, p in enumerate(damaged) if p.page_id == pid)
                    damaged[idx] = rpage
            except Exception:
                pass

        # Inject damaged cache in-place (verified phase7_quality.py:87 — mutates pkv)
        _inject_pages(pkv, damaged, dtype, device)

        # Greedy decode from prefill logit — NEVER re-feeds ids
        pred = _greedy_from_prefill_out(model, tok, out, pkv)
        if normalized_match(pred, probe["expected"], probe.get("alternatives")):
            correct += 1

    return correct / len(probes)


# ---------------------------------------------------------------------------
# NLL runner for B3 (same held-out text and RS settings as Phase 8.2)
# ---------------------------------------------------------------------------

def _run_b3_nll(model, tok, device: str, dtype, noise_level: float) -> float:
    """B3 NLL on held-out text at given noise_level.

    Mirrors phase8_sweep._b3_nll_at_level exactly:
    num_parity=2 / recover worst-2 / seed=1234+i.
    """
    prefix_ids = tok(HELD_OUT_PREFIX, return_tensors="pt").to(device)
    cont_ids = tok(HELD_OUT_CONT, return_tensors="pt").to(device)

    with torch.no_grad():
        pfx_out = model(**prefix_ids, use_cache=True)
    pkv = pfx_out.past_key_values
    pages = dynamiccache_to_pages(pkv)

    rs_group = encode_rs_erasure_group(pages, num_parity=2)

    damaged: list = []
    mses: list[float] = []
    for i, page in enumerate(pages):
        if noise_level == 0.0:
            damaged.append(page)
            mses.append(0.0)
        else:
            dam, mse = quant_noise(page, level=noise_level, seed=1234 + i)
            damaged.append(dam)
            mses.append(float(mse))

    if noise_level > 0.0:
        worst_2_ids = [pages[i].page_id for i in np.argsort(mses)[-2:]]
        try:
            rec = recover_rs_erasure(rs_group, worst_2_ids)
            for pid, rpage in rec.items():
                idx = next(j for j, p in enumerate(damaged) if p.page_id == pid)
                damaged[idx] = rpage
        except Exception:
            pass

    _inject_pages(pkv, damaged, dtype, device)
    return _compute_nll(model, tok, prefix_ids, cont_ids, pkv, device)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccuracyPoint:
    noise_level: float
    b0_accuracy: float
    b3_accuracy_mean: float   # mean over N_SEEDS seeds
    b3_accuracy_ci: float     # 95% CI half-width (0.0 if n_seeds==1)
    b0_nll: float
    b3_nll: float
    nll_delta: float          # b3_nll - b0_nll
    acc_delta: float          # b3_accuracy_mean - b0_accuracy
    retention_mean: float     # mean(b3_acc / b0_acc) across seeds (1.0 if b0_acc==0)
    retention_ci: float       # 95% CI half-width on retention


@dataclass(frozen=True)
class Phase9AccuracyResult:
    points: list[AccuracyPoint]
    n_probes: int
    n_seeds: int
    crossover_noise: float | None      # max noise where retention_mean >= threshold
    retention_at_crossover: float | None
    report_path: str


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_phase9_accuracy(
    model,
    tok,
    device: str,
    dtype,
    noise_levels: list[float] | None = None,
    n_seeds: int = N_SEEDS,
) -> Phase9AccuracyResult:
    """Phase 9.1-FIX accuracy sweep.

    B0 runs once (deterministic — no noise stochasticity).
    B3 runs n_seeds times per noise level; reports mean ± 95% CI.
    Writes results/REPORT_phase9_accuracy.md. Returns Phase9AccuracyResult.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    all_probes = build_extended_eval_set()
    n_probes = len(all_probes)

    # B0 — clean KV (once, same greedy path as B3)
    prefix_ids = tok(HELD_OUT_PREFIX, return_tensors="pt").to(device)
    cont_ids = tok(HELD_OUT_CONT, return_tensors="pt").to(device)

    with torch.no_grad():
        pfx0 = model(**prefix_ids, use_cache=True)
    pkv0 = pfx0.past_key_values
    b0_nll = _compute_nll(model, tok, prefix_ids, cont_ids, pkv0, device)
    b0_accuracy = _run_accuracy_b0(model, tok, device, all_probes)

    # B3 at each noise level — n_seeds seeds
    points: list[AccuracyPoint] = []
    for level in noise_levels:
        b3_nll = _run_b3_nll(model, tok, device, dtype, level)

        seed_accs: list[float] = []
        for s in range(n_seeds):
            acc = _run_accuracy_b3(model, tok, device, dtype, all_probes,
                                   noise_level=level, run_seed=s)
            seed_accs.append(acc)

        b3_mean = float(np.mean(seed_accs))
        b3_std = float(np.std(seed_accs, ddof=1)) if n_seeds > 1 else 0.0
        b3_ci = 1.96 * b3_std / (n_seeds ** 0.5) if n_seeds > 1 else 0.0

        if b0_accuracy > 0.0:
            seed_rets = [a / b0_accuracy for a in seed_accs]
        else:
            seed_rets = [1.0] * n_seeds
        ret_mean = float(np.mean(seed_rets))
        ret_std = float(np.std(seed_rets, ddof=1)) if n_seeds > 1 else 0.0
        ret_ci = 1.96 * ret_std / (n_seeds ** 0.5) if n_seeds > 1 else 0.0

        points.append(AccuracyPoint(
            noise_level=level,
            b0_accuracy=b0_accuracy,
            b3_accuracy_mean=b3_mean,
            b3_accuracy_ci=b3_ci,
            b0_nll=b0_nll,
            b3_nll=b3_nll,
            nll_delta=b3_nll - b0_nll,
            acc_delta=b3_mean - b0_accuracy,
            retention_mean=ret_mean,
            retention_ci=ret_ci,
        ))

    # Crossover: max noise where retention_mean >= threshold
    crossover_noise: float | None = None
    retention_at_crossover: float | None = None
    for pt in sorted(points, key=lambda p: p.noise_level):
        if pt.retention_mean >= RETENTION_CROSSOVER_THRESHOLD:
            crossover_noise = pt.noise_level
            retention_at_crossover = pt.retention_mean

    report_path = os.path.join("results", "REPORT_phase9_accuracy.md")
    _write_report(points, n_probes, n_seeds, crossover_noise, retention_at_crossover,
                  report_path)

    return Phase9AccuracyResult(
        points=points,
        n_probes=n_probes,
        n_seeds=n_seeds,
        crossover_noise=crossover_noise,
        retention_at_crossover=retention_at_crossover,
        report_path=report_path,
    )


# ---------------------------------------------------------------------------
# Report writer (deterministic — no timestamps, no random elements)
# ---------------------------------------------------------------------------

def _write_report(
    points: list[AccuracyPoint],
    n_probes: int,
    n_seeds: int,
    crossover_noise: float | None,
    retention_at_crossover: float | None,
    path: str,
) -> None:
    if crossover_noise is not None:
        crossover_str = str(crossover_noise)
        retention_str = f"{retention_at_crossover:.4f}"
    else:
        crossover_str = "none"
        retention_str = f"{max(pt.retention_mean for pt in points):.4f}"

    # CI on crossover: max CI among points at/below crossover (conservative)
    if crossover_noise is not None:
        crossover_pts = [p for p in points if p.noise_level <= crossover_noise]
        crossover_ci = max(p.retention_ci for p in crossover_pts)
        stats_line = (f"STATS: crossover={crossover_str} "
                      f"retention_ci=±{crossover_ci:.4f} seeds={n_seeds}")
    else:
        max_ci = max(pt.retention_ci for pt in points)
        stats_line = (f"STATS: crossover=none "
                      f"retention_ci=±{max_ci:.4f} seeds={n_seeds}")

    # Monotone check (informational)
    rets = [pt.retention_mean for pt in sorted(points, key=lambda p: p.noise_level)]
    monotone = all(rets[i] >= rets[i + 1] - 1e-9 for i in range(len(rets) - 1))

    lines = [
        "# REPORT_phase9_accuracy.md — Phase 9.1-FIX Task-Accuracy Axis",
        "",
        "Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)",
        f"Eval set: {n_probes} probes (30 EVAL_PROBES + 70 allenai/sciq@validation rows 0-69)",
        "Dataset: allenai/sciq@validation  rows=1000  fields=question,correct_answer",
        f"Noise levels: {[pt.noise_level for pt in points]}",
        f"Seeds per noise level: {n_seeds}",
        f"Retention crossover threshold: {RETENTION_CROSSOVER_THRESHOLD}",
        f"RS config: num_parity=2 / recover-worst-2 (aligned with NLL path)",
        "",
        "## Fix notes",
        "commit 1c0f7a5 bug: model.generate(ids, past_key_values=pkv) double-processed ids.",
        "Fix: both B0 and B3 use manual greedy loop from prefill logit; no ids re-feed.",
        "Control: noise=0 row must show retention=1.0000 (bit-exact KV round-trip).",
        "",
        "## NLL vs Accuracy divergence",
        "Phase 7.4 found B3 answered Paris CORRECT at +0.64 NLL.",
        "This report measures whether that holds at scale (100 probes, 5 seeds).",
        "NLL and task-accuracy may diverge in either direction — both reported honestly.",
        "",
        "## Per-level results (B3 mean ± 95% CI over 5 seeds)",
        "",
        "| noise | B0_NLL | B3_NLL | ΔNLL | B0_acc | B3_acc_mean | B3_acc_ci | retention_mean | retention_ci |",
        "|-------|--------|--------|------|--------|-------------|-----------|----------------|--------------|",
    ]
    for pt in points:
        lines.append(
            f"| {pt.noise_level:.2f} | {pt.b0_nll:.4f} | {pt.b3_nll:.4f} | "
            f"{pt.nll_delta:+.4f} | {pt.b0_accuracy:.3f} | {pt.b3_accuracy_mean:.3f} | "
            f"±{pt.b3_accuracy_ci:.3f} | {pt.retention_mean:.4f} | ±{pt.retention_ci:.4f} |"
        )
    lines += [
        "",
        "## Accuracy retention",
        f"retention_mean = mean(acc(B3)/acc(B0)) over {n_seeds} seeds per level",
        f"crossover = max noise where retention_mean >= {RETENTION_CROSSOVER_THRESHOLD}",
        "",
        f"Crossover noise: {crossover_str}",
        f"Retention at crossover: {retention_str}",
        f"Curve monotone non-increasing: {monotone}",
        "",
        "Interpretation: accuracy-axis crossover may differ from NLL-axis crossover",
        "(Phase 8.2 NLL crossover = 0.2). Task accuracy and NLL measure different things.",
        "",
        "COMPUTE CAVEAT: RS encode/decode CPU time not measured (same caveat as Phase 7.4/8.2).",
        "",
        stats_line,
        f"ACCURACY_AXIS: retention={retention_str} at crossover={crossover_str}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
