"""
Phase 9.1 — Task-accuracy axis.

Extends Phase 8.1 eval set from 30 to 100 probes: reuses EVAL_PROBES (30)
and appends 70 from allenai/sciq@validation rows 0-69, hardcoded here to
avoid pyarrow/torch DLL conflict on Windows (pyarrow crashes when loaded
after torch in the same process).

Dataset verified 2026-07-02 via standalone load_dataset call (no torch):
  id=allenai/sciq  split=validation  rows=1000
  prompt_field='question'  answer_field='correct_answer'
  Rows 0-69 extracted and hardcoded below. Field names verified at
  datasets.load_dataset call + row inspection (question:str, correct_answer:str).

Metric added: accuracy_retention = acc(B3) / acc(B0) per noise level,
reported alongside NLL. Does NOT replace NLL metric.

Emits: results/REPORT_phase9_accuracy.md

APIs reused (all verified in prior phases):
  - phase7_quality: HELD_OUT_PREFIX, HELD_OUT_CONT, _compute_nll, _inject_pages,
                    _total_kv_bits  (phase7_quality.py:69,87,101,46)
  - eval_set: EVAL_PROBES, normalized_match  (eval_set.py:33,79)
  - coding: encode_rs_erasure_group, recover_rs_erasure
  - lossy_tier: quant_noise
  - real_model_adapter: dynamiccache_to_pages, pages_to_kv_tensors
  - model.generate(input_ids, past_key_values, max_new_tokens=8, do_sample=False)
    verified Phase 7.4
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
# Dataset spec (for attribution and reproducibility audit)
# ---------------------------------------------------------------------------
SCIQ_DATASET_ID = "allenai/sciq"
SCIQ_SPLIT = "validation"
SCIQ_N_ROWS = 70  # rows 0..69 extracted 2026-07-02

# ---------------------------------------------------------------------------
# allenai/sciq@validation rows 0-69 — hardcoded to avoid pyarrow/torch conflict
# Verified source: load_dataset('allenai/sciq', split='validation')[0:70]
# Fields used: 'question' (prompt), 'correct_answer' (gold answer)
# ---------------------------------------------------------------------------
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
    ("Earthquakes, which may occur on california’s abundant faults, can also trigger what?", "landslides"),
    ("In which way do particles of water move in deep water?", "circles"),
    ("What is the name of the study of heat engines?", "thermodynamics"),
]

assert len(_SCIQ_RAW) == 70, f"Expected 70 sciq rows, got {len(_SCIQ_RAW)}"


def _sciq_probes() -> list[dict]:
    """Return 70 sciq probes formatted for normalized_match scoring."""
    return [
        {"prompt": f"{q} Answer with one or two words:", "expected": a}
        for q, a in _SCIQ_RAW
    ]


def build_extended_eval_set() -> list[dict]:
    """Return 30 EVAL_PROBES + 70 allenai/sciq rows = 100 probes total."""
    return list(EVAL_PROBES) + _sciq_probes()


# ---------------------------------------------------------------------------
# Noise levels and crossover threshold
# ---------------------------------------------------------------------------
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

# Crossover = max noise where retention >= this (a measurement threshold, not a gate)
RETENTION_CROSSOVER_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# B0 accuracy runner (clean KV)
# ---------------------------------------------------------------------------

def _run_accuracy_b0(model, tok, device: str, probes: list[dict]) -> float:
    """task_accuracy on clean KV for arbitrary probe list."""
    model.eval()
    correct = 0
    for probe in probes:
        ids = tok(probe["prompt"], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if normalized_match(pred, probe["expected"], probe.get("alternatives")):
            correct += 1
    return correct / len(probes)


# ---------------------------------------------------------------------------
# B3 accuracy runner (quant_noise → RS recover → inject)
# ---------------------------------------------------------------------------

def _run_accuracy_b3(
    model,
    tok,
    device: str,
    dtype,
    probes: list[dict],
    noise_level: float,
) -> float:
    """task_accuracy on AEPK-damaged KV for arbitrary probe list.

    Seed scheme: seed = 8000 + probe_idx * 100 + layer_idx
    Probes 0-29 reproduce Phase 8.1 seeds exactly;
    probes 30-99 extend naturally (deterministic, no collision).
    """
    model.eval()
    correct = 0
    for probe_idx, probe in enumerate(probes):
        ids = tok(probe["prompt"], return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            pfx_out = model(ids, use_cache=True)
        pkv = pfx_out.past_key_values
        pages = dynamiccache_to_pages(pkv)

        rs_group = encode_rs_erasure_group(pages, num_parity=1)

        damaged: list = []
        mses: list[float] = []
        for j, page in enumerate(pages):
            if noise_level == 0.0:
                damaged.append(page)
                mses.append(0.0)
            else:
                dam, mse = quant_noise(page, level=noise_level,
                                       seed=8000 + probe_idx * 100 + j)
                damaged.append(dam)
                mses.append(float(mse))

        if noise_level > 0.0:
            try:
                worst_idx = int(np.argmax(mses))
                worst_id = pages[worst_idx].page_id
                rec = recover_rs_erasure(rs_group, [worst_id])
                damaged[worst_idx] = rec[worst_id]
            except Exception:
                pass

        _inject_pages(pkv, damaged, dtype, device)

        with torch.no_grad():
            out = model.generate(ids, past_key_values=pkv, max_new_tokens=8,
                                 do_sample=False)
        pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if normalized_match(pred, probe["expected"], probe.get("alternatives")):
            correct += 1

    return correct / len(probes)


# ---------------------------------------------------------------------------
# NLL runner for B3 (same held-out text and RS settings as Phase 8.2)
# ---------------------------------------------------------------------------

def _run_b3_nll(model, tok, device: str, dtype, noise_level: float) -> float:
    """B3 NLL on held-out text at given noise_level (mirrors phase8_sweep)."""
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
    b3_accuracy: float
    b0_nll: float
    b3_nll: float
    nll_delta: float
    acc_delta: float
    retention: float  # b3_accuracy / b0_accuracy (1.0 if b0_accuracy==0)


@dataclass(frozen=True)
class Phase9AccuracyResult:
    points: list[AccuracyPoint]
    n_probes: int
    crossover_noise: float | None
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
) -> Phase9AccuracyResult:
    """Phase 9.1 accuracy sweep.

    Loads 100 probes (30 EVAL_PROBES + 70 allenai/sciq), runs B0 once,
    B3 at each noise level. Computes accuracy_retention and NLL per level.
    Writes results/REPORT_phase9_accuracy.md. Returns Phase9AccuracyResult.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    all_probes = build_extended_eval_set()
    n_probes = len(all_probes)

    # B0 — clean KV (once, independent of noise_level)
    prefix_ids = tok(HELD_OUT_PREFIX, return_tensors="pt").to(device)
    cont_ids = tok(HELD_OUT_CONT, return_tensors="pt").to(device)

    with torch.no_grad():
        pfx0 = model(**prefix_ids, use_cache=True)
    pkv0 = pfx0.past_key_values
    b0_nll = _compute_nll(model, tok, prefix_ids, cont_ids, pkv0, device)
    b0_accuracy = _run_accuracy_b0(model, tok, device, all_probes)

    # B3 at each noise level
    points: list[AccuracyPoint] = []
    for level in noise_levels:
        b3_nll = _run_b3_nll(model, tok, device, dtype, level)
        b3_accuracy = _run_accuracy_b3(model, tok, device, dtype, all_probes, level)

        retention = (b3_accuracy / b0_accuracy) if b0_accuracy > 0.0 else 1.0
        points.append(AccuracyPoint(
            noise_level=level,
            b0_accuracy=b0_accuracy,
            b3_accuracy=b3_accuracy,
            b0_nll=b0_nll,
            b3_nll=b3_nll,
            nll_delta=b3_nll - b0_nll,
            acc_delta=b3_accuracy - b0_accuracy,
            retention=retention,
        ))

    # Crossover: max noise where retention >= RETENTION_CROSSOVER_THRESHOLD
    crossover_noise: float | None = None
    retention_at_crossover: float | None = None
    for pt in sorted(points, key=lambda p: p.noise_level):
        if pt.retention >= RETENTION_CROSSOVER_THRESHOLD:
            crossover_noise = pt.noise_level
            retention_at_crossover = pt.retention

    report_path = os.path.join("results", "REPORT_phase9_accuracy.md")
    _write_report(points, n_probes, crossover_noise, retention_at_crossover, report_path)

    return Phase9AccuracyResult(
        points=points,
        n_probes=n_probes,
        crossover_noise=crossover_noise,
        retention_at_crossover=retention_at_crossover,
        report_path=report_path,
    )


# ---------------------------------------------------------------------------
# Report writer (deterministic — no time stamps, no random elements)
# ---------------------------------------------------------------------------

def _write_report(
    points: list[AccuracyPoint],
    n_probes: int,
    crossover_noise: float | None,
    retention_at_crossover: float | None,
    path: str,
) -> None:
    if crossover_noise is not None:
        crossover_str = str(crossover_noise)
        retention_str = f"{retention_at_crossover:.4f}"
    else:
        crossover_str = "none"
        retention_str = f"{max(pt.retention for pt in points):.4f}"

    lines = [
        "# REPORT_phase9_accuracy.md — Phase 9.1 Task-Accuracy Axis",
        "",
        "Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)",
        f"Eval set: {n_probes} probes (30 EVAL_PROBES + 70 allenai/sciq@validation rows 0-69)",
        "Dataset: allenai/sciq@validation  rows=1000  fields=question,correct_answer",
        f"Noise levels: {[pt.noise_level for pt in points]}",
        f"Retention crossover threshold: {RETENTION_CROSSOVER_THRESHOLD}",
        "",
        "## NLL vs Accuracy divergence",
        "Phase 7.4 found B3 answered 'Paris' CORRECT at +0.64 NLL.",
        "This report measures whether that pattern holds at scale (100 probes).",
        "NLL and task-accuracy may diverge in either direction — both reported honestly.",
        "",
        "## Per-level results",
        "",
        "| noise | B0_NLL | B3_NLL | ΔNLL | B0_acc | B3_acc | Δacc | retention |",
        "|-------|--------|--------|------|--------|--------|------|-----------|",
    ]
    for pt in points:
        lines.append(
            f"| {pt.noise_level:.2f} | {pt.b0_nll:.4f} | {pt.b3_nll:.4f} | "
            f"{pt.nll_delta:+.4f} | {pt.b0_accuracy:.3f} | {pt.b3_accuracy:.3f} | "
            f"{pt.acc_delta:+.3f} | {pt.retention:.4f} |"
        )
    lines += [
        "",
        "## Accuracy retention",
        f"retention = acc(B3)/acc(B0); crossover = max noise where retention >= "
        f"{RETENTION_CROSSOVER_THRESHOLD}",
        "",
        f"Crossover noise: {crossover_str}",
        f"Retention at crossover: {retention_str}",
        "",
        "Interpretation: accuracy-axis crossover may differ from NLL-axis crossover",
        "(Phase 8.2 NLL crossover = 0.2). Task accuracy and NLL measure different things.",
        "",
        "COMPUTE CAVEAT: RS encode/decode CPU time not measured (same caveat as Phase 7.4/8.2).",
        "",
        f"ACCURACY_AXIS: retention={retention_str} at crossover={crossover_str}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
