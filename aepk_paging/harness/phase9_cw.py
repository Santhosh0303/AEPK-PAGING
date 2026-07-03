"""
Phase 9-CW — the confident-wrong error-regime novelty test.

The project's NOVELTY claim (RESEARCH_LOG): content-agnostic *physics* detection
catches KV corruption that the model's own *logprob/confidence* misses (the
"confident-wrong" blind spot), where token-semantic detectors are blind.

Phases 9.1-9.3 showed the ERROR regime (quant_noise) barely dents long-context
task accuracy, so there was nothing to heal and the ablation was within noise.
9-CW asks the sharper question: does a corruption exist that simultaneously
  (a) DROPS real task accuracy (answer becomes wrong),
  (b) is INVISIBLE to the model's confidence (output entropy stays ~clean), and
  (c) IS flagged by a calibrated content-agnostic physics fingerprint?
If yes -> the novelty is demonstrated. If no -> honest negative.

DEEP-ROOT AUDIT (2026-07-03, why this file does NOT reuse detect.py detectors):
  Real-model KVPage.K is 3D [seq_len, num_kv_heads, head_dim] (real_model_adapter).
  * detect.attention_distribution does norm(K, axis=1) -> norms over the HEADS
    axis -> returns a [T, head_dim] matrix, not a per-token [T] distribution (FLAW A).
  * detect.attention_mass_detector(expected_mass=None) compares attention_mass()
    (a softmax mass in (0,1)) against page.attention_mass (a mean key-norm ~tens)
    -> deviation ~= the mean norm -> flags EVERY clean page (FLAW B, verified).
  Both are degenerate on 3D pages. This file defines correct, flatten-aware,
  self-consistent fingerprints and calibrates their thresholds on clean pages
  (FPR control) so "physics flags it" is a meaningful statement, not an artifact.

Honesty spine (S9): zero edits to Phase 2-5 source. Verdict lines are runtime
expressions. Nothing is tuned to force SHOWN; a NOT_SHOWN result is reported as is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Correct, flatten-aware physics fingerprints (self-consistent scalars).
# All operate on a KVPage whose K/V may be 2D [T, F] or 3D [T, H, D].
# ---------------------------------------------------------------------------

def _flat_K(page) -> np.ndarray:
    K = np.asarray(page.K, dtype=np.float32)
    return K.reshape(K.shape[0], -1)          # [T, H*D] per-token key vectors


def _flat_V(page) -> np.ndarray:
    V = np.asarray(page.V, dtype=np.float32)
    return V.reshape(V.shape[0], -1)


def fp_key_norm_mean(page) -> float:
    """Mean per-token key-norm over the FULL flattened key vector.
    Matches real_model_adapter's attention_mass convention."""
    return float(np.linalg.norm(_flat_K(page), axis=1).mean())


def fp_key_mass(page, *, top_fraction: float = 0.5) -> float:
    """CORRECT attention-mass: softmax over per-token key-norms (flatten first),
    summed over the leading top_fraction of tokens. Scalar in (0, 1]."""
    norms = np.linalg.norm(_flat_K(page), axis=1)          # [T]
    shifted = norms - norms.max()
    w = np.exp(shifted)
    w = w / w.sum()
    keep = max(1, int(np.ceil(w.shape[0] * top_fraction)))
    return float(np.sum(np.sort(w)[::-1][:keep]))          # mass of the heaviest tokens


def fp_norm_ratio(page) -> float:
    """||K|| / ||V|| over the whole page (order-independent; 3D-safe)."""
    k = float(np.linalg.norm(_flat_K(page)))
    v = float(np.linalg.norm(_flat_V(page)))
    return k / (v + 1e-12)


def fp_v_mean_shift(page) -> float:
    """Norm of the per-feature MEAN value vector across tokens (a DC-offset
    fingerprint). Random V averages toward 0, so this is small on clean pages;
    a COHERENT bias added to every token spikes it. Catches norm-preserving
    directional corruption that fp_norm_ratio is blind to (audit finding 9-CW)."""
    return float(np.linalg.norm(_flat_V(page).mean(axis=0)))


def fp_k_mean_shift(page) -> float:
    """DC-offset fingerprint on K (directional key corruption)."""
    return float(np.linalg.norm(_flat_K(page).mean(axis=0)))


FINGERPRINTS: dict[str, Callable[[object], float]] = {
    "key_norm_mean": fp_key_norm_mean,   # key magnitude
    "key_mass": fp_key_mass,             # attention concentration
    "norm_ratio": fp_norm_ratio,        # K/V balance
    "v_mean_shift": fp_v_mean_shift,    # directional/DC value corruption
    "k_mean_shift": fp_k_mean_shift,    # directional/DC key corruption
}


# ---------------------------------------------------------------------------
# Threshold calibration (FPR control): a corruption "flags" on fingerprint f
# only if |f(corrupt) - f(clean)| exceeds the NATURAL page-to-page spread of f
# across the clean pages, scaled by `sigma_mult`. Clean-vs-clean deviation is 0,
# so the clean false-positive rate is 0 by construction; the calibrated bar is
# the natural variation the corruption must exceed to be genuinely detectable.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    tau: dict[str, float]          # per-fingerprint detection threshold
    clean_spread: dict[str, float] # natural std across clean pages (for the report)


def calibrate(clean_pages: list, *, sigma_mult: float = 3.0) -> Calibration:
    tau: dict[str, float] = {}
    spread: dict[str, float] = {}
    for name, fp in FINGERPRINTS.items():
        vals = np.array([fp(p) for p in clean_pages], dtype=np.float64)
        s = float(vals.std()) if vals.size > 1 else 0.0
        spread[name] = s
        # floor avoids a degenerate tau=0 when all clean pages are identical
        tau[name] = max(sigma_mult * s, 1e-6 * (abs(float(vals.mean())) + 1.0))
    return Calibration(tau=tau, clean_spread=spread)


def physics_flags(clean_page, corrupt_page, calib: Calibration) -> dict[str, bool]:
    """Per-fingerprint: does the corruption move it beyond the calibrated bar?"""
    out: dict[str, bool] = {}
    for name, fp in FINGERPRINTS.items():
        dev = abs(fp(corrupt_page) - fp(clean_page))
        out[name] = bool(dev > calib.tau[name])
    return out


def any_physics_flag(clean_page, corrupt_page, calib: Calibration) -> bool:
    return any(physics_flags(clean_page, corrupt_page, calib).values())


# ---------------------------------------------------------------------------
# STRUCTURED corruptions (NOT broadband quant_noise). These are the candidates
# for confident-wrong: coherent perturbations that redirect the read-out toward
# a WRONG but self-consistent answer (so the model stays confident) while moving
# a physics fingerprint. Each returns a NEW KVPage; deterministic given inputs.
# ---------------------------------------------------------------------------

def _rebuild(page, K, V, tag):
    from aepk_paging.kv_page import KVPage
    return KVPage(
        page_id=page.page_id, layer=page.layer, token_range=page.token_range,
        K=K.astype(np.float32), V=V.astype(np.float32),
        precision_tag=f"{page.precision_tag}+{tag}",
        attention_mass=page.attention_mass,   # keep the STORED clean baseline
    )


def corrupt_k_scale(page, factor: float):
    """Scale all keys by `factor`. Concentrates (factor>1) or flattens (factor<1)
    attention over this page's tokens -> shifts key_mass / key_norm_mean, and
    redirects real attention. Coherent (single scalar) -> answer can flip while
    the output distribution stays low-entropy (confident-wrong candidate)."""
    return _rebuild(page, _as3d(page, np.asarray(page.K, np.float32) * np.float32(factor)),
                    np.asarray(page.V, np.float32), f"k_scale{factor}")


def corrupt_v_scale(page, factor: float):
    """Scale all values by `factor`. Amplifies/attenuates this page's readout
    contribution -> shifts norm_ratio; changes the generated content coherently."""
    return _rebuild(page, np.asarray(page.K, np.float32),
                    _as3d(page, np.asarray(page.V, np.float32) * np.float32(factor)),
                    f"v_scale{factor}")


def corrupt_v_bias(page, magnitude: float, seed: int):
    """Add a single FIXED direction (not per-element noise) to every value vector.
    Coherent bias -> the read-out is pushed consistently toward a wrong region,
    keeping the model confident. Shifts norm_ratio; leaves key_mass untouched."""
    V = np.asarray(page.V, np.float32)
    flat = V.reshape(V.shape[0], -1)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=flat.shape[1]).astype(np.float32)
    direction /= (np.linalg.norm(direction) + 1e-12)
    biased = flat + np.float32(magnitude) * direction        # same dir every token
    return _rebuild(page, np.asarray(page.K, np.float32),
                    biased.reshape(V.shape), f"v_bias{magnitude}")


def _as3d(page, flat_or_same):
    """Return an array matching page.K/V's original shape (no-op if already same)."""
    target = np.asarray(page.K).shape
    a = np.asarray(flat_or_same, np.float32)
    return a.reshape(target) if a.shape != target else a


CORRUPTIONS = {
    "k_scale_0.5":  lambda p, s: corrupt_k_scale(p, 0.5),
    "k_scale_2.0":  lambda p, s: corrupt_k_scale(p, 2.0),
    "v_scale_0.5":  lambda p, s: corrupt_v_scale(p, 0.5),
    "v_scale_2.0":  lambda p, s: corrupt_v_scale(p, 2.0),
    "v_bias_1.0":   lambda p, s: corrupt_v_bias(p, 1.0, s),
}


# ---------------------------------------------------------------------------
# Confidence from REAL model logits (NOT the toy fixed_kv_readout_logits).
# ---------------------------------------------------------------------------

def token_entropy(logits_row: np.ndarray) -> float:
    """Shannon entropy (nats) of the next-token distribution from real logits."""
    x = np.asarray(logits_row, dtype=np.float64)
    x = x - x.max()
    w = np.exp(x)
    p = w / w.sum()
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


# ===========================================================================
# GPU sweep — real Qwen2.5 confident-wrong measurement.
# Requires torch + a loaded model; NOT exercised by the CPU tests above.
# ===========================================================================
# SUBTLETY (the 9.1 bug): a corruption to the PREFIX cache only affects answer
# tokens that ATTEND it. If we prefill the whole prompt and read the first-answer
# logit from that (clean) prefill, corruption is invisible. So we prefill all but
# the LAST prompt token, corrupt the cache, then forward the last token THROUGH
# the corrupted cache — now the first answer token (and its entropy) attends the
# corruption. Clean baseline runs the identical path with clean pages injected.

@dataclass(frozen=True)
class CWPoint:
    corruption: str
    clean_acc: float
    corrupt_acc: float
    dacc: float                 # corrupt - clean (want < 0)
    clean_entropy: float        # mean first-answer-token entropy, clean
    corrupt_entropy: float      # mean first-answer-token entropy, corrupt
    dentropy: float             # corrupt - clean (confident-wrong => ~0 or negative)
    logprob_blind: bool         # confidence would NOT flag (dentropy below bar)
    physics_flag_rate: float    # fraction of probes any calibrated fingerprint flags
    novelty_shown: bool         # dacc<=-dacc_min AND logprob_blind AND physics flags


def _decode_under_cache(model, tok, ids, pkv, device, n_new=6):
    """Forward the last prompt token through pkv, then greedy-decode n_new tokens.
    Returns (text, first_token_entropy). pkv already holds prefix = ids[:, :-1]."""
    import torch
    last = ids[:, -1:]
    with torch.no_grad():
        step = model(last, past_key_values=pkv, use_cache=True)
    logits = step.logits[:, -1]
    ent = token_entropy(logits[0].float().cpu().numpy())
    gen = []
    cur = logits
    pkv2 = step.past_key_values
    for _ in range(n_new):
        nxt = cur.argmax(-1, keepdim=True)
        gen.append(nxt)
        with torch.no_grad():
            s = model(nxt, past_key_values=pkv2, use_cache=True)
        pkv2 = s.past_key_values
        cur = s.logits[:, -1]
    text = tok.decode(torch.cat(gen, dim=-1)[0], skip_special_tokens=True)
    return text, ent


def run_phase9_cw(model, tok, device, dtype, probes, *,
                  target_k: int = 4, dacc_min: float = 0.10,
                  entropy_bar: float | None = None, seed: int = 4242):
    """Sweep CORRUPTIONS on real KV. For each: measure clean vs corrupt task
    accuracy, first-answer-token entropy (confidence), and calibrated physics
    flag rate. Emits CONFIDENT_WRONG verdict lines. Nothing tuned to force SHOWN.

    target_k: corrupt the top-k pages by fp_key_norm_mean (the influential ones).
    entropy_bar: if None, set to the clean-run entropy std across probes (a probe
      is 'confident-wrong' when corrupt entropy does NOT exceed clean by > bar)."""
    import torch
    from aepk_paging.harness.phase7_quality import _inject_pages
    from aepk_paging.harness.eval_set import normalized_match
    from aepk_paging.real_model_adapter import dynamiccache_to_pages

    def prefix_pkv(prompt):
        enc = tok(prompt, return_tensors="pt").to(device)
        ids = enc.input_ids
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=True)
        return ids, out.past_key_values

    # clean pass (once): accuracy, entropy, and clean pages for calibration
    clean_correct, clean_ents, calib_pages = 0, [], []
    per_probe = []
    for probe in probes:
        ids, pkv = prefix_pkv(probe["prompt"])
        pages = dynamiccache_to_pages(pkv)
        _inject_pages(pkv, pages, dtype, device)        # no-op clean inject
        text, ent = _decode_under_cache(model, tok, ids, pkv, device)
        ok = normalized_match(text, probe["expected"], probe.get("alternatives"))
        clean_correct += int(ok); clean_ents.append(ent)
        calib_pages.extend(pages)
        # rank influential pages once (clean) for this probe
        order = sorted(range(len(pages)), key=lambda i: -fp_key_norm_mean(pages[i]))
        per_probe.append((probe, order))
    clean_acc = clean_correct / max(1, len(probes))
    calib = calibrate(calib_pages)
    if entropy_bar is None:
        entropy_bar = float(np.std(clean_ents)) if len(clean_ents) > 1 else 0.1

    points = []
    for cname, cfn in CORRUPTIONS.items():
        corr_correct, corr_ents, flag_hits = 0, [], 0
        for (probe, order) in per_probe:
            ids, pkv = prefix_pkv(probe["prompt"])
            pages = dynamiccache_to_pages(pkv)
            tgt = set(order[:target_k])
            new_pages, flagged = [], False
            for i, p in enumerate(pages):
                if i in tgt:
                    c = cfn(p, seed + i)
                    if any_physics_flag(p, c, calib):
                        flagged = True
                    new_pages.append(c)
                else:
                    new_pages.append(p)
            _inject_pages(pkv, new_pages, dtype, device)
            text, ent = _decode_under_cache(model, tok, ids, pkv, device)
            ok = normalized_match(text, probe["expected"], probe.get("alternatives"))
            corr_correct += int(ok); corr_ents.append(ent); flag_hits += int(flagged)
        corrupt_acc = corr_correct / max(1, len(probes))
        dacc = corrupt_acc - clean_acc
        dent = float(np.mean(corr_ents) - np.mean(clean_ents))
        logprob_blind = dent <= entropy_bar          # confidence would NOT fire
        flag_rate = flag_hits / max(1, len(probes))
        shown = (dacc <= -dacc_min) and logprob_blind and (flag_rate >= 0.5)
        points.append(CWPoint(cname, clean_acc, corrupt_acc, dacc,
                              float(np.mean(clean_ents)), float(np.mean(corr_ents)),
                              dent, logprob_blind, flag_rate, shown))
    return points, calib, entropy_bar
