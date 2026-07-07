"""Phase 10 step (18) — heal-cost MICROBENCHMARKS (Path B minimal).

Times the primitive operations of the live-heal path on real Qwen2.5-1.5B KV pages, in the
live-heal config (GROUP_SIZE=4 sibling layer-pages, NUM_PARITY=1 -> single-erasure recovery):
  (a) ENCODE     — parity generation per page group (encode_rs_erasure_group)
  (b) HEAL       — recover ONE erased page from parity (recover_rs_erasure)
  (c) RECOMPUTE  — re-prefill the same prompt prefix (the HONEST alternative to healing)
  (d) FINGERPRINT— physics fingerprint compute per page (fp_key_norm_mean)
  (e) OVERHEAD   — parity storage bytes, MEASURED vs ANALYTIC (must match exactly)

Protocol (PREREG_phase10_healcost.md): N=100 timed reps per quantity, report MEDIAN + IQR; the
whole timing suite runs TWICE; run-2 medians must land within +/-20% of run-1 (timings never
reproduce byte-exactly — this is the pre-registered determinism gate; the RUNTIME ECONOMY timing
exemption). Row DATA that CAN be exact — the parity storage bytes — is asserted exact-match.

Scope caveat (labeled MICROBENCHMARKS throughout): these are primitive-operation latencies, NOT
serving throughput. There is no request stream, no batching, no vLLM/paged-attention integration;
end-to-end serving cost stays future work. HEAL vs RECOMPUTE is the one comparison that matters
for the erasure-conversion claim (heal should beat a full prefix recompute).

Honesty spine S9: zero edits to Phase 2-5 source. Reuses coding.encode_rs_erasure_group /
recover_rs_erasure and phase9_cw.fp_key_norm_mean + the live-heal GROUP_SIZE/NUM_PARITY. The
median/IQR/overhead math is deterministic and CPU-testable with no model. HEAL_COST line is a
runtime f-string.
"""

from __future__ import annotations

import time

import numpy as np

from aepk_paging.harness.phase10_liveheal import GROUP_SIZE, NUM_PARITY
from aepk_paging.coding import encode_rs_erasure_group, recover_rs_erasure, _page_row_bytes

N_REPS = 100                # timed reps per quantity
MEDIAN_TOL = 0.20           # run-2 median within +/-20% of run-1 (PREREG timing gate)


# ---- deterministic, CPU-testable math (no model) ----------------------------

def median_iqr(samples) -> tuple[float, float]:
    """(median, IQR) of a sample list. IQR = P75 - P25. Empty -> (nan, nan)."""
    a = np.asarray(samples, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(a))
    iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
    return med, iqr


def parity_overhead_bytes(group_pages, num_parity: int = NUM_PARITY) -> tuple[int, int]:
    """(measured, analytic) parity storage bytes for the group. Measured = the encoded
    parity_bytes buffer size; analytic = num_parity * per-page row length (K bytes + V bytes).
    They MUST be equal (exact-match gate)."""
    group = encode_rs_erasure_group(list(group_pages), num_parity)
    measured = int(group.parity_bytes.nbytes)
    row, _ = _page_row_bytes(group_pages[0])
    analytic = int(num_parity * row.size)
    return measured, analytic


def overhead_pct(group_pages, num_parity: int = NUM_PARITY) -> float:
    """Parity bytes as a percentage of the protected DATA bytes (num_parity/group_size for
    equal-size sibling pages -> 25.0 at the live-heal 4/1 config)."""
    measured, _ = parity_overhead_bytes(group_pages, num_parity)
    data_bytes = sum(_page_row_bytes(p)[0].size for p in group_pages)
    return 100.0 * measured / data_bytes


def within_tolerance(m1: float, m2: float, tol: float = MEDIAN_TOL) -> bool:
    """True if m2 is within +/-tol of m1 (relative). m1<=0 falls back to abs equality."""
    if not (m1 == m1 and m2 == m2):        # nan guard
        return False
    if m1 <= 0:
        return m2 == m1
    return abs(m2 - m1) <= tol * m1


def _time_reps(fn, reps: int = N_REPS) -> list[float]:
    """Time `fn()` `reps` times; return per-rep milliseconds. One warm-up call excluded."""
    fn()                                   # warm-up (JIT/caches), not counted
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


# ---- GPU/model-backed timing run (real Qwen pages) --------------------------

def run_healcost(model, tok, device, dtype, *, probe, reps: int = N_REPS) -> dict:
    """Time ENCODE / HEAL / RECOMPUTE / FINGERPRINT on real KV pages for one probe, plus the
    exact parity storage overhead. Returns a dict of (median, iqr) ms per quantity + overhead."""
    import torch
    from aepk_paging.harness.phase9_cw import fp_key_norm_mean
    from aepk_paging.real_model_adapter import dynamiccache_to_pages

    enc = tok(probe["prompt"], return_tensors="pt").to(device)
    ids = enc.input_ids

    def prefill():
        with torch.no_grad():
            model(ids[:, :-1], use_cache=True)

    with torch.no_grad():
        out = model(ids[:, :-1], use_cache=True)
    pages = dynamiccache_to_pages(out.past_key_values)
    # live-heal group: top-GROUP_SIZE pages by fingerprint influence (matches liveheal)
    order = sorted(range(len(pages)), key=lambda i: -fp_key_norm_mean(pages[i]))
    group_pages = [pages[i] for i in order[:GROUP_SIZE]]
    group = encode_rs_erasure_group(group_pages, NUM_PARITY)
    tgt_id = group_pages[0].page_id

    def encode():
        encode_rs_erasure_group(group_pages, NUM_PARITY)

    def heal():
        recover_rs_erasure(group, [tgt_id])

    def fingerprint():
        fp_key_norm_mean(group_pages[0])

    def recompute():
        prefill()
        if device != "cpu":
            torch.cuda.synchronize()

    res: dict = {}
    for name, fn in (("encode", encode), ("heal", heal),
                     ("recompute", recompute), ("fingerprint", fingerprint)):
        res[name] = median_iqr(_time_reps(fn, reps))
    measured, analytic = parity_overhead_bytes(group_pages, NUM_PARITY)
    res["parity_bytes_measured"] = measured
    res["parity_bytes_analytic"] = analytic
    res["parity_overhead_pct"] = overhead_pct(group_pages, NUM_PARITY)
    res["group_page_shape"] = tuple(np.asarray(group_pages[0].K).shape)
    return res


def write_healcost_report(run1: dict, run2: dict, path="results/REPORT_phase10_healcost.md"):
    """Write the microbenchmark report from the two timing runs. HEAL_COST line is runtime.
    Returns (heal_ms, recompute_ms, ratio, overhead_pct, all_within_tol, bytes_exact)."""
    import os

    quantities = ("encode", "heal", "recompute", "fingerprint")
    within = {q: within_tolerance(run1[q][0], run2[q][0]) for q in quantities}
    all_within = all(within.values())
    bytes_exact = (run1["parity_bytes_measured"] == run1["parity_bytes_analytic"]
                   == run2["parity_bytes_measured"] == run2["parity_bytes_analytic"])

    heal_ms = run1["heal"][0]
    recompute_ms = run1["recompute"][0]
    ratio = recompute_ms / heal_ms if heal_ms > 0 else float("inf")
    ov = run1["parity_overhead_pct"]

    L = [
        "# REPORT_phase10_healcost.md — Phase 10 step (18) heal-cost MICROBENCHMARKS (Path B minimal)",
        "",
        "Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). Live-heal config: "
        f"GROUP_SIZE={GROUP_SIZE} sibling layer-pages, NUM_PARITY={NUM_PARITY} "
        f"(single-erasure recovery). Group page K-shape={run1['group_page_shape']}. "
        f"N={N_REPS} timed reps per quantity; MEDIAN + IQR (ms). Suite run TWICE; PREREG gate = "
        f"run-2 median within +/-{int(MEDIAN_TOL*100)}% of run-1 (timings never reproduce "
        "byte-exactly). Parity storage BYTES are exact-match (measured vs analytic).",
        "",
        "**MICROBENCHMARKS — primitive-operation latencies, NOT serving throughput.** No request "
        "stream, no batching, no vLLM/paged-attention integration; end-to-end serving cost is "
        "future work. The load-bearing comparison is HEAL vs RECOMPUTE (a full prefix re-prefill).",
        "",
        "| quantity | run1 median (ms) | run1 IQR | run2 median (ms) | within +/-20% |",
        "|----------|------------------|----------|------------------|---------------|",
    ]
    for q in quantities:
        L.append(f"| {q} | {run1[q][0]:.4f} | {run1[q][1]:.4f} | {run2[q][0]:.4f} | {within[q]} |")
    L += [
        "",
        f"Parity storage: measured={run1['parity_bytes_measured']} bytes "
        f"analytic={run1['parity_bytes_analytic']} bytes exact_match={bytes_exact} "
        f"(overhead={ov:.2f}% of protected data = NUM_PARITY/GROUP_SIZE).",
        "",
        "## Interpretation",
        "HEAL restores one page from parity by inverting a k x k Cauchy submatrix over GF(2^8) — "
        "a fixed-size linear-algebra op independent of prompt length. RECOMPUTE re-prefills the "
        "whole prompt prefix through the model — it grows with context. On this single-page, "
        "short-prompt microbenchmark the ratio below is the per-operation speedup of erasure "
        "healing over recompute; it is NOT an end-to-end serving number. FINGERPRINT is the "
        "per-page detection cost paid at encode time; ENCODE is the one-time parity build per group.",
        "",
        f"HEAL_COST: heal_ms={heal_ms:.4f} recompute_ms={recompute_ms:.4f} "
        f"ratio={ratio:.2f} parity_overhead_pct={ov:.2f}",
        "",
        f"DETERMINISM: all_medians_within_tol={all_within} (per-quantity: {within}); "
        f"parity_bytes_exact={bytes_exact}.",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return heal_ms, recompute_ms, ratio, ov, all_within, bytes_exact


# ============================================================================
# Step 23 — encode OFF the hot path (async/overlapped parity build). New section;
# nothing above is edited. A parity group closes every GROUP_SIZE decoded tokens;
# the SYNC arm encodes inline, the ASYNC arm runs encode in a worker thread while
# decode (GPU) continues, so the CPU parity build overlaps the GPU decode step.
# ============================================================================

ASYNC_TOKENS = 200          # decoded tokens per arm (>= 200)


def groups_closed(n_tokens: int, group_size: int = GROUP_SIZE) -> int:
    """Number of parity encodes triggered by decoding n_tokens (one per closed group)."""
    return int(n_tokens) // int(group_size)


def amortized_overhead_pct(baseline_ms: float, arm_ms: float) -> float:
    """Residual per-token overhead of `arm` vs a pure-decode `baseline`, as a percentage. Negative
    means the arm is faster than baseline (measurement jitter or fully hidden work)."""
    if not (baseline_ms == baseline_ms and arm_ms == arm_ms):    # nan guard
        return float("nan")
    if baseline_ms <= 0:
        return float("nan")
    return 100.0 * (arm_ms - baseline_ms) / baseline_ms


def run_encodeasync(model, tok, device, dtype, *, probe, n_tokens: int = ASYNC_TOKENS,
                    group_size: int = GROUP_SIZE) -> dict:
    """Measure per-token decode latency under three schedules on real Qwen pages:
    decode_only (no parity), SYNC (encode inline every group_size tokens), ASYNC (encode in a
    worker thread overlapped with decode). Also returns the sync/async parity bytes for the
    byte-identical correctness check. Deterministic parity; timings are wall-clock."""
    import threading

    import torch
    from aepk_paging.harness.phase9_cw import fp_key_norm_mean
    from aepk_paging.real_model_adapter import dynamiccache_to_pages

    enc = tok(probe["prompt"], return_tensors="pt").to(device)
    ids = enc.input_ids

    # fixed representative parity group (top-influence sibling pages), from the prompt prefill
    with torch.no_grad():
        out0 = model(ids[:, :-1], use_cache=True)
    pages = dynamiccache_to_pages(out0.past_key_values)
    order = sorted(range(len(pages)), key=lambda i: -fp_key_norm_mean(pages[i]))
    group_pages = [pages[i] for i in order[:group_size]]

    def encode_parity():
        return encode_rs_erasure_group(group_pages, NUM_PARITY).parity_bytes

    def fresh_decode_state():
        with torch.no_grad():
            o = model(ids, use_cache=True)
        if device != "cpu":
            torch.cuda.synchronize()
        return o.past_key_values, o.logits[:, -1:].argmax(-1)

    def run_arm(mode):
        pkv, cur = fresh_decode_state()
        threads = []
        last_parity = {}

        def step():
            nonlocal pkv, cur
            with torch.no_grad():
                o = model(cur, past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            cur = o.logits[:, -1:].argmax(-1)
            if device != "cpu":
                torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(n_tokens):
            step()
            if mode != "decode_only" and (i + 1) % group_size == 0:
                if mode == "async":
                    th = threading.Thread(
                        target=lambda: last_parity.__setitem__("p", encode_parity()))
                    th.start()
                    threads.append(th)
                else:                                    # sync: inline on the hot path
                    last_parity["p"] = encode_parity()
        for th in threads:
            th.join()
        total_ms = (time.perf_counter() - t0) * 1e3
        return total_ms / n_tokens, last_parity.get("p")

    decode_only_ms, _ = run_arm("decode_only")
    sync_ms, sync_parity = run_arm("sync")
    async_ms, async_parity = run_arm("async")

    parity_bytes_exact = bool(
        sync_parity is not None and async_parity is not None
        and np.array_equal(sync_parity, async_parity)
        and np.array_equal(sync_parity, encode_parity()))

    return {
        "n_tokens": int(n_tokens), "group_size": int(group_size),
        "groups_closed": groups_closed(n_tokens, group_size),
        "decode_only_ms_per_tok": float(decode_only_ms),
        "sync_ms_per_tok": float(sync_ms),
        "async_ms_per_tok": float(async_ms),
        "parity_bytes_len": int(sync_parity.size) if sync_parity is not None else 0,
        "parity_bytes_exact": parity_bytes_exact,
    }


def write_encodeasync_report(run1: dict, run2: dict,
                             path="results/REPORT_phase10_encodeasync.md"):
    """ENCODE_ASYNC report from two timing runs. overhead_pct = residual async per-token overhead
    vs pure decode. Latency gate = run-2 within +/-20% of run-1; parity bytes exact across arms
    and runs. Runtime f-string verdict."""
    import os

    sync_within = within_tolerance(run1["sync_ms_per_tok"], run2["sync_ms_per_tok"])
    async_within = within_tolerance(run1["async_ms_per_tok"], run2["async_ms_per_tok"])
    all_within = sync_within and async_within
    bytes_exact = bool(run1["parity_bytes_exact"] and run2["parity_bytes_exact"])
    overhead = amortized_overhead_pct(run1["decode_only_ms_per_tok"], run1["async_ms_per_tok"])
    sync_overhead = amortized_overhead_pct(run1["decode_only_ms_per_tok"], run1["sync_ms_per_tok"])

    L = [
        "# REPORT_phase10_encodeasync.md — Phase 10 step 23 encode off the hot path",
        "",
        "Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). A parity group closes every "
        f"GROUP_SIZE={GROUP_SIZE} decoded tokens (NUM_PARITY={NUM_PARITY}); "
        f"{run1['groups_closed']} encodes over {run1['n_tokens']} tokens. Three schedules timed: "
        "decode_only (no parity), SYNC (encode inline on the hot path), ASYNC (encode in a worker "
        "thread overlapped with the GPU decode step). Per-token latency = arm wall-clock / "
        "n_tokens. Suite run TWICE; PREREG gate = run-2 within +/-20% of run-1 (timings never "
        "reproduce byte-exactly). Parity BYTES are exact-match across arms and runs (async overlap "
        "must not change the encoded parity).",
        "",
        "| schedule | run1 ms/tok | run2 ms/tok | within +/-20% |",
        "|----------|-------------|-------------|---------------|",
        f"| decode_only | {run1['decode_only_ms_per_tok']:.4f} | "
        f"{run2['decode_only_ms_per_tok']:.4f} | - |",
        f"| sync | {run1['sync_ms_per_tok']:.4f} | {run2['sync_ms_per_tok']:.4f} | {sync_within} |",
        f"| async | {run1['async_ms_per_tok']:.4f} | {run2['async_ms_per_tok']:.4f} | "
        f"{async_within} |",
        "",
        f"SYNC amortized overhead vs decode_only = {sync_overhead:.2f}%; ASYNC residual overhead = "
        f"{overhead:.2f}%. parity_bytes_len={run1['parity_bytes_len']} exact={bytes_exact}.",
        "",
        "## Interpretation",
        "SYNC pays the parity build inline, so its per-token latency carries the amortized encode "
        "cost (sync overhead above). ASYNC dispatches the encode (CPU: GF(2^8) linear algebra over "
        "page bytes) to a worker thread while the GPU decode step proceeds, hiding it — the "
        "residual async overhead is what remains on the hot path after overlap. The encode is "
        "deterministic, so the async-built parity is byte-identical to the sync-built parity "
        "(correctness of the overlap). ALLOWED to land anywhere: if the async overhead does not "
        "vanish (thread/stream sync, GIL contention), it is reported as-is — MICROBENCHMARK scope, "
        "not serving throughput.",
        "",
        f"ENCODE_ASYNC: sync_ms_per_tok={run1['sync_ms_per_tok']:.4f} "
        f"async_ms_per_tok={run1['async_ms_per_tok']:.4f} overhead_pct={overhead:.2f} "
        f"parity_bytes_exact={bytes_exact}",
        "",
        f"DETERMINISM: sync_within={sync_within} async_within={async_within} "
        f"all_within_tol={all_within} parity_bytes_exact={bytes_exact}.",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return run1["sync_ms_per_tok"], run1["async_ms_per_tok"], overhead, bytes_exact, all_within


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from aepk_paging.harness.eval_set_easy import get_combined_probes

    MID = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map="cuda").eval()
    probe = get_combined_probes()[0]
    r1 = run_healcost(model, tok, "cuda", torch.float16, probe=probe)
    r2 = run_healcost(model, tok, "cuda", torch.float16, probe=probe)
    heal_ms, rec_ms, ratio, ov, ok, bx = write_healcost_report(r1, r2)
    print(f"HEAL_COST: heal_ms={heal_ms:.4f} recompute_ms={rec_ms:.4f} "
          f"ratio={ratio:.2f} parity_overhead_pct={ov:.2f}")
    print(f"DETERMINISM: all_within_tol={ok} parity_bytes_exact={bx}")
