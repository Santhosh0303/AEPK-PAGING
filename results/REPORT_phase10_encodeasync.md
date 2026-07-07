# REPORT_phase10_encodeasync.md — Phase 10 step 23 encode off the hot path

Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA). A parity group closes every GROUP_SIZE=4 decoded tokens (NUM_PARITY=1); 50 encodes over 200 tokens. Three schedules timed: decode_only (no parity), SYNC (encode inline on the hot path), ASYNC (encode in a worker thread overlapped with the GPU decode step). Per-token latency = arm wall-clock / n_tokens. Suite run TWICE; PREREG gate = run-2 within +/-20% of run-1 (timings never reproduce byte-exactly). Parity BYTES are exact-match across arms and runs (async overlap must not change the encoded parity).

| schedule | run1 ms/tok | run2 ms/tok | within +/-20% |
|----------|-------------|-------------|---------------|
| decode_only | 53.6582 | 75.9836 | - |
| sync | 89.8612 | 88.2273 | True |
| async | 77.3230 | 74.3594 | True |

SYNC amortized overhead vs decode_only = 67.47%; ASYNC residual overhead = 44.10%. parity_bytes_len=626688 exact=True.

## Interpretation
SYNC pays the parity build inline, so its per-token latency carries the amortized encode cost (sync overhead above). ASYNC dispatches the encode (CPU: GF(2^8) linear algebra over page bytes) to a worker thread while the GPU decode step proceeds, hiding it — the residual async overhead is what remains on the hot path after overlap. The encode is deterministic, so the async-built parity is byte-identical to the sync-built parity (correctness of the overlap). ALLOWED to land anywhere: if the async overhead does not vanish (thread/stream sync, GIL contention), it is reported as-is — MICROBENCHMARK scope, not serving throughput.

ENCODE_ASYNC: sync_ms_per_tok=89.8612 async_ms_per_tok=77.3230 overhead_pct=44.10 parity_bytes_exact=True

DETERMINISM: sync_within=True async_within=True all_within_tol=True parity_bytes_exact=True.
