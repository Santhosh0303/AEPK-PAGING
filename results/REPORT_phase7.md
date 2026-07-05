## Real-model validation (Phase 7.4)
Model: Qwen/Qwen2.5-1.5B-Instruct fp16 (CUDA)
Held-out text: 'Artificial intelligence systems must handle memory...'

| Baseline | NLL | Storage bits | Residual MSE | Compute (s) |
|----------|-----|-------------|--------------|-------------|
| B0_no_protection | 4.2500 | 3,211,264 | 0.000000 | 0.077 |
| B1_all_resident | 4.2500 | 3,211,264 | 0.000000 | 0.077 |
| B2_erasure_parity | 4.2500 | 3,325,952 | 0.000000 | 0.118 |
| B3_full_AEPK | 4.8881 | 931,840 | 0.083437 | 0.128 |

Task probe: 'What is the capital of France? Answer in one word:'
  B0 answer: 'Paris. The city known' → CORRECT
  B3 answer: 'Paris France, What.' → CORRECT

Rate-distortion gate: B3 wins 0/50 lambda points
Lambda win range: None
COMPUTE CAVEAT: RS encode/decode CPU time reported above; NOT mixed into RD gate.

**REAL-MODEL GATE VERDICT: FAIL**
