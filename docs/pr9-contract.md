# PR 9 contract: constrained output-head acceleration

Status: accepted
Date: 2026-07-31
Base commit: `1a9d65e`

## Outcome

PR 9 turns constrained decoding from a post-logit filter into a native output
projection optimization. For each sequence, the CUDA backend scores only the
embedding rows that are legal in the current grammar state and performs the
deterministic argmax without materializing a full-vocabulary logit tensor.

The same release removes application semantics from the core finite-language
representation. A generic token DFA maps prefix-free token sequences to opaque
choice IDs; market actions and the final autonomy workload are consumers rather
than grammar-layer concepts.

## Correctness invariants

- input hidden states and tied embedding weights are FP16;
- every dot product accumulates in FP32;
- equal scores resolve to the lowest token ID;
- NaNs never win and an all-NaN valid set falls back to its lowest token ID;
- empty, oversized, or out-of-vocabulary device candidate rows emit the
  explicit invalid-token sentinel;
- host-visible model APIs reject empty or out-of-range legal sets before
  mutating context state;
- restricted model decoding exactly matches the full output head whenever the
  full winner is present in the legal set;
- generic token DFAs reject duplicate IDs, duplicate sequences, prefix
  collisions, invalid tokens, and explicit resource-limit violations.

## Performance contract

One source-bound NVIDIA L4 gate compares:

1. cuBLAS projection over all 49,152 SmolLM2 vocabulary rows followed by
   restricted greedy selection;
2. the fused restricted output head over 2, 8, 32, and 128 legal rows.

Batch sizes are 1, 16, and 256. Every measurement records both latencies,
speedup, score counts, avoided logit materialization, GPU identity, and exact
token parity. Measurements describe one locked L4 run; the gate does not assume
that every constrained cardinality must outperform cuBLAS.

## Budget

The existing combined vLLM/native Modal gate remains limited to one L4, one
container, 15 minutes, and $0.239364. Accepted app
`ap-cudAIzy6z8tpmSyazMUWTy` is bound to commit `236c0346` and source bundle
`1f6ec1e5`; it passed 6/6 remote CTests, native and vLLM token parity, and all
12 restricted-head benchmark cells. Conservative project accounting after the
run is $5.331288, below the $24 software cap that protects the user's $6
reserve.
