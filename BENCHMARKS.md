# Benchmarks

The accepted performance artifact was produced on one NVIDIA L4 and is bound
to the code that ran. This file reports the measurements without extrapolating
to other models, GPUs, or serving configurations.

## Artifact identity

| Field | Value |
|---|---|
| Commit | `236c0346b724332c821cce46f79f8853d3e3a447` |
| Source archive SHA-256 | `1f6ec1e5cd42ab84a8f907ff2eed77ed60dc5f81feffd56a5b3eb0b3295f8da2` |
| GPU | NVIDIA L4, compute capability 8.9 |
| Model | SmolLM2-135M, locked revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Vocabulary / hidden size | 49,152 / 576 |
| vLLM | 0.25.1 |
| Remote CTests | 6 passed, 0 failed |

The committed benchmark summary retains the source identity, hardware, parity
result, and complete restricted-head matrix needed to interpret the claims.
Internal run logs and project notes are intentionally excluded from the public
repository.

## Restricted CUDA output head

The baseline performs a cuBLAS projection over all 49,152 tied-embedding rows,
then restricts greedy selection. The custom path scores only the legal rows and
reduces them directly. Both paths use FP32 dot-product accumulation and must
select the same token.

| Batch rows | Legal rows | Full projection (us) | Restricted (us) | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 2 | 238.672 | 12.872 | 18.542x |
| 1 | 8 | 238.643 | 14.864 | 16.055x |
| 1 | 32 | 238.533 | 26.196 | 9.106x |
| 1 | 128 | 238.635 | 44.702 | 5.338x |
| 16 | 2 | 249.651 | 13.155 | 18.978x |
| 16 | 8 | 249.631 | 15.012 | 16.629x |
| 16 | 32 | 249.334 | 26.423 | 9.436x |
| 16 | 128 | 248.832 | 45.609 | 5.456x |
| 256 | 2 | 365.312 | 17.633 | 20.717x |
| 256 | 8 | 370.473 | 28.938 | 12.802x |
| 256 | 32 | 375.009 | 91.177 | 4.113x |
| 256 | 128 | 374.170 | 294.144 | 1.272x |

All 12 cells had exact token parity. The result also shows the limit of the
optimization: its advantage narrows as the legal set grows because the custom
kernel gives up cuBLAS's dense-matrix efficiency.

## Full-model conformance

The accepted native record loaded 269,373,104 device bytes and generated
`[198, 198, 504]` from prompt IDs `[0, 1, 2, 3]`. vLLM produced the same greedy
sequence from the same locked checkpoint.

The native record reports 0.496 seconds for model load and 0.269 seconds for
the conformance inference. That inference time is retained for reproducibility,
not presented as a throughput comparison: it is a three-token correctness
case with a narrow runtime and no production serving loop.

## vLLM ablations

The vLLM lane holds the model, token-ID prompt family, greedy settings, GPU,
and result schema constant while changing execution or cache mode.

| Comparison | Speedup |
|---|---:|
| CUDA Graph vs. eager, one request | 3.378x |
| CUDA Graph vs. eager, batch of 16 | 2.089x |
| Eager batch request throughput vs. eager single | 28.365x |
| CUDA Graph batch request throughput vs. graph single | 17.539x |
| Warm 128-token shared prefix vs. cold replay | 2.170x |

The native and vLLM numbers answer different questions. The restricted-head
matrix isolates one native kernel against a full-vocabulary native baseline.
The vLLM matrix measures execution-mode and prefix-cache effects inside vLLM.
It does not compare end-to-end native throughput against vLLM.

## Measurement rules

- Timed CUDA work uses CUDA events after warm-up.
- Every result records exact input and output token IDs.
- Restricted and full output paths must have token parity before speedup is
  accepted.
- Source identity comes from a clean Git archive, not an uncommitted checkout.
- Model revision, artifact size, and SHA-256 are checked before loading.
- GPU class, timeout, and maximum compute cost are fixed before dispatch.
- Failed or incomplete gates do not become accepted benchmark evidence.

The Modal reproduction entry points are documented in
[`tools/modal/README.md`](tools/modal/README.md). A new run may produce
different timings because cloud hardware state and software versions outside
the locked container can vary. It should preserve the correctness invariants
and emit a new source-bound artifact rather than overwrite this summary.
