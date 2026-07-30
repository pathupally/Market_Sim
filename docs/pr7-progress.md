# PR 7 progress log

Last updated: 2026-07-29
Branch: `codex/pr-007-inference-kernel-core`
Base: `bf190b5`

## Milestone 1 — shared inference contract and vLLM lane

Status: locally complete
Commit: `5de3c8d`

Delivered:

- strict, source-bound CPU/native-CUDA/vLLM inference artifact;
- token-ID-only vLLM 0.25.1 adapter;
- locked SmolLM2 revision, checkpoint size, and checkpoint SHA-256;
- digest-qualified CUDA 12.9 Modal image;
- one-container L4 budget gate;
- local dependency injection tests that do not import vLLM or CUDA.

Evidence:

- focused Python tests: 10 passed;
- repository-root Python discovery: 100 passed, 1 expected skip;
- Apple Clang Debug: build passed, 4/4 CTests passed;
- Apple Clang ASan/UBSan: build passed, 4/4 CTests passed;
- `git diff --check`: passed.

Remote status: not dispatched. No PR 7 Modal cost has been incurred.

## Milestone 2 — FP16 cuBLAS model primitive

Status: remotely validated

Delivered:

- move-only cuBLAS handle with explicit host pointer mode;
- explicit-stream packed-row-major FP16 linear operation;
- FP32 cuBLAS accumulation and FP16 output;
- checked shape arithmetic, exact allocation validation, and alias rejection;
- tiny and SmolLM2-shaped differential tests;
- fused-QKV benchmark at `576 -> 960` for rows 1, 16, and 256;
- one combined Modal gate that builds/tests native CUDA, records RMSNorm and
  cuBLAS benchmarks, then runs vLLM token parity.

Local evidence:

- public CUDA headers compile without CUDA headers on macOS;
- repository-root Python discovery: 101 passed, 1 expected skip;
- Apple Clang Debug: 4/4 CTests passed;
- Apple Clang ASan/UBSan: 4/4 CTests passed;
- `git diff --check`: passed.

Remote evidence:

- accepted commit: `97c82fa9ee090af89af471c407077f888a3fd622`;
- source archive SHA-256:
  `cdd6312083081caa4ad31ea99f515c030b09fece7283636a0138e2b9589ec831`;
- Modal application: `ap-bzkxrYkL94NR58w8JYsO9U`;
- CUDA 12.9.41/GNU 11.4.0 build: 55/55 steps passed;
- CTest: 6/6 passed, including CUDA linear parity and lifecycle probe;
- FP16 fused-QKV cuBLAS:
  - rows 1: 5.784 microseconds, 0.191 TFLOP/s;
  - rows 16: 4.794 microseconds, 3.691 TFLOP/s;
  - rows 256: 8.264 microseconds, 34.260 TFLOP/s;
- vLLM prompt `[0, 1, 2, 3]` generated `[198, 198, 504]`;
- measured three-token vLLM request: 184.202 milliseconds;
- maximum authorized compute cost: $0.239364.

The complete accepted summary is in `docs/pr7-modal-result.json`. The returned
parent-process PyTorch allocator peak was zero because vLLM owns GPU allocations
in a spawned engine worker. This field is marked invalid in the recorded
artifact, and the next commit uses a device-wide NVML peak sampler instead.

## Accepted gate

The source-bound Modal L4 invocation:

1. compile the clean Git archive with CUDA 12.9 and cuBLAS;
2. run the CUDA unit tests, including model-shaped linear parity;
3. record CUDA-event RMSNorm and fused-QKV benchmark JSON;
4. verify the 269 MB checkpoint from the persistent cache;
5. run vLLM on prompt tokens `[0, 1, 2, 3]`;
6. require output tokens `[198, 198, 504]`.

Result: **passed**.
Maximum function compute cost: **$0.239364**.
Maximum duration: **15 L4 minutes**.
Maximum containers: **1**.

No additional PR 7 Modal run is required before starting the custom RoPE and
SwiGLU kernel milestone.
