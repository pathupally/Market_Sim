# PR 7 progress log

Last updated: 2026-07-30
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

Remote status: covered by the accepted combined gates below.

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

The initial gate's parent-process PyTorch allocator peak was zero because vLLM
owns GPU allocations in a spawned engine worker. Commit `16ba06c` replaced that
invalid metric with device-wide NVML sampling, which milestone 3 validated.

## Milestone 3 — FP16 RoPE and fused SwiGLU

Status: remotely validated
Commit: `99761ba`

Delivered:

- explicit-stream, in-place FP16 Llama half-rotation RoPE;
- a single angle calculation per token/dimension pair shared across all 9
  query and 3 KV heads;
- exact allocation, checked arithmetic, alias, theta, and launch validation;
- explicit-stream fused FP16 `SiLU(gate) * up` with supported in-place gate
  reuse;
- differential tests against the independent FP32 CPU oracle at tiny and exact
  SmolLM2 GQA shapes;
- a strict model-shaped CUDA-event benchmark for RoPE and SwiGLU.

Local evidence:

- repository-root Python discovery: 102 passed, 1 expected skip;
- Apple Clang Debug: 4/4 CTests passed;
- Apple Clang ASan/UBSan: 4/4 CTests passed;
- `git diff --check`: passed.

Remote evidence:

- accepted commit: `99761ba937d24106afa01a03df86222c00bc75b8`;
- source archive SHA-256:
  `d4ab650eab8419a0f2979763dfff327bd3a5af64507722f45153388016ce77b7`;
- Modal application: `ap-bkWKBEliPSSqc6GqpM6cZ9`;
- CUDA 12.9.41/GNU 11.4.0 build: 59/59 steps passed;
- CTest: 6/6 passed, including CUDA differential tests;
- FP16 RoPE:
  - rows 1: 4.102 microseconds;
  - rows 16: 4.194 microseconds;
  - rows 256: 4.294 microseconds, 170.772 logical GiB/s;
- fused FP16 SwiGLU:
  - rows 1: 2.008 microseconds;
  - rows 16: 2.053 microseconds;
  - rows 256: 4.409 microseconds, 498.356 logical GiB/s;
- vLLM prompt `[0, 1, 2, 3]` again generated `[198, 198, 504]`;
- measured three-token vLLM request: 144.992 milliseconds;
- device-wide NVML peak: 8,943,173,632 bytes;
- maximum authorized compute cost: $0.239364.

The complete latest accepted summary is in `docs/pr7-modal-result.json`.

## Milestone 4 — contiguous FP16 KV write and greedy selection

Status: remotely validated
Commits: `16f9b0f`, `4042428`

Delivered:

- explicit-stream FP16 writes from packed key/value tensors into contiguous
  `[batch, maximum_context, 3, 64]` caches;
- bounds-guarded device positions, exact allocation checks, alias rejection,
  and checked shape arithmetic;
- deterministic block-reduction greedy selection over FP16 logits;
- lowest-token-ID tie behavior across reduction lanes;
- NaN-loses semantics and a defined all-NaN fallback;
- tiny, batched, sparse-position, untouched-cache, and full 49,152-vocabulary
  differential tests.

Local evidence:

- repository-root Python discovery: 102 passed, 1 expected skip;
- Apple Clang Debug: 4/4 CTests passed;
- Apple Clang ASan/UBSan: 4/4 CTests passed;
- `git diff --check`: passed.

Remote process:

- first source-bound attempt `ap-4ftVs1KlLbAPHyKWheHUEa` failed during NVCC
  compilation because `std::numeric_limits` members are host-only under the
  pinned CUDA frontend;
- commit `4042428` replaced those expressions with explicit device constants;
- accepted commit: `4042428c0edc10a47e8a9480b8c236985221e263`;
- source archive SHA-256:
  `f0d308cdd1503baec2e77bf7310553a4ced51d2798f30df0c8ec81efde4097fa`;
- accepted Modal application: `ap-rBeYAoSg3HuV35yYCZ50Xf`;
- CUDA 12.9.41/GNU 11.4.0 build: 61/61 steps passed;
- CTest: 6/6 passed, including KV-cache mutation and full-vocabulary greedy
  tests;
- vLLM prompt `[0, 1, 2, 3]` again generated `[198, 198, 504]`;
- measured three-token vLLM request: 141.785 milliseconds;
- device-wide NVML peak: 8,943,173,632 bytes;
- each attempt was bounded to $0.239364; the failed reservation remains counted
  conservatively in the project budget tracker.

## Milestone 5 — composed native FP16 decoder layer

Status: remotely validated
Commits: `883f26d`, `0918106`

Delivered:

- causal FP16 grouped-query attention over contiguous KV storage;
- FP16 RMSNorm, residual add, and embedding-gather kernels;
- one explicit-stream SmolLM2 decoder-layer compositor;
- fixed caller-owned intermediate buffers reused across attention and MLP
  phases;
- cuBLAS Q/K/V, output, gate/up, and down projections;
- custom RoPE, KV write, causal GQA attention, SwiGLU, and residual kernels in
  one vertical execution path;
- a two-token differential test against the independent rounded-FP32 decoder
  oracle, including resulting K/V state.

Local evidence:

- repository-root Python discovery: 102 passed, 1 expected skip;
- Apple Clang Debug: 4/4 CTests passed;
- Apple Clang ASan/UBSan: 4/4 CTests passed;
- portable public CUDA headers compile without CUDA headers on macOS;
- `git diff --check`: passed.

Remote process:

- first source-bound attempt `ap-vmg9q7hcXyPyDkOWLoMQVU` failed during
  compilation on an incorrect `CublasHandle` validity accessor;
- commit `0918106` changed the compositor to the established
  `cublas.handle().valid()` interface;
- accepted commit: `0918106604958421a40468b04393bfd6cf397b11`;
- source archive SHA-256:
  `0ba43f3cf258e5e3e6bc29af94a98b526f566682efe3b78fb7b7ab881f2eb7b7`;
- accepted Modal application: `ap-UFxPIfhOmUEN0Q7VUQ7Wol`;
- CUDA 12.9.41/GNU 11.4.0 build: 66/66 steps passed;
- CTest: 6/6 passed, including native decoder-layer FP16/FP32 differential
  parity;
- vLLM prompt `[0, 1, 2, 3]` again generated `[198, 198, 504]`;
- measured three-token vLLM request: 181.620 milliseconds;
- device-wide NVML peak: 8,943,173,632 bytes;
- each attempt was bounded to $0.239364; the failed reservation remains counted
  conservatively in the project budget tracker.

## Milestone 6 — full native SmolLM2 prefill and decode

Status: remotely validated; PR 7 implementation complete
Commits: `24c08d0`, `182e504`

Delivered:

- locked BF16 safetensors loading through the existing safe model binder;
- deterministic BF16-to-FP16 device-weight materialization;
- tied embedding/output-head storage;
- all 30 SmolLM2 decoder layers on the native CUDA compositor;
- per-layer contiguous FP16 K/V caches;
- separate fixed prefill and decode workspaces;
- final FP16 RMSNorm, tied LM-head projection, and deterministic greedy select;
- strict native inference evidence embedded in the combined gate;
- exact three-step native token parity with the CPU/PyTorch fixture and vLLM.

Local evidence:

- repository-root Python discovery: 103 passed, 1 expected skip;
- Apple Clang Debug: 4/4 CTests passed;
- Apple Clang ASan/UBSan: 4/4 CTests passed;
- portable public CUDA headers compile without CUDA headers on macOS;
- `git diff --check`: passed.

Remote process:

- first source-bound attempt `ap-SYiL3qoF194M3u18mFqJyz` compiled all native
  sources but failed at the final static link because `marketforge_cuda` had
  not yet declared its direct `MarketForge::core` dependency;
- commit `182e504` records that target dependency;
- accepted commit: `182e5047748aaf2744e24c4d34d281f15fb60e70`;
- source archive SHA-256:
  `1ef5a05a22fb1501b09cd8250975d70e31bf8b1d506135a1111ff1138e8fece4`;
- accepted Modal application: `ap-j6LvUOCbJJHsMlAtYQ5ZeT`;
- CUDA 12.9.41/GNU 11.4.0 build: 69/69 steps passed;
- CTest: 6/6 passed;
- native FP16 prompt `[0, 1, 2, 3]` plus two decode calls generated
  `[198, 198, 504]`;
- vLLM independently generated `[198, 198, 504]`;
- native model load: 438.170 milliseconds;
- native three-step inference: 286.153 milliseconds;
- native owned device memory: 269,372,588 bytes;
- vLLM measured three-token request: 139.291 milliseconds;
- vLLM device-wide NVML peak: 8,943,173,632 bytes;
- each attempt was bounded to $0.239364; the failed reservation remains counted
  conservatively in the project budget tracker.

## Accepted gate

The source-bound Modal L4 invocation:

1. compile the clean Git archive with CUDA 12.9 and cuBLAS;
2. run CUDA unit tests, including model-shaped primitives and composed
   decoder-layer differential parity;
3. record CUDA-event RMSNorm, fused-QKV, RoPE, and SwiGLU benchmark JSON;
4. verify the 269 MB checkpoint from the persistent cache;
5. run the full 30-layer native FP16 prefill/decode path;
6. run vLLM on prompt tokens `[0, 1, 2, 3]`;
7. require both backends to produce `[198, 198, 504]`.

Result: **passed**.
Maximum function compute cost: **$0.239364**.
Maximum duration: **15 L4 minutes**.
Maximum containers: **1**.

The latest accepted gate is source-bound to the complete PR 7 implementation.
