# PR 1 evidence report

Date: 2026-07-28
Status: **acceptance complete; merge-ready**

## Delivered

- CPU-only target-based CMake project;
- checked dtype, shape, element-count, byte-count, and row-major offset APIs;
- non-owning standard-layout tensor views;
- move-only aligned host buffer;
- validated SmolLM2 and Qwen2 model specifications;
- parameter, static-weight, KV-token, and runtime memory estimators;
- explicit checkpoint download ceilings;
- debug, release, and ASan/UBSan presets;
- self-contained offline test harness;
- bounded CPU-only Modal Linux validation with project budget guards.

## Local environment

- Apple M3 Pro / ARM64
- 36 GB unified memory
- Apple Clang 16
- CMake 4.4.0

## Evidence

| Preset | Configure | Build | Tests |
|---|---|---|---|
| `mac-debug` | pass | pass with warnings-as-errors | 16/16 pass |
| `mac-sanitize` | pass | pass with ASan + UBSan | 16/16 pass |
| `mac-release` | pass | pass with warnings-as-errors | 16/16 pass |

### Modal Linux environment

- run: `ap-gJw9O9Sbf4dWvwGq2uGsWf`
- image: `im-cXC8Buu4X2iHfz5gTLo5aY`
- Linux 4.19 gVisor / x86_64 / glibc 2.36
- GCC 12.2.0
- Clang 14.0.6
- CMake 3.25.1
- Ninja 1.11.1

| Configuration | Configure | Build | Tests |
|---|---|---|---|
| GCC debug | pass | pass with warnings-as-errors | 16/16 pass |
| Clang debug | pass | pass with ASan + UBSan | 16/16 pass |

Additional checks:

- CUDA-on configuration fails with the planned message before CMake probes a
  CUDA compiler.
- `git diff --check` passes.
- portable headers and sources include no CUDA, cuBLAS, PyTorch, Python, or
  llama.cpp headers.
- SmolLM2 computed parameters: 134,515,008.
- Qwen2.5 computed parameters: 494,032,768.
- conservative BF16-mapped plus FP32-materialized weight memory:
  0.75 GiB SmolLM2 and 2.76 GiB Qwen.
- a 36 GiB profile with 1 GiB workspace, 4 GiB reserve, and 10,000 × 40 unique
  FP16 KV tokens fits for both locked models.
- 10,000 × 1,024 unique KV tokens is correctly rejected for both.

## Modal budget evidence

- GPU requested: none.
- Validation function wall time: 7.346 seconds.
- Estimated function compute: about $0.00023 at rates published 2026-07-28.
- Per-invocation function ceiling: $0.0184 from the 600-second timeout; image
  build charges are separate.
- Project soft cap: $24, leaving $6 of the $30 monthly budget in reserve.

There are no remaining PR 1 acceptance gates. The next implementation unit is
PR 2, the safe mapped model/config loader.
