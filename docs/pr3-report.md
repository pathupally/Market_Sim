# PR 3 evidence report

Date: 2026-07-28
Status: **acceptance complete; merge-ready**

## Delivered

- readable scalar FP32 RMSNorm and `[out, in]` linear projection;
- exact Llama half-rotation RoPE with explicit position bounds;
- grouped-query head mapping and causal attention;
- row-maximum-subtracted FP32 softmax;
- contiguous per-batch K/V append and read;
- attention output projection and residual;
- post-attention RMSNorm and gate/up/SiLU/down SwiGLU residual;
- move-only, 64-byte-aligned, reusable `CpuWorkspace`;
- one complete SmolLM2-compatible decoder-layer function;
- optional trace views after the attention and MLP residuals;
- validation of dtype, shape, host memory, positions, workspace capacity, and
  prohibited overlap before layer mutation;
- deterministic PyTorch exporter and a tiny committed safetensors oracle.

This is deliberately a correctness implementation. The linear entry point is
the future GEMM adapter boundary, but PR 3 uses the scalar fallback everywhere.
It makes no CPU-performance claim and adds no CUDA, tokenizer, sampling, or
full-model generation code.

## Numerical oracle

The reference fixture executes the actual Transformers `LlamaDecoderLayer` and
exports 29 tensors: inputs, weights, every important intermediate, final
residual, and the resulting K/V cache.

| Contract | Value |
|---|---|
| PyTorch | 2.3.1 |
| Transformers | 4.40.1 |
| Architecture repository | `HuggingFaceTB/SmolLM2-135M` |
| Locked architecture revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Shape | batch 2, tokens 3, hidden 8, intermediate 12 |
| Attention | 2 query heads, 1 KV head, head dimension 4 |
| Fixture size | 7,936 bytes |
| Fixture SHA-256 | `0dcbd066353e3c1fbefcad291f3854b23dacc355d8cec4402c7bc9981c34d0d3` |

The exporter assigns every parameter and input from explicit formulas. It then
checks its manually decomposed forward pass against the pinned
`LlamaDecoderLayer`. Safetensors 0.4.3 does not preserve a deterministic
metadata-key order across processes, so the exporter canonicalizes its JSON
header before hashing. Two independent exports now reproduce the same fixture
and manifest hashes byte for byte.

The operator conventions were checked against the pinned
[Transformers 4.40.1 Llama implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.40.1/src/transformers/models/llama/modeling_llama.py):
FP32 RMS accumulation, half-rotation RoPE, repeated K/V heads, max-subtracted
softmax, pre-norm attention, and pre-norm SwiGLU.

Declared absolute error ceilings:

| Operator/output | Absolute tolerance |
|---|---:|
| RMSNorm | `2e-6` |
| RoPE and K/V writes | `2e-6` |
| Attention and projections | `5e-6` |
| Complete decoder layer | `1e-5` |

Both the decomposed C++ pipeline and `smollm2_decoder_layer_f32` pass these
ceilings. The golden checks include input norm, projected and rotated Q/K/V,
attention logits and probabilities, attention heads, output projection, both
residual points, post-attention norm, gate/up/SwiGLU/down intermediates, final
hidden state, and cache contents.

## Behavioral and fault coverage

The native suite covers:

- hand-computable RMSNorm and linear projections;
- stable softmax with masked tails, finite row sums, and all-masked rejection;
- RoPE positions zero and greater than zero, including an assertion that fails
  under adjacent-pair rotation;
- batch sizes one and two, context lengths one and greater than one, and causal
  masking;
- grouped-query mappings with two and four query heads;
- targeted noncontiguous K/V appends whose untouched cache regions retain
  sentinels;
- a zero-projection decoder-layer identity;
- 1,000 repeated layer calls with one stable preallocated workspace;
- rejection of wrong dtype, non-host memory, insufficient workspace, invalid
  position, and illegal trace aliasing without hidden-state mutation;
- the 100,000-input deterministic safetensors parser corpus inherited from
  PR 2.

The source fixture remains read-only during differential tests, which also
checks that the input tensor is unchanged.

## Test evidence

### Local Apple M3 Pro

| Preset | Build | Native tests |
|---|---|---|
| `mac-debug` | Apple Clang warnings-as-errors pass | 35/35 pass |
| `mac-sanitize` | ASan + UBSan pass | 35/35 pass |
| `mac-release` | Apple Clang warnings-as-errors pass | 35/35 pass |

Twelve Python tests pass for golden-file version/provenance/hash contracts,
byte reproducibility, tensor contents, model-fetch safety, and Modal budget
arithmetic.

The ASan/UBSan executable also remapped and rebound the real development
checkpoint:

```text
SmolLM2-135M metadata verified: 30 layers, tied output embedding
```

That locked BF16 file remains outside Git at 269,060,552 bytes. Its SHA-256 is
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
The larger 988 MB Qwen checkpoint was not downloaded.

### Modal Linux CPU

- final run: `ap-rrMWvK2YOCGTDbMJcsycku`;
- Linux 4.19 gVisor / x86_64 / glibc 2.36;
- GCC 12.2.0 warnings-as-errors: pass, 35/35 tests;
- Clang 14.0.6 ASan/UBSan: pass, 35/35 tests;
- function wall time: 24.390 seconds;
- estimated function compute: $0.00075;
- maximum preflight estimate: $0.0184;
- GPU requested: none.

The estimate uses the repository's recorded standard rates for two physical CPU
cores and 2 GiB. Modal's Usage & Billing page remains authoritative for billed
cost, including image-build overhead.

## Gate to PR 4

There are no remaining PR 3 acceptance gates. PR 4 is the first full-model
orchestration step: pretokenized SmolLM2-135M greedy CPU decode, logits and token
parity against pinned PyTorch, and stable memory across repeated decode. It
should reuse the 269 MB mapped checkpoint already present on this machine.

PR 4 must not add tokenizer implementation, CUDA, Qwen weights, custom GEMM
optimization, or broad architecture abstraction.
