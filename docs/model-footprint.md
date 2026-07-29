# Local model footprint

Verified: 2026-07-28

## Development machine

- Apple M3 Pro, 12 CPU cores
- 36 GB unified memory
- macOS/ARM64
- approximately 36 GiB free disk at verification time

Only the non-sensitive capacity fields are recorded. Model fit must not depend
on the machine serial number or other device identifiers.

## Locked candidates

| Model | Locked revision | Official BF16 weights | Parameters computed from config | Worst-case mapped BF16 + FP32 weights |
|---|---|---:|---:|---:|
| SmolLM2-135M | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 269 MB | 134,515,008 | 807,090,048 bytes (0.75 GiB) |
| Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | 988 MB | 494,032,768 | 2,964,196,608 bytes (2.76 GiB) |

The worst-case column deliberately counts both the mapped BF16 payload and a
complete FP32 materialization. File-backed pages are not necessarily all
resident simultaneously, so it is conservative.

The repository enforces download ceilings of 300 MiB for SmolLM2 and 1,100 MiB
for Qwen. PR 2's downloader will reject artifacts exceeding those ceilings or
failing the locked hash.

## KV capacity

With FP16 K and V:

| Model | KV bytes/token | 10,000 × 40 unique tokens | 10,000 × 1,024 unique tokens |
|---|---:|---:|---:|
| SmolLM2-135M | 23,040 | 8.58 GiB | 219.7 GiB |
| Qwen2.5-0.5B | 12,288 | 4.58 GiB | 117.2 GiB |

Both models comfortably support local correctness tests and small Metal/CPU
baselines. Ten thousand short unique suffixes can fit within 36 GB under the
conservative estimator; ten thousand long resident contexts cannot. Runtime
metrics therefore keep registered, KV-resident, runnable, and batched sequence
counts separate.

The local correctness envelope is intentionally bounded to prompts of at most
512 tokens, outputs of at most 12 tokens, and small batches. The estimator
reserves 1 GiB for activation/workspace memory and 4 GiB for the OS and
unmodeled runtime overhead. “Fits locally” does not mean Qwen's advertised
32,768-token maximum context is safe for the readable CPU attention oracle.

## Sources

- [SmolLM2 files and 269 MB safetensors size](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2)
- [Qwen2.5 files and 988 MB safetensors size](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
