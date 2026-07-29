# PR 2 evidence report

Date: 2026-07-28
Status: **acceptance complete; merge-ready**

## Delivered

- move-only, read-only `mmap` ownership for macOS and Linux;
- dependency-free JSON parser with UTF-8, duplicate-key, grammar, and depth
  validation;
- safetensors metadata parser with a 16 MiB default header ceiling;
- exact dtype, rank, shape, byte-count, offset, coverage, and file-bound checks;
- borrowed tensor lookup whose lifetime is owned by `SafeTensorFile`;
- SmolLM2 and Qwen config parsing into the existing `ModelSpec`;
- exact SmolLM2 required tensor table and optional output-head handling;
- tied output/embedding alias behavior;
- immutable model lock with repository commits, sizes, SHA-256 hashes, licenses,
  and architecture contracts;
- standard-library-only fetch/verify/purge tool with an explicit external cache;
- generated valid and malformed fixtures plus a libFuzzer entry point;
- a standalone full-checkpoint metadata inspection executable.

No inference operator, tokenizer, CUDA source, or model-weight copy was added.

## Small-model artifact policy

| Artifact | Locked size | Local state | Default fetch policy |
|---|---:|---|---|
| SmolLM2-135M config | 704 B | verified | allowed |
| SmolLM2-135M BF16 weights | 269,060,552 B | verified | allowed |
| Qwen2.5-0.5B config | 659 B | verified | allowed |
| Qwen2.5-0.5B BF16 weights | 988,097,824 B | not downloaded | blocked |

The Smol checkpoint lives outside the repository under the user's
`Library/Caches/marketforge` directory. The repository contains no weight file
larger than the small JSON fixtures.

Verified hashes:

- Smol config:
  `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`
- Smol weights:
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- Qwen config:
  `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`

The ASan/UBSan inspection of the real Smol file reported:

```text
SmolLM2-135M metadata verified: 30 layers, tied output embedding
```

The check maps the file read-only and parses its metadata. It does not
materialize a second 269 MB weight copy.

## Test evidence

### Local Apple M3 Pro

| Preset | Build | Tests |
|---|---|---|
| `mac-debug` | warnings-as-errors pass | 25/25 pass |
| `mac-sanitize` | ASan + UBSan pass | 25/25 pass |
| `mac-release` | warnings-as-errors pass | 25/25 pass |

Nine Python tests also pass for downloader corruption handling, moving-revision
rejection, manifest ceilings, Qwen weight blocking, duplicate manifest keys,
and Modal budget arithmetic.

The safetensors suite covers:

- valid F32, F16, BF16, I32, rank-0, and empty tensors;
- every truncation boundary of the generated valid fixture;
- invalid UTF-8 and JSON;
- duplicate keys and unsupported dtypes;
- excessive rank, negative extents, and extent multiplication overflow;
- reversed offsets, overlap, gaps, byte mismatch, and trailing data;
- header resource ceilings;
- missing, wrong-shaped, and unexpected Smol tensors;
- tied and optional explicit output heads;
- 100,000 deterministic parser inputs under sanitizers.

### Modal Linux

- final run: `ap-NtebtCCMnXv8VfAYN16in5`
- Linux 4.19 gVisor / x86_64 / glibc 2.36
- GCC 12.2.0 warnings-as-errors: pass, 25/25 tests
- Clang 14.0.6 ASan/UBSan: pass, 25/25 tests
- function wall time: 16.74 seconds
- estimated function compute: $0.00051
- GPU requested: none

An earlier Linux run exposed a recursive-value incompatibility specific to
Clang 14 with libstdc++. JSON object members now use explicit owned nodes; the
final GCC, Clang, and Apple matrices all pass.

## Remaining boundaries

- Qwen weight binding is intentionally deferred.
- Tokenization and numerical operators begin in later PRs.
- The code-side soft cap supplements, but does not replace, the enforced $30
  Modal Workspace budget that should be configured in Modal.
- Actual Modal billing, including image builds, remains authoritative in the
  Usage & Billing dashboard.

There are no remaining PR 2 acceptance gates. PR 3 begins with readable FP32
transformer operators and tiny deterministic parity fixtures.
