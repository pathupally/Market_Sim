# MarketForge

MarketForge is a C++20/CUDA inference runtime for large populations of
stateful, short-output agents. Its demonstration workload is a small
prediction-market simulation; the primary engineering work is inference
scheduling, KV-cache ownership, constrained decoding, and CUDA performance.

The implementation is complete through PR 5: portable tensor/model contracts,
safe mapped SmolLM2 loading, a readable FP32 CPU decoder-layer oracle,
pretokenized full-model greedy decode, an immutable tokenizer-derived action
DFA, and a deterministic integer-accounted market trace. The locked 30-layer
checkpoint produces complete logits and margin-qualified greedy tokens that
match a pinned PyTorch fixture. Repeated decode retains fixed runtime storage.
CUDA is intentionally not enabled yet.

## Build on this Mac

```sh
cmake --preset mac-debug
cmake --build --preset mac-debug
ctest --preset mac-debug
```

Sanitizers:

```sh
cmake --preset mac-sanitize
cmake --build --preset mac-sanitize
ctest --preset mac-sanitize
```

## Model policy

Weights are never committed and no default test downloads a model.

- Development: SmolLM2-135M, 269 MB of BF16 safetensors.
- Later demonstration: Qwen2.5-0.5B-Instruct, 988 MB of BF16 safetensors.

Both models fit the 36 GB M3 Pro development machine, including a conservative
BF16-mapped plus FP32-materialized CPU reference. See
[the footprint report](docs/model-footprint.md) and the
[canonical architecture plan](docs/inference-runtime-plan.md).

The locked 269 MB SmolLM2 checkpoint has been downloaded to the user cache and
passed SHA-256 verification plus a full 30-layer metadata bind on this machine.
Its 2,104,556-byte `tokenizer.json` is independently locked for opt-in offline
action-catalog regeneration; no default test needs that external file.
Qwen remains config-only; its 988 MB weight file is blocked by the default
fetch manifest.

PR 3's committed numerical oracle is only 7,936 bytes. It was generated with
PyTorch 2.3.1 and Transformers 4.40.1 against the locked SmolLM2 architecture
revision; it does not contain pretrained model weights. See
[the PR 3 evidence report](docs/pr3-report.md).

PR 4 adds a 590,600-byte full-logit fixture from explicit token IDs
`[0, 1, 2, 3]`. The C++ FP32 path compares all 147,456 logits and exactly
reproduces three greedy tokens without implementing tokenization. See
[the PR 4 evidence report](docs/pr4-report.md).

PR 5 compiles the finite 1,585-action market language with pinned Python
tooling into checked-in token IDs. Native code consumes only immutable token
tables and exposes semantic actions; it does not implement a tokenizer. Run the
deterministic demonstration with:

```sh
./build/mac-debug/marketforge_market_trace
```

See [the PR 5 evidence report](docs/pr5-report.md).

The canonical real-model conformance path verifies the supplied checkpoint
against `models/model-lock.json`, hashes the fixture, and verifies that the
fixture records the same checkpoint before launching C++:

```sh
.venv/bin/python -m tools.model.conformance \
  --executable build/mac-debug/marketforge_smollm2_conformance \
  --checkpoint /path/to/cache/smollm2-135m/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors \
  --fixture tests/fixtures/golden/smollm2-pr4-greedy-f32.safetensors
```

## Bounded Modal validation

Linux portability checks run on a CPU-only Modal container. The repository
policy reserves $6 of the $30 monthly budget and stops discretionary project
jobs at a $24 soft cap. See [the Modal job guide](tools/modal/README.md).
