# Model artifacts

This directory contains model provenance metadata only. Weight files are cached
outside the source tree and ignored by Git.

[`model-lock.json`](model-lock.json) pins repositories, immutable revisions,
allowed files, sizes, SHA-256 hashes, licenses, and expected architectures.

The default development checkpoint is SmolLM2-135M: its complete BF16
safetensors file is 269,060,552 bytes. Its 2,104,556-byte `tokenizer.json` is
also locked so the finite PR 5 action catalog can be regenerated outside the
native runtime. Qwen2.5-0.5B is locked for later configuration testing, but its
988 MB weight file is deliberately not in the fetch allowlist.

Use `tools/model/fetch.py` with an explicit cache directory outside this source
tree. The downloader rejects branch names, unlisted files, oversize responses,
and hash mismatches.

Default tests do not fetch or load the tokenizer. With the verified artifact
already present, the opt-in offline reproducibility check is:

```sh
MARKETFORGE_TOKENIZER_JSON=/absolute/path/to/tokenizer.json \
  .venv/bin/python -m unittest tools.model.test_compile_action_dfa

.venv/bin/python tools/model/compile_action_dfa.py \
  --tokenizer /absolute/path/to/tokenizer.json \
  --output src/grammar/generated/smollm2_market_action_v1.inc \
  --check
```

Example opt-in fetch and full metadata bind:

```sh
python3 tools/model/fetch.py fetch \
  --model smollm2-135m \
  --cache-dir /absolute/path/to/marketforge-cache

./build/mac-debug/marketforge_model_inspect \
  /absolute/path/to/marketforge-cache/smollm2-135m/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json \
  /absolute/path/to/marketforge-cache/smollm2-135m/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors
```
