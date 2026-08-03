# Model artifacts

This directory stores provenance metadata, not weights. Model files belong in
an external cache and are ignored by Git.

[`model-lock.json`](model-lock.json) pins the repository, immutable revision,
allowed filenames, byte sizes, SHA-256 hashes, license, and architecture for
each artifact.

SmolLM2-135M is the execution target. Its complete BF16 safetensors file is
269,060,552 bytes. The lock also covers its 2,104,556-byte `tokenizer.json`,
which is used offline to regenerate the finite-action DFA fixture. Qwen2.5-0.5B
is present only for configuration and capacity tests; its 988 MB weight file is
not in the fetch allowlist.

`tools/model/fetch.py` requires an explicit cache directory outside the source
tree. It rejects moving revisions, unlisted files, oversize responses, and hash
mismatches.

## Fetch and inspect

```sh
python3 tools/model/fetch.py fetch \
  --model smollm2-135m \
  --cache-dir /absolute/path/to/marketforge-cache

./build/mac-debug/marketforge_model_inspect \
  /absolute/path/to/marketforge-cache/smollm2-135m/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json \
  /absolute/path/to/marketforge-cache/smollm2-135m/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors
```

The fetch is opt-in. Default builds and tests do not access the network or load
model weights.

## Regenerate the DFA fixture

With the locked tokenizer already present outside the checkout:

```sh
MARKETFORGE_TOKENIZER_JSON=/absolute/path/to/tokenizer.json \
  .venv/bin/python -m unittest tools.model.test_compile_action_dfa

.venv/bin/python tools/model/compile_action_dfa.py \
  --tokenizer /absolute/path/to/tokenizer.json \
  --output src/grammar/generated/smollm2_market_action_v1.inc \
  --check
```

The generator writes model revision, tokenizer hash, package identity, and the
canonical source vocabulary into the fixture. `--check` fails if regeneration
would change the committed output.
