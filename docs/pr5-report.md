# PR 5 evidence: immutable action DFA and deterministic market trace

Status: local acceptance complete

Date: 2026-07-29

## Scope

PR 5 adds two portable CPU modules:

- `MarketForge::grammar`, containing invariant semantic actions, a checked
  trie-to-flat-DFA builder, and the generated SmolLM2 market-action catalog;
- `MarketForge::market`, containing an atomic integer frequent-batch market.

The model runtime still accepts token IDs and produces logits. It has no
tokenizer, grammar, or market-policy dependency.

## Locked tokenizer and generated catalog

The catalog uses the exact SmolLM2 repository revision already locked for the
weights:

| Field | Value |
|---|---|
| Repository | `HuggingFaceTB/SmolLM2-135M` |
| Revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Artifact | `tokenizer.json` |
| Bytes | 2,104,556 |
| SHA-256 | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| Generator package | `tokenizers==0.19.1` |
| Vocabulary | 49,152 |

The tokenizer stays outside Git. `tools/model/compile_action_dfa.py` verifies
its byte size, SHA-256, package version, and vocabulary before encoding. It
rejects special/unknown tokens, failed byte-for-byte decode round trips,
duplicate sequences, prefix collisions, out-of-range IDs, and sequences longer
than twelve tokens.

The checked-in generated artifact contains exactly 1,585 newline-terminated
actions. Observed lengths are 3–11 tokens. Regenerating twice is deterministic,
and the opt-in check against the verified tokenizer is byte-identical.

## Immutable DFA evidence

`Action` has no public default or raw-field constructor. `HOLD` fixes quantity
and tick to zero. Buy and sell factories accept only quantities 1–8 and ticks
1–99.

The DFA builder first constructs a checked temporary trie and then flattens it
into private state, arc, and terminal-action arrays. It rejects every malformed
language category frozen in `docs/pr5-contract.md`. The real catalog has:

| Measure | Value |
|---|---:|
| Semantic actions / terminal states | 1,585 |
| Reachable states | 3,230 |
| Arcs | 3,229 |
| Maximum outgoing candidate count | 11 |

Tests traverse every canonical sequence, decode every terminal, prove every
flat state reachable, prove each terminal action unique, and inspect every
state's strictly sorted outgoing arcs.

## Market semantics and exact trace

The market uses signed 64-bit cash, unsigned inventory, ticks 1–99, and 10,000
cash units per tick. Orders are validated in full at their limits. Clearing
maximizes matched volume, minimizes imbalance, chooses the nearest prior tick,
and finally the lower tick. Price and agent-ID priorities are deterministic.
All deltas are validated on a copy and committed atomically.

The executable obtains every trace action by traversing the generated DFA. Its
result is:

```text
batch 1 traded=1 tick=55 quantity=2
batch 2 traded=1 tick=58 quantity=3
batch 3 traded=0 tick=0 quantity=0
agent 0 cash=8900000 shares=2
agent 1 cash=8260000 shares=3
agent 2 cash=6100000 shares=3
agent 3 cash=6740000 shares=2
total cash=30000000 shares=10
last tick=58
```

The property suite checks all 24 input permutations of the first four-agent
batch and all 125 combinations of a bounded three-agent action space. Cash and
shares are conserved in every successful case. Fault tests cover duplicate and
unknown agents, insufficient cash and inventory, seller-cash overflow, and
atomic non-mutation.

## Local acceptance

Environment: Apple M3 Pro, ARM64, Apple Clang 16.

| Gate | Result |
|---|---|
| Debug warnings-as-errors build | pass |
| Debug CTest | 2/2 pass; 50 native cases |
| Release warnings-as-errors build and CTest | 2/2 pass |
| ASan + UBSan build and CTest | 2/2 pass |
| Python suite in the pinned project environment | 20 pass, 1 opt-in skip |
| Opt-in real-tokenizer Python test | 3/3 pass |
| Byte-identical generated catalog check | pass |
| Apple clang-format 16 dry run | pass |
| PR 4 guarded real-checkpoint conformance | pass in debug, release, and ASan/UBSan; all 147,456 logits and exact tokens |

The real-checkpoint regression used one measured repeat after the existing
warm-up windows. It reported zero live-heap and RSS growth between repeat
windows.

## Deliberate exclusions

No CUDA, Modal call, networked default test, native tokenizer, scheduler,
restricted output-projection kernel, Qwen support, model-weight download,
sampling, fee, shorting, settlement, or performance claim was added. The
lower-tick fallback is tested directly through the same clearing-candidate
comparator used by the market; with an integer prior tick it is normally
redundant after the earlier monotone-demand/supply tie-breaks.

The upstream model repository identifies the tokenizer/model as Apache-2.0.
MarketForge still has no project-level `LICENSE`; one should be selected before
public distribution.
