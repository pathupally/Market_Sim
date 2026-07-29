# PR 5 contract: immutable action DFA and deterministic market trace

Status: frozen

PR 5 adds a CPU-only, tokenizer-derived finite action grammar and a minimal
frequent-batch prediction market. It must consume the existing token-ID/logit
boundary without adding grammar or market policy to `CpuSmolLm2`.

## Locked inputs

- Model: `HuggingFaceTB/SmolLM2-135M`
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Tokenizer artifact: `tokenizer.json`
- Tokenizer size: `2,104,556` bytes
- Tokenizer SHA-256:
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`
- Tokenizer Git blob:
  `f922b1797f0c88e71addc8393787831f2477a4bd`
- Tokenizer package used by the generator: `tokenizers==0.19.1`
- Vocabulary size: `49,152`

The tokenizer remains in the external verified model cache. Add it to
`models/model-lock.json` and the existing bounded fetch allowlist. No default
test may download it or require it.

## Canonical action language

Semantic actions are:

```text
HOLD
BUY YES <quantity 1..8> @ <tick 1..99>
SELL YES <quantity 1..8> @ <tick 1..99>
```

The canonical transport strings end in one newline. There are exactly 1,585
actions. The locked tokenizer encodes them without special tokens in 3–11
tokens, with no duplicates, prefix collisions, unknown/special-token hits, or
decode round-trip failures.

Provide an invariant-preserving `Action` value type. `HOLD` must have zero
quantity and tick; buy/sell actions must have quantity 1–8 and tick 1–99.
Invalid actions must not be publicly constructible.

## Immutable DFA

Build a temporary checked trie and flatten it into private immutable state and
arc arrays. Outgoing arcs are sorted by token. Public operations provide:

- the root state;
- the allowed outgoing arcs for a state;
- a checked state transition;
- terminal-state inspection;
- checked terminal-action decoding;
- action, state, and arc counts.

Construction rejects:

- empty languages or token sequences;
- duplicate semantic actions or token sequences;
- prefix collisions;
- token IDs outside the configured vocabulary;
- sequences longer than 12 tokens;
- invalid or exceeded action/state/arc resource limits;
- any inconsistent terminal state.

The checked-in SmolLM2 catalog is generated deterministically from the locked
tokenizer. Regeneration must be byte-identical. Synthetic token IDs are allowed
only in builder fault tests.

## Deterministic frequent-batch market

Use integer accounting:

- tick range: 1–99;
- cash units per tick: 10,000;
- one YES contract;
- no fees, credit, shorting, persistent orders, or cancellations.

Each registered agent has a unique ID, signed 64-bit cash, and unsigned YES
inventory. Each agent may submit at most one action per batch. Missing actions
are HOLD. Duplicate or unknown IDs reject the complete batch.

Validate a buyer against its complete order at its limit and a seller against
its complete submitted inventory. Checked arithmetic is mandatory. Any invalid
input, insufficient resources, or overflow leaves all accounts and the last
traded tick unchanged.

For every candidate tick `t`:

```text
demand(t)  = sum buy quantities whose limit is >= t
supply(t)  = sum sell quantities whose limit is <= t
matched(t) = min(demand(t), supply(t))
```

Choose the lexicographically smallest key:

```text
(-matched volume, absolute demand/supply imbalance,
 distance from last traded tick, lower tick)
```

If maximum matched volume is zero, report no trade and preserve the last tick.
Otherwise use one uniform clearing tick. Buyer priority is higher limit then
lower agent ID; seller priority is lower limit then lower agent ID. Return
fills in stable agent-ID order.

## Canonical three-batch trace

Initial state:

```text
last tick = 50
agent 0: cash 10,000,000, shares 0
agent 1: cash 10,000,000, shares 0
agent 2: cash  5,000,000, shares 5
agent 3: cash  5,000,000, shares 5
```

1. Agent 0 buys 2 at 60; agent 2 sells 2 at 55. Clear 2 at 55.
2. Agent 1 buys 3 at 58; agent 2 sells 1 at 59; agent 3 sells 3 at
   58. Clear 3 at 58.
3. Agent 0 buys 1 at 40; agent 2 sells 1 at 70. No trade.

Final state:

```text
agent 0: cash 8,900,000, shares 2
agent 1: cash 8,260,000, shares 3
agent 2: cash 6,100,000, shares 3
agent 3: cash 6,740,000, shares 2
total:   cash 30,000,000, shares 10
last tick = 58
```

The trace executable must obtain its semantic actions by traversing the
generated DFA token sequences.

## Acceptance gates

- Exhaustively traverse and decode all 1,585 canonical sequences.
- Prove every flattened state is reachable, outgoing arcs are unique/sorted,
  and each terminal action appears exactly once.
- Cover every builder rejection category and representative invalid runtime
  transitions.
- Regenerate the catalog twice and compare bytes; opt-in regeneration from the
  verified cached tokenizer must match the checked-in artifact.
- Test HOLD, crossing, no-cross, partial fills, all clearing tie-breaks, and
  both price/agent priority rules.
- Test duplicate/unknown agents, insufficient cash/inventory, and arithmetic
  failure for atomic non-mutation.
- Test all 24 input permutations of the canonical four-agent batch.
- Exhaustively conserve cash and shares over a bounded small action space.
- Match the exact three-batch trace and final state.
- Preserve all PR 1–4 debug, release, sanitizer, Python, formatting, and
  guarded checkpoint-conformance gates.
- Use no CUDA, Modal, model-weight download, networked default test, native
  tokenizer, scheduler, quantization, Qwen support, or performance claim.
