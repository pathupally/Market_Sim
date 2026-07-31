#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "marketforge/core/result.hpp"
#include "marketforge/grammar/action_dfa.hpp"

namespace marketforge {

using ChoiceId = std::uint32_t;

struct EncodedChoice {
  ChoiceId id{0};
  std::vector<token_id_t> tokens;
};

struct TokenDfaConfig {
  std::uint32_t vocabulary_size{0};
  std::uint32_t maximum_sequence_tokens{32};
  std::uint32_t maximum_choices{4096};
  std::uint32_t maximum_states{65536};
  std::uint32_t maximum_arcs{65536};
};

// A backend-neutral deterministic token trie for finite structured choices.
// Choice IDs carry application semantics outside the grammar layer.
class TokenDfa {
public:
  [[nodiscard]] static Result<TokenDfa>
  build(std::span<const EncodedChoice> language,
        TokenDfaConfig config) noexcept;

  [[nodiscard]] constexpr GrammarState root() const noexcept { return {}; }
  [[nodiscard]] Result<std::span<const GrammarArc>>
  allowed(GrammarState state) const noexcept;
  [[nodiscard]] Result<GrammarState> advance(GrammarState state,
                                             token_id_t token) const noexcept;
  [[nodiscard]] bool terminal(GrammarState state) const noexcept;
  [[nodiscard]] Result<ChoiceId>
  decode_terminal(GrammarState state) const noexcept;

  [[nodiscard]] std::size_t choice_count() const noexcept {
    return choices_.size();
  }
  [[nodiscard]] std::size_t state_count() const noexcept {
    return states_.size();
  }
  [[nodiscard]] std::size_t arc_count() const noexcept { return arcs_.size(); }

private:
  struct StateRecord {
    std::uint32_t first_arc{0};
    std::uint32_t arc_count{0};
    std::uint32_t terminal_choice{0};
  };

  static constexpr std::uint32_t no_terminal = UINT32_MAX;

  TokenDfa(std::vector<StateRecord> states, std::vector<GrammarArc> arcs,
           std::vector<ChoiceId> choices) noexcept
      : states_(std::move(states)), arcs_(std::move(arcs)),
        choices_(std::move(choices)) {}

  [[nodiscard]] bool valid(GrammarState state) const noexcept {
    return state.value < states_.size();
  }

  std::vector<StateRecord> states_;
  std::vector<GrammarArc> arcs_;
  std::vector<ChoiceId> choices_;
};

} // namespace marketforge
