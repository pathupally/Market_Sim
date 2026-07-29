#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "marketforge/core/result.hpp"
#include "marketforge/grammar/action.hpp"

namespace marketforge {

using token_id_t = std::uint32_t;

struct GrammarState {
  std::uint32_t value{0};

  friend constexpr bool operator==(GrammarState, GrammarState) = default;
};

struct GrammarArc {
  token_id_t token{0};
  GrammarState next{};

  friend constexpr bool operator==(const GrammarArc&,
                                   const GrammarArc&) = default;
};

struct EncodedAction {
  Action action;
  std::vector<token_id_t> tokens;
};

struct ActionDfaConfig {
  std::uint32_t vocabulary_size{0};
  std::uint32_t maximum_sequence_tokens{12};
  std::uint32_t maximum_actions{4096};
  std::uint32_t maximum_states{65536};
  std::uint32_t maximum_arcs{65536};
};

class ActionDfa {
public:
  [[nodiscard]] static Result<ActionDfa>
  build(std::span<const EncodedAction> language,
        ActionDfaConfig config) noexcept;

  [[nodiscard]] constexpr GrammarState root() const noexcept { return {}; }
  [[nodiscard]] Result<std::span<const GrammarArc>>
  allowed(GrammarState state) const noexcept;
  [[nodiscard]] Result<GrammarState> advance(GrammarState state,
                                             token_id_t token) const noexcept;
  [[nodiscard]] bool terminal(GrammarState state) const noexcept;
  [[nodiscard]] Result<Action>
  decode_terminal(GrammarState state) const noexcept;

  [[nodiscard]] std::size_t action_count() const noexcept {
    return actions_.size();
  }
  [[nodiscard]] std::size_t state_count() const noexcept {
    return states_.size();
  }
  [[nodiscard]] std::size_t arc_count() const noexcept { return arcs_.size(); }

private:
  struct StateRecord {
    std::uint32_t first_arc{0};
    std::uint32_t arc_count{0};
    std::uint32_t terminal_action{0};
  };

  static constexpr std::uint32_t no_terminal = UINT32_MAX;

  ActionDfa(std::vector<StateRecord> states, std::vector<GrammarArc> arcs,
            std::vector<Action> actions) noexcept
      : states_(std::move(states)), arcs_(std::move(arcs)),
        actions_(std::move(actions)) {}

  [[nodiscard]] bool valid(GrammarState state) const noexcept {
    return state.value < states_.size();
  }

  std::vector<StateRecord> states_;
  std::vector<GrammarArc> arcs_;
  std::vector<Action> actions_;
};

} // namespace marketforge
