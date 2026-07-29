#include "marketforge/grammar/action_dfa.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace marketforge {

namespace {

struct TrieNode {
  std::map<token_id_t, std::uint32_t> children;
  std::uint32_t terminal_action{UINT32_MAX};
};

using ActionKey = std::tuple<ActionKind, std::uint8_t, std::uint8_t>;

ActionKey key(Action action) noexcept {
  return {action.kind(), action.quantity(), action.price_tick()};
}

Result<ActionDfa> invalid_dfa(ErrorCode code = ErrorCode::invalid_argument) {
  return Result<ActionDfa>::failure(Status::failure(code));
}

} // namespace

Result<ActionDfa> ActionDfa::build(std::span<const EncodedAction> language,
                                   ActionDfaConfig config) noexcept {
  try {
    if (language.empty() || config.vocabulary_size == 0 ||
        config.maximum_sequence_tokens == 0 || config.maximum_actions == 0 ||
        config.maximum_states == 0 || config.maximum_arcs == 0) {
      return invalid_dfa();
    }
    if (language.size() > config.maximum_actions) {
      return invalid_dfa(ErrorCode::resource_limit);
    }

    std::vector<TrieNode> trie(1);
    std::vector<Action> actions;
    actions.reserve(language.size());
    std::set<ActionKey> action_keys;
    std::uint32_t arc_count = 0;

    for (const auto& encoded : language) {
      if (encoded.tokens.empty() ||
          encoded.tokens.size() > config.maximum_sequence_tokens) {
        return invalid_dfa();
      }
      if (!action_keys.insert(key(encoded.action)).second) {
        return invalid_dfa();
      }

      std::uint32_t state = 0;
      for (const auto token : encoded.tokens) {
        if (token >= config.vocabulary_size) {
          return invalid_dfa();
        }
        if (trie[state].terminal_action != UINT32_MAX) {
          return invalid_dfa();
        }
        auto iterator = trie[state].children.find(token);
        if (iterator == trie[state].children.end()) {
          if (trie.size() >= config.maximum_states ||
              arc_count >= config.maximum_arcs) {
            return invalid_dfa(ErrorCode::resource_limit);
          }
          const auto next = static_cast<std::uint32_t>(trie.size());
          trie.emplace_back();
          iterator = trie[state].children.emplace(token, next).first;
          ++arc_count;
        }
        state = iterator->second;
      }

      if (trie[state].terminal_action != UINT32_MAX ||
          !trie[state].children.empty()) {
        return invalid_dfa();
      }
      trie[state].terminal_action = static_cast<std::uint32_t>(actions.size());
      actions.push_back(encoded.action);
    }

    std::vector<StateRecord> states;
    std::vector<GrammarArc> arcs;
    states.reserve(trie.size());
    arcs.reserve(arc_count);
    for (const auto& node : trie) {
      states.push_back(StateRecord{
          static_cast<std::uint32_t>(arcs.size()),
          static_cast<std::uint32_t>(node.children.size()),
          node.terminal_action,
      });
      for (const auto& [token, next] : node.children) {
        arcs.push_back(GrammarArc{token, GrammarState{next}});
      }
    }
    if (arcs.size() != arc_count || actions.size() != language.size()) {
      return invalid_dfa(ErrorCode::invalid_format);
    }
    return Result<ActionDfa>::success(
        ActionDfa(std::move(states), std::move(arcs), std::move(actions)));
  } catch (const std::bad_alloc&) {
    return invalid_dfa(ErrorCode::allocation_failed);
  }
}

Result<std::span<const GrammarArc>>
ActionDfa::allowed(GrammarState state) const noexcept {
  if (!valid(state)) {
    return Result<std::span<const GrammarArc>>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  const auto& record = states_[state.value];
  return Result<std::span<const GrammarArc>>::success(
      std::span<const GrammarArc>(arcs_).subspan(record.first_arc,
                                                 record.arc_count));
}

Result<GrammarState> ActionDfa::advance(GrammarState state,
                                        token_id_t token) const noexcept {
  const auto outgoing = allowed(state);
  if (!outgoing) {
    return Result<GrammarState>::failure(outgoing.status());
  }
  const auto arcs = outgoing.value();
  const auto iterator =
      std::lower_bound(arcs.begin(), arcs.end(), token,
                       [](const GrammarArc& arc, token_id_t candidate) {
                         return arc.token < candidate;
                       });
  if (iterator == arcs.end() || iterator->token != token) {
    return Result<GrammarState>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return Result<GrammarState>::success(iterator->next);
}

bool ActionDfa::terminal(GrammarState state) const noexcept {
  return valid(state) && states_[state.value].terminal_action != no_terminal;
}

Result<Action> ActionDfa::decode_terminal(GrammarState state) const noexcept {
  if (!valid(state)) {
    return Result<Action>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  const auto index = states_[state.value].terminal_action;
  if (index == no_terminal || index >= actions_.size()) {
    return Result<Action>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return Result<Action>::success(actions_[index]);
}

} // namespace marketforge
