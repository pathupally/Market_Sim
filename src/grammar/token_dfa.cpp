#include "marketforge/grammar/token_dfa.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

namespace marketforge {
namespace {

struct TrieNode {
  std::map<token_id_t, std::uint32_t> children;
  std::uint32_t terminal_choice{UINT32_MAX};
};

Result<TokenDfa> invalid_dfa(
    const ErrorCode code = ErrorCode::invalid_argument) {
  return Result<TokenDfa>::failure(Status::failure(code));
}

} // namespace

Result<TokenDfa> TokenDfa::build(
    const std::span<const EncodedChoice> language,
    const TokenDfaConfig config) noexcept {
  try {
    if (language.empty() || config.vocabulary_size == 0 ||
        config.maximum_sequence_tokens == 0 ||
        config.maximum_choices == 0 || config.maximum_states == 0 ||
        config.maximum_arcs == 0) {
      return invalid_dfa();
    }
    if (language.size() > config.maximum_choices) {
      return invalid_dfa(ErrorCode::resource_limit);
    }

    std::vector<TrieNode> trie(1);
    std::vector<ChoiceId> choices;
    choices.reserve(language.size());
    std::set<ChoiceId> choice_ids;
    std::uint32_t arc_count = 0;
    for (const auto& encoded : language) {
      if (encoded.id == no_terminal || encoded.tokens.empty() ||
          encoded.tokens.size() > config.maximum_sequence_tokens ||
          !choice_ids.insert(encoded.id).second) {
        return invalid_dfa();
      }
      std::uint32_t state = 0;
      for (const auto token : encoded.tokens) {
        if (token >= config.vocabulary_size ||
            trie[state].terminal_choice != no_terminal) {
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
      if (trie[state].terminal_choice != no_terminal ||
          !trie[state].children.empty()) {
        return invalid_dfa();
      }
      trie[state].terminal_choice = static_cast<std::uint32_t>(choices.size());
      choices.push_back(encoded.id);
    }

    std::vector<StateRecord> states;
    std::vector<GrammarArc> arcs;
    states.reserve(trie.size());
    arcs.reserve(arc_count);
    for (const auto& node : trie) {
      states.push_back(StateRecord{
          static_cast<std::uint32_t>(arcs.size()),
          static_cast<std::uint32_t>(node.children.size()),
          node.terminal_choice,
      });
      for (const auto& [token, next] : node.children) {
        arcs.push_back(GrammarArc{token, GrammarState{next}});
      }
    }
    if (arcs.size() != arc_count || choices.size() != language.size()) {
      return invalid_dfa(ErrorCode::invalid_format);
    }
    return Result<TokenDfa>::success(TokenDfa(
        std::move(states), std::move(arcs), std::move(choices)));
  } catch (const std::bad_alloc&) {
    return invalid_dfa(ErrorCode::allocation_failed);
  }
}

Result<std::span<const GrammarArc>>
TokenDfa::allowed(const GrammarState state) const noexcept {
  if (!valid(state)) {
    return Result<std::span<const GrammarArc>>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  const auto& record = states_[state.value];
  return Result<std::span<const GrammarArc>>::success(
      std::span<const GrammarArc>(arcs_).subspan(record.first_arc,
                                                 record.arc_count));
}

Result<GrammarState> TokenDfa::advance(const GrammarState state,
                                       const token_id_t token) const noexcept {
  const auto outgoing = allowed(state);
  if (!outgoing) {
    return Result<GrammarState>::failure(outgoing.status());
  }
  const auto arcs = outgoing.value();
  const auto iterator =
      std::lower_bound(arcs.begin(), arcs.end(), token,
                       [](const GrammarArc& arc, const token_id_t candidate) {
                         return arc.token < candidate;
                       });
  if (iterator == arcs.end() || iterator->token != token) {
    return Result<GrammarState>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return Result<GrammarState>::success(iterator->next);
}

bool TokenDfa::terminal(const GrammarState state) const noexcept {
  return valid(state) && states_[state.value].terminal_choice != no_terminal;
}

Result<ChoiceId>
TokenDfa::decode_terminal(const GrammarState state) const noexcept {
  if (!valid(state)) {
    return Result<ChoiceId>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  const auto index = states_[state.value].terminal_choice;
  if (index == no_terminal || index >= choices_.size()) {
    return Result<ChoiceId>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return Result<ChoiceId>::success(choices_[index]);
}

} // namespace marketforge
