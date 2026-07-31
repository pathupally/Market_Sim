#include <array>
#include <cstdint>
#include <vector>

#include "marketforge/core/status.hpp"
#include "marketforge/grammar/token_dfa.hpp"
#include "test_support.hpp"

namespace {

using marketforge::ChoiceId;
using marketforge::EncodedChoice;
using marketforge::ErrorCode;
using marketforge::GrammarState;
using marketforge::TokenDfa;
using marketforge::TokenDfaConfig;

TokenDfaConfig config() {
  return TokenDfaConfig{64, 8, 16, 32, 32};
}

MF_TEST(token_dfa_builds_generic_choices_and_decodes_terminal_ids) {
  const std::array language{
      EncodedChoice{41, {9, 4}},
      EncodedChoice{7, {3, 8, 1}},
      EncodedChoice{99, {9, 6}},
  };
  auto result = TokenDfa::build(language, config());
  MF_CHECK(result);
  auto dfa = std::move(result).value();
  MF_CHECK_EQ(dfa.choice_count(), 3);
  MF_CHECK_EQ(dfa.state_count(), 7);
  MF_CHECK_EQ(dfa.arc_count(), 6);

  const auto root = dfa.allowed(dfa.root());
  MF_CHECK(root);
  MF_CHECK_EQ(root.value().size(), 2);
  MF_CHECK_EQ(root.value()[0].token, 3);
  MF_CHECK_EQ(root.value()[1].token, 9);

  auto state = dfa.advance(dfa.root(), 9);
  MF_CHECK(state);
  state = dfa.advance(state.value(), 6);
  MF_CHECK(state);
  MF_CHECK(dfa.terminal(state.value()));
  const auto decoded = dfa.decode_terminal(state.value());
  MF_CHECK(decoded);
  MF_CHECK_EQ(decoded.value(), ChoiceId{99});
}

MF_TEST(token_dfa_rejects_ambiguous_or_invalid_languages_atomically) {
  const std::array duplicate_ids{
      EncodedChoice{1, {2}},
      EncodedChoice{1, {3}},
  };
  MF_CHECK_EQ(TokenDfa::build(duplicate_ids, config()).status().code,
              ErrorCode::invalid_argument);

  const std::array prefix_collision{
      EncodedChoice{1, {2}},
      EncodedChoice{2, {2, 3}},
  };
  MF_CHECK_EQ(TokenDfa::build(prefix_collision, config()).status().code,
              ErrorCode::invalid_argument);

  const std::array out_of_range{EncodedChoice{1, {64}}};
  MF_CHECK_EQ(TokenDfa::build(out_of_range, config()).status().code,
              ErrorCode::invalid_argument);

  const std::array reserved_id{EncodedChoice{UINT32_MAX, {1}}};
  MF_CHECK_EQ(TokenDfa::build(reserved_id, config()).status().code,
              ErrorCode::invalid_argument);
}

MF_TEST(token_dfa_enforces_explicit_resource_limits) {
  const std::array language{
      EncodedChoice{1, {1, 2}},
      EncodedChoice{2, {3, 4}},
  };
  auto limited = config();
  limited.maximum_states = 2;
  MF_CHECK_EQ(TokenDfa::build(language, limited).status().code,
              ErrorCode::resource_limit);

  const auto result = TokenDfa::build(language, config());
  MF_CHECK(result);
  const auto& dfa = result.value();
  MF_CHECK_EQ(dfa.allowed(GrammarState{99}).status().code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(dfa.advance(dfa.root(), 63).status().code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(dfa.decode_terminal(dfa.root()).status().code,
              ErrorCode::invalid_argument);
}

} // namespace
