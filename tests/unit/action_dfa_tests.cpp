#include "test_support.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "marketforge/grammar/action_dfa.hpp"

namespace {

using marketforge::Action;
using marketforge::ActionDfa;
using marketforge::ActionDfaConfig;
using marketforge::EncodedAction;
using marketforge::ErrorCode;
using marketforge::GrammarState;

Action buy(std::uint32_t quantity, std::uint32_t tick) {
  const auto action = Action::buy_yes(quantity, tick);
  MF_CHECK(action);
  return action.value();
}

Action sell(std::uint32_t quantity, std::uint32_t tick) {
  const auto action = Action::sell_yes(quantity, tick);
  MF_CHECK(action);
  return action.value();
}

ActionDfaConfig config() {
  return ActionDfaConfig{100, 12, 4096, 65536, 65536};
}

MF_TEST(action_invariants_are_enforced_at_construction) {
  static_assert(!std::is_default_constructible_v<Action>);
  static_assert(std::is_copy_constructible_v<Action>);

  const auto hold = Action::hold();
  MF_CHECK_EQ(hold.kind(), marketforge::ActionKind::hold);
  MF_CHECK_EQ(hold.quantity(), 0);
  MF_CHECK_EQ(hold.price_tick(), 0);

  for (const auto quantity : {0U, 9U}) {
    MF_CHECK(!Action::buy_yes(quantity, 50));
    MF_CHECK(!Action::sell_yes(quantity, 50));
  }
  for (const auto tick : {0U, 100U}) {
    MF_CHECK(!Action::buy_yes(1, tick));
    MF_CHECK(!Action::sell_yes(1, tick));
  }
  MF_CHECK(Action::buy_yes(1, 1));
  MF_CHECK(Action::buy_yes(8, 99));
  MF_CHECK(Action::sell_yes(1, 1));
  MF_CHECK(Action::sell_yes(8, 99));
}

MF_TEST(action_dfa_builds_sorted_checked_transitions) {
  const std::vector<EncodedAction> language{
      {Action::hold(), {9}},
      {buy(1, 1), {2, 5}},
      {sell(1, 1), {2, 3}},
  };
  const auto built = ActionDfa::build(language, config());
  MF_CHECK(built);
  const auto& dfa = built.value();
  MF_CHECK_EQ(dfa.action_count(), 3);
  MF_CHECK_EQ(dfa.state_count(), 5);
  MF_CHECK_EQ(dfa.arc_count(), 4);

  const auto root_arcs = dfa.allowed(dfa.root());
  MF_CHECK(root_arcs);
  MF_CHECK_EQ(root_arcs.value().size(), 2);
  MF_CHECK_EQ(root_arcs.value()[0].token, 2);
  MF_CHECK_EQ(root_arcs.value()[1].token, 9);

  auto state = dfa.advance(dfa.root(), 2);
  MF_CHECK(state);
  MF_CHECK(!dfa.terminal(state.value()));
  state = dfa.advance(state.value(), 3);
  MF_CHECK(state);
  MF_CHECK(dfa.terminal(state.value()));
  const auto decoded = dfa.decode_terminal(state.value());
  MF_CHECK(decoded);
  MF_CHECK_EQ(decoded.value(), sell(1, 1));

  MF_CHECK(!dfa.advance(dfa.root(), 99));
  MF_CHECK(!dfa.allowed(GrammarState{999}));
  MF_CHECK(!dfa.decode_terminal(dfa.root()));
  MF_CHECK(!dfa.decode_terminal(GrammarState{999}));
}

MF_TEST(action_dfa_rejects_all_malformed_language_categories) {
  const std::vector<EncodedAction> empty;
  MF_CHECK(!ActionDfa::build(empty, config()));

  MF_CHECK(!ActionDfa::build(std::vector<EncodedAction>{{Action::hold(), {}}},
                             config()));

  auto too_long = std::vector<std::uint32_t>(13, 1);
  MF_CHECK(!ActionDfa::build(
      std::vector<EncodedAction>{{Action::hold(), std::move(too_long)}},
      config()));

  MF_CHECK(!ActionDfa::build(
      std::vector<EncodedAction>{{Action::hold(), {100}}}, config()));

  MF_CHECK(!ActionDfa::build(
      std::vector<EncodedAction>{
          {Action::hold(), {1}},
          {Action::hold(), {2}},
      },
      config()));

  MF_CHECK(!ActionDfa::build(
      std::vector<EncodedAction>{
          {Action::hold(), {1}},
          {buy(1, 1), {1}},
      },
      config()));

  MF_CHECK(!ActionDfa::build(
      std::vector<EncodedAction>{
          {Action::hold(), {1}},
          {buy(1, 1), {1, 2}},
      },
      config()));
  MF_CHECK(!ActionDfa::build(
      std::vector<EncodedAction>{
          {Action::hold(), {1, 2}},
          {buy(1, 1), {1}},
      },
      config()));

  auto action_limited = config();
  action_limited.maximum_actions = 1;
  const auto two_actions = std::vector<EncodedAction>{
      {Action::hold(), {1}},
      {buy(1, 1), {2}},
  };
  const auto action_failure = ActionDfa::build(two_actions, action_limited);
  MF_CHECK(!action_failure);
  MF_CHECK_EQ(action_failure.status().code, ErrorCode::resource_limit);

  auto state_limited = config();
  state_limited.maximum_states = 1;
  MF_CHECK(!ActionDfa::build(std::vector<EncodedAction>{{Action::hold(), {1}}},
                             state_limited));

  auto arc_limited = config();
  arc_limited.maximum_arcs = 1;
  MF_CHECK(!ActionDfa::build(
      std::vector<EncodedAction>{{Action::hold(), {1, 2}}}, arc_limited));

  auto invalid_config = config();
  invalid_config.vocabulary_size = 0;
  MF_CHECK(!ActionDfa::build(std::vector<EncodedAction>{{Action::hold(), {1}}},
                             invalid_config));
}

} // namespace
