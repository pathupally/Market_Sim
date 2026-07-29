#include "test_support.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "marketforge/market/batch_market.hpp"

namespace {

using marketforge::Account;
using marketforge::Action;
using marketforge::ActionKind;
using marketforge::AgentAction;
using marketforge::BatchMarket;
using marketforge::BatchResult;
using marketforge::ClearingCandidate;
using marketforge::ErrorCode;
using marketforge::Fill;

Action buy(std::uint32_t quantity, std::uint32_t tick) {
  const auto result = Action::buy_yes(quantity, tick);
  MF_CHECK(result);
  return result.value();
}

Action sell(std::uint32_t quantity, std::uint32_t tick) {
  const auto result = Action::sell_yes(quantity, tick);
  MF_CHECK(result);
  return result.value();
}

BatchMarket market(std::span<const Account> accounts,
                   std::uint32_t reference = 50) {
  auto result = BatchMarket::create(accounts, reference);
  MF_CHECK(result);
  return std::move(result).value();
}

const Fill* fill_for(const BatchResult& result, std::uint32_t agent) {
  const auto iterator =
      std::find_if(result.fills.begin(), result.fills.end(),
                   [agent](const Fill& fill) { return fill.agent == agent; });
  return iterator == result.fills.end() ? nullptr : &*iterator;
}

MF_TEST(batch_market_validates_initial_state) {
  const std::vector<Account> empty;
  MF_CHECK(!BatchMarket::create(empty, 50));
  MF_CHECK(!BatchMarket::create(std::vector<Account>{{0, 1, 0}}, 0));
  MF_CHECK(!BatchMarket::create(std::vector<Account>{{0, 1, 0}}, 100));
  MF_CHECK(
      !BatchMarket::create(std::vector<Account>{{0, 1, 0}, {0, 2, 0}}, 50));
  MF_CHECK(!BatchMarket::create(std::vector<Account>{{0, -1, 0}}, 50));
}

MF_TEST(clearing_candidate_comparator_covers_every_lexicographic_tie_break) {
  const ClearingCandidate incumbent{4, 3, 2, 51};
  MF_CHECK(marketforge::better_clearing_candidate(
      ClearingCandidate{5, 99, 99, 99}, incumbent));
  MF_CHECK(marketforge::better_clearing_candidate(
      ClearingCandidate{4, 2, 99, 99}, incumbent));
  MF_CHECK(marketforge::better_clearing_candidate(
      ClearingCandidate{4, 3, 1, 99}, incumbent));
  MF_CHECK(marketforge::better_clearing_candidate(
      ClearingCandidate{4, 3, 2, 49}, incumbent));
  MF_CHECK(!marketforge::better_clearing_candidate(
      ClearingCandidate{4, 3, 2, 52}, incumbent));
}

MF_TEST(batch_market_crosses_at_one_uniform_price_and_handles_no_trade) {
  auto state = market(std::vector<Account>{
      {0, 10000000, 0},
      {1, 5000000, 5},
  });
  const auto crossed = state.clear(std::vector<AgentAction>{
      {0, buy(2, 60)},
      {1, sell(2, 55)},
  });
  MF_CHECK(crossed);
  MF_CHECK(crossed.value().traded);
  MF_CHECK_EQ(crossed.value().clearing_tick, 55);
  MF_CHECK_EQ(crossed.value().matched_quantity, 2);
  MF_CHECK_EQ(state.accounts()[0], (Account{0, 8900000, 2}));
  MF_CHECK_EQ(state.accounts()[1], (Account{1, 6100000, 3}));

  const auto before =
      std::vector<Account>(state.accounts().begin(), state.accounts().end());
  const auto no_trade = state.clear(std::vector<AgentAction>{
      {0, buy(1, 40)},
      {1, sell(1, 70)},
  });
  MF_CHECK(no_trade);
  MF_CHECK(!no_trade.value().traded);
  MF_CHECK_EQ(no_trade.value().clearing_tick, 0);
  MF_CHECK_EQ(no_trade.value().matched_quantity, 0);
  MF_CHECK_EQ(state.last_traded_tick(), 55);
  MF_CHECK(std::equal(state.accounts().begin(), state.accounts().end(),
                      before.begin(), before.end()));
}

MF_TEST(batch_market_clearing_uses_volume_imbalance_and_reference_ties) {
  {
    auto state = market(std::vector<Account>{
        {0, 10000000, 0},
        {1, 5000000, 1},
        {2, 5000000, 1},
    });
    const auto result = state.clear(std::vector<AgentAction>{
        {0, buy(2, 60)},
        {1, sell(1, 40)},
        {2, sell(1, 60)},
    });
    MF_CHECK(result);
    MF_CHECK_EQ(result.value().matched_quantity, 2);
    MF_CHECK_EQ(result.value().clearing_tick, 60);
  }
  {
    auto state = market(std::vector<Account>{
        {0, 10000000, 0},
        {1, 10000000, 0},
        {2, 5000000, 1},
        {3, 5000000, 1},
    });
    const auto result = state.clear(std::vector<AgentAction>{
        {0, buy(1, 60)},
        {1, buy(1, 40)},
        {2, sell(1, 40)},
        {3, sell(1, 60)},
    });
    MF_CHECK(result);
    MF_CHECK_EQ(result.value().matched_quantity, 1);
    MF_CHECK_EQ(result.value().clearing_tick, 50);
  }
  {
    auto state =
        market(std::vector<Account>{{0, 10000000, 0}, {1, 5000000, 1}}, 53);
    const auto result = state.clear(std::vector<AgentAction>{
        {0, buy(1, 60)},
        {1, sell(1, 40)},
    });
    MF_CHECK(result);
    MF_CHECK_EQ(result.value().clearing_tick, 53);
  }
}

MF_TEST(batch_market_price_then_agent_priority_and_partial_fills_are_exact) {
  {
    auto state = market(std::vector<Account>{
        {0, 10000000, 0},
        {1, 10000000, 0},
        {2, 5000000, 1},
    });
    const auto result = state.clear(std::vector<AgentAction>{
        {0, buy(1, 55)},
        {1, buy(1, 60)},
        {2, sell(1, 50)},
    });
    MF_CHECK(result);
    MF_CHECK(fill_for(result.value(), 1) != nullptr);
    MF_CHECK(fill_for(result.value(), 0) == nullptr);
  }
  {
    auto state = market(std::vector<Account>{
        {0, 10000000, 0},
        {1, 10000000, 0},
        {2, 5000000, 4},
    });
    const auto result = state.clear(std::vector<AgentAction>{
        {0, buy(3, 60)},
        {1, buy(3, 60)},
        {2, sell(4, 50)},
    });
    MF_CHECK(result);
    MF_CHECK_EQ(result.value().matched_quantity, 4);
    MF_CHECK_EQ(fill_for(result.value(), 0)->quantity, 3);
    MF_CHECK_EQ(fill_for(result.value(), 1)->quantity, 1);
  }
  {
    auto state = market(std::vector<Account>{
        {0, 10000000, 0},
        {1, 5000000, 3},
        {2, 5000000, 3},
    });
    const auto result = state.clear(std::vector<AgentAction>{
        {0, buy(4, 60)},
        {1, sell(3, 50)},
        {2, sell(3, 50)},
    });
    MF_CHECK(result);
    MF_CHECK_EQ(fill_for(result.value(), 1)->quantity, 3);
    MF_CHECK_EQ(fill_for(result.value(), 2)->quantity, 1);
  }
  {
    auto state = market(std::vector<Account>{
        {0, 10000000, 0},
        {1, 5000000, 1},
        {2, 5000000, 1},
    });
    const auto result = state.clear(std::vector<AgentAction>{
        {0, buy(1, 60)},
        {1, sell(1, 50)},
        {2, sell(1, 40)},
    });
    MF_CHECK(result);
    MF_CHECK(fill_for(result.value(), 2) != nullptr);
    MF_CHECK(fill_for(result.value(), 1) == nullptr);
  }
}

MF_TEST(batch_market_rejects_faults_atomically) {
  auto state = market(std::vector<Account>{
      {0, 1000000, 0},
      {1, 1000000, 1},
  });
  const auto initial =
      std::vector<Account>(state.accounts().begin(), state.accounts().end());
  const auto initial_tick = state.last_traded_tick();

  const std::vector<std::vector<AgentAction>> invalid_batches{
      {{0, Action::hold()}, {0, Action::hold()}},
      {{99, Action::hold()}},
      {{0, buy(2, 99)}},
      {{1, sell(2, 50)}},
  };
  for (const auto& batch : invalid_batches) {
    const auto result = state.clear(batch);
    MF_CHECK(!result);
    MF_CHECK(std::equal(state.accounts().begin(), state.accounts().end(),
                        initial.begin(), initial.end()));
    MF_CHECK_EQ(state.last_traded_tick(), initial_tick);
  }

  auto overflow = market(std::vector<Account>{
      {0, 10000000, 0},
      {1, std::numeric_limits<std::int64_t>::max(), 1},
  });
  const auto overflow_before = std::vector<Account>(overflow.accounts().begin(),
                                                    overflow.accounts().end());
  const auto result = overflow.clear(std::vector<AgentAction>{
      {0, buy(1, 50)},
      {1, sell(1, 50)},
  });
  MF_CHECK(!result);
  MF_CHECK_EQ(result.status().code, ErrorCode::arithmetic_overflow);
  MF_CHECK(std::equal(overflow.accounts().begin(), overflow.accounts().end(),
                      overflow_before.begin(), overflow_before.end()));
  MF_CHECK_EQ(overflow.last_traded_tick(), 50);
}

} // namespace
