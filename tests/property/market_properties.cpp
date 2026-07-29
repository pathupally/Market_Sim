#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include "marketforge/grammar/smollm2_market_actions.hpp"
#include "marketforge/market/batch_market.hpp"

namespace {

using marketforge::Account;
using marketforge::Action;
using marketforge::AgentAction;
using marketforge::BatchMarket;

Action require_buy(std::uint32_t quantity, std::uint32_t tick) {
  const auto action = Action::buy_yes(quantity, tick);
  MF_CHECK(action);
  return action.value();
}

Action require_sell(std::uint32_t quantity, std::uint32_t tick) {
  const auto action = Action::sell_yes(quantity, tick);
  MF_CHECK(action);
  return action.value();
}

std::pair<std::int64_t, std::uint64_t>
totals(std::span<const Account> accounts) {
  std::int64_t cash = 0;
  std::uint64_t shares = 0;
  for (const auto& account : accounts) {
    cash += account.cash;
    shares += account.yes_shares;
  }
  return {cash, shares};
}

MF_TEST(canonical_market_batch_is_invariant_over_all_24_input_permutations) {
  const std::vector<Account> initial{
      {0, 10000000, 0},
      {1, 10000000, 0},
      {2, 5000000, 5},
      {3, 5000000, 5},
  };
  const std::array<AgentAction, 4> canonical{
      AgentAction{0, require_buy(2, 60)},
      AgentAction{1, Action::hold()},
      AgentAction{2, require_sell(2, 55)},
      AgentAction{3, Action::hold()},
  };
  std::array<std::size_t, 4> permutation{0, 1, 2, 3};
  std::vector<Account> expected_accounts;
  marketforge::BatchResult expected_result;
  std::size_t count = 0;
  do {
    auto created = BatchMarket::create(initial, 50);
    MF_CHECK(created);
    auto state = std::move(created).value();
    std::vector<AgentAction> batch;
    for (const auto index : permutation) {
      batch.push_back(canonical[index]);
    }
    const auto result = state.clear(batch);
    MF_CHECK(result);
    if (count == 0) {
      expected_accounts.assign(state.accounts().begin(),
                               state.accounts().end());
      expected_result = result.value();
    } else {
      MF_CHECK(std::equal(state.accounts().begin(), state.accounts().end(),
                          expected_accounts.begin(), expected_accounts.end()));
      MF_CHECK_EQ(result.value(), expected_result);
    }
    ++count;
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  MF_CHECK_EQ(count, 24);
}

MF_TEST(bounded_action_space_exhaustively_conserves_cash_and_shares) {
  const std::vector<Account> initial{
      {0, 2000000, 2},
      {1, 2000000, 2},
      {2, 2000000, 2},
  };
  const std::array<Action, 5> actions{
      Action::hold(),      require_buy(1, 40),  require_buy(1, 60),
      require_sell(1, 40), require_sell(1, 60),
  };
  const auto expected = totals(initial);
  std::size_t cases = 0;
  for (const auto first : actions) {
    for (const auto second : actions) {
      for (const auto third : actions) {
        auto created = BatchMarket::create(initial, 50);
        MF_CHECK(created);
        auto state = std::move(created).value();
        const auto result = state.clear(std::vector<AgentAction>{
            {0, first},
            {1, second},
            {2, third},
        });
        MF_CHECK(result);
        MF_CHECK_EQ(totals(state.accounts()), expected);
        ++cases;
      }
    }
  }
  MF_CHECK_EQ(cases, 125);
}

MF_TEST(canonical_three_batch_trace_decodes_through_dfa_and_matches_exactly) {
  const auto catalog_result = marketforge::SmolLm2MarketActionCatalog::create();
  MF_CHECK(catalog_result);
  const auto& catalog = catalog_result.value();
  auto decode = [&catalog](Action action) {
    const auto tokens = catalog.tokens(action);
    MF_CHECK(tokens);
    auto state = catalog.dfa().root();
    for (const auto token : tokens.value()) {
      const auto next = catalog.dfa().advance(state, token);
      MF_CHECK(next);
      state = next.value();
    }
    const auto decoded = catalog.dfa().decode_terminal(state);
    MF_CHECK(decoded);
    return decoded.value();
  };

  const std::vector<Account> initial{
      {0, 10000000, 0},
      {1, 10000000, 0},
      {2, 5000000, 5},
      {3, 5000000, 5},
  };
  auto created = BatchMarket::create(initial, 50);
  MF_CHECK(created);
  auto state = std::move(created).value();

  auto first = state.clear(std::vector<AgentAction>{
      {0, decode(require_buy(2, 60))},
      {2, decode(require_sell(2, 55))},
  });
  MF_CHECK(first);
  MF_CHECK_EQ(first.value().clearing_tick, 55);
  MF_CHECK_EQ(first.value().matched_quantity, 2);

  auto second = state.clear(std::vector<AgentAction>{
      {1, decode(require_buy(3, 58))},
      {2, decode(require_sell(1, 59))},
      {3, decode(require_sell(3, 58))},
  });
  MF_CHECK(second);
  MF_CHECK_EQ(second.value().clearing_tick, 58);
  MF_CHECK_EQ(second.value().matched_quantity, 3);

  auto third = state.clear(std::vector<AgentAction>{
      {0, decode(require_buy(1, 40))},
      {2, decode(require_sell(1, 70))},
  });
  MF_CHECK(third);
  MF_CHECK(!third.value().traded);

  const std::vector<Account> expected{
      {0, 8900000, 2},
      {1, 8260000, 3},
      {2, 6100000, 3},
      {3, 6740000, 2},
  };
  MF_CHECK(std::equal(state.accounts().begin(), state.accounts().end(),
                      expected.begin(), expected.end()));
  MF_CHECK_EQ(totals(state.accounts()),
              (std::pair<std::int64_t, std::uint64_t>{30000000, 10}));
  MF_CHECK_EQ(state.last_traded_tick(), 58);
}

} // namespace
