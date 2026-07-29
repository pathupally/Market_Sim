#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#include "marketforge/grammar/smollm2_market_actions.hpp"
#include "marketforge/market/batch_market.hpp"

namespace {

using marketforge::Action;
using marketforge::AgentAction;
using marketforge::BatchMarket;
using marketforge::SmolLm2MarketActionCatalog;

marketforge::Result<Action> decode(const SmolLm2MarketActionCatalog& catalog,
                                   Action action) {
  const auto tokens = catalog.tokens(action);
  if (!tokens) {
    return marketforge::Result<Action>::failure(tokens.status());
  }
  auto state = catalog.dfa().root();
  for (const auto token : tokens.value()) {
    const auto next = catalog.dfa().advance(state, token);
    if (!next) {
      return marketforge::Result<Action>::failure(next.status());
    }
    state = next.value();
  }
  return catalog.dfa().decode_terminal(state);
}

bool append(std::vector<AgentAction>& batch,
            const SmolLm2MarketActionCatalog& catalog,
            marketforge::agent_id_t agent, Action action) {
  const auto decoded = decode(catalog, action);
  if (!decoded) {
    return false;
  }
  batch.push_back(AgentAction{agent, decoded.value()});
  return true;
}

} // namespace

int main() {
  const auto catalog_result = SmolLm2MarketActionCatalog::create();
  if (!catalog_result) {
    std::cerr << "catalog construction failed\n";
    return 1;
  }
  const auto& catalog = catalog_result.value();
  const std::vector<marketforge::Account> accounts{
      {0, 10000000, 0},
      {1, 10000000, 0},
      {2, 5000000, 5},
      {3, 5000000, 5},
  };
  auto market_result = BatchMarket::create(accounts, 50);
  if (!market_result) {
    std::cerr << "market construction failed\n";
    return 1;
  }
  auto market = std::move(market_result).value();

  std::vector<std::vector<AgentAction>> batches(3);
  const auto buy_2_60 = Action::buy_yes(2, 60);
  const auto sell_2_55 = Action::sell_yes(2, 55);
  const auto buy_3_58 = Action::buy_yes(3, 58);
  const auto sell_1_59 = Action::sell_yes(1, 59);
  const auto sell_3_58 = Action::sell_yes(3, 58);
  const auto buy_1_40 = Action::buy_yes(1, 40);
  const auto sell_1_70 = Action::sell_yes(1, 70);
  if (!buy_2_60 || !sell_2_55 || !buy_3_58 || !sell_1_59 || !sell_3_58 ||
      !buy_1_40 || !sell_1_70 ||
      !append(batches[0], catalog, 0, buy_2_60.value()) ||
      !append(batches[0], catalog, 2, sell_2_55.value()) ||
      !append(batches[1], catalog, 1, buy_3_58.value()) ||
      !append(batches[1], catalog, 2, sell_1_59.value()) ||
      !append(batches[1], catalog, 3, sell_3_58.value()) ||
      !append(batches[2], catalog, 0, buy_1_40.value()) ||
      !append(batches[2], catalog, 2, sell_1_70.value())) {
    std::cerr << "DFA action decoding failed\n";
    return 1;
  }

  for (std::size_t index = 0; index < batches.size(); ++index) {
    const auto result = market.clear(batches[index]);
    if (!result) {
      std::cerr << "batch " << index + 1 << " failed\n";
      return 1;
    }
    std::cout << "batch " << index + 1
              << " traded=" << (result.value().traded ? 1 : 0)
              << " tick=" << static_cast<int>(result.value().clearing_tick)
              << " quantity=" << result.value().matched_quantity << '\n';
  }

  std::int64_t total_cash = 0;
  std::uint64_t total_shares = 0;
  for (const auto& account : market.accounts()) {
    std::cout << "agent " << account.id << " cash=" << account.cash
              << " shares=" << account.yes_shares << '\n';
    total_cash += account.cash;
    total_shares += account.yes_shares;
  }
  std::cout << "total cash=" << total_cash << " shares=" << total_shares
            << '\n';
  std::cout << "last tick=" << static_cast<int>(market.last_traded_tick())
            << '\n';
  return 0;
}
