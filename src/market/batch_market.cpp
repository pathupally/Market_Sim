#include "marketforge/market/batch_market.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <set>
#include <tuple>

namespace marketforge {

namespace {

struct Order {
  std::size_t account_index;
  agent_id_t agent;
  ActionKind side;
  std::uint8_t quantity;
  std::uint8_t limit;
  std::uint8_t fill{0};
};

Result<BatchResult> market_failure(ErrorCode code) {
  return Result<BatchResult>::failure(Status::failure(code));
}

bool checked_add_cash(cash_t value, cash_t delta, cash_t& output) noexcept {
  if ((delta > 0 && value > std::numeric_limits<cash_t>::max() - delta) ||
      (delta < 0 && value < std::numeric_limits<cash_t>::min() - delta)) {
    return false;
  }
  output = value + delta;
  return true;
}

} // namespace

bool better_clearing_candidate(ClearingCandidate candidate,
                               ClearingCandidate incumbent) noexcept {
  return candidate.matched_quantity > incumbent.matched_quantity ||
         (candidate.matched_quantity == incumbent.matched_quantity &&
          (candidate.imbalance < incumbent.imbalance ||
           (candidate.imbalance == incumbent.imbalance &&
            (candidate.reference_distance < incumbent.reference_distance ||
             (candidate.reference_distance == incumbent.reference_distance &&
              candidate.tick < incumbent.tick)))));
}

Result<BatchMarket> BatchMarket::create(std::span<const Account> accounts,
                                        std::uint32_t reference_tick) noexcept {
  try {
    if (accounts.empty() || reference_tick < 1 || reference_tick > 99) {
      return Result<BatchMarket>::failure(
          Status::failure(ErrorCode::invalid_argument));
    }
    std::vector<Account> sorted(accounts.begin(), accounts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const Account& left, const Account& right) {
                return left.id < right.id;
              });
    for (std::size_t index = 0; index < sorted.size(); ++index) {
      if (sorted[index].cash < 0 ||
          (index != 0 && sorted[index - 1].id == sorted[index].id)) {
        return Result<BatchMarket>::failure(
            Status::failure(ErrorCode::invalid_argument));
      }
    }
    return Result<BatchMarket>::success(BatchMarket(
        std::move(sorted), static_cast<std::uint8_t>(reference_tick)));
  } catch (const std::bad_alloc&) {
    return Result<BatchMarket>::failure(
        Status::failure(ErrorCode::allocation_failed));
  }
}

Result<BatchResult>
BatchMarket::clear(std::span<const AgentAction> actions) noexcept {
  try {
    std::set<agent_id_t> submitted;
    std::vector<Order> buys;
    std::vector<Order> sells;
    buys.reserve(actions.size());
    sells.reserve(actions.size());

    for (const auto& submission : actions) {
      if (!submitted.insert(submission.agent).second) {
        return market_failure(ErrorCode::invalid_argument);
      }
      const auto account_iterator =
          std::lower_bound(accounts_.begin(), accounts_.end(), submission.agent,
                           [](const Account& account, agent_id_t id) {
                             return account.id < id;
                           });
      if (account_iterator == accounts_.end() ||
          account_iterator->id != submission.agent) {
        return market_failure(ErrorCode::invalid_argument);
      }
      if (submission.action.kind() == ActionKind::hold) {
        continue;
      }
      const auto account_index =
          static_cast<std::size_t>(account_iterator - accounts_.begin());
      const auto quantity = submission.action.quantity();
      const auto limit = submission.action.price_tick();
      const cash_t limit_notional = static_cast<cash_t>(quantity) *
                                    static_cast<cash_t>(limit) *
                                    cash_units_per_tick;
      Order order{account_index, submission.agent, submission.action.kind(),
                  quantity, limit};
      if (submission.action.kind() == ActionKind::buy_yes) {
        if (account_iterator->cash < limit_notional) {
          return market_failure(ErrorCode::insufficient_resources);
        }
        buys.push_back(order);
      } else {
        if (account_iterator->yes_shares < quantity) {
          return market_failure(ErrorCode::insufficient_resources);
        }
        sells.push_back(order);
      }
    }

    std::uint64_t best_matched = 0;
    std::uint64_t best_imbalance = std::numeric_limits<std::uint64_t>::max();
    std::uint32_t best_distance = std::numeric_limits<std::uint32_t>::max();
    std::uint8_t clearing_tick = 0;
    for (std::uint32_t tick = 1; tick <= 99; ++tick) {
      std::uint64_t demand = 0;
      std::uint64_t supply = 0;
      for (const auto& order : buys) {
        if (order.limit >= tick) {
          if (demand >
              std::numeric_limits<std::uint64_t>::max() - order.quantity) {
            return market_failure(ErrorCode::arithmetic_overflow);
          }
          demand += order.quantity;
        }
      }
      for (const auto& order : sells) {
        if (order.limit <= tick) {
          if (supply >
              std::numeric_limits<std::uint64_t>::max() - order.quantity) {
            return market_failure(ErrorCode::arithmetic_overflow);
          }
          supply += order.quantity;
        }
      }
      const auto matched = std::min(demand, supply);
      const auto imbalance =
          demand >= supply ? demand - supply : supply - demand;
      const auto distance = static_cast<std::uint32_t>(std::abs(
          static_cast<int>(tick) - static_cast<int>(last_traded_tick_)));
      const bool better =
          clearing_tick == 0 ||
          better_clearing_candidate(
              ClearingCandidate{matched, imbalance, distance,
                                static_cast<std::uint8_t>(tick)},
              ClearingCandidate{best_matched, best_imbalance, best_distance,
                                clearing_tick});
      if (better) {
        best_matched = matched;
        best_imbalance = imbalance;
        best_distance = distance;
        clearing_tick = static_cast<std::uint8_t>(tick);
      }
    }

    if (best_matched == 0) {
      return Result<BatchResult>::success(BatchResult{});
    }
    if (best_matched > std::numeric_limits<std::uint32_t>::max()) {
      return market_failure(ErrorCode::arithmetic_overflow);
    }

    std::sort(buys.begin(), buys.end(),
              [](const Order& left, const Order& right) {
                return std::tuple{-left.limit, left.agent} <
                       std::tuple{-right.limit, right.agent};
              });
    std::sort(sells.begin(), sells.end(),
              [](const Order& left, const Order& right) {
                return std::tuple{left.limit, left.agent} <
                       std::tuple{right.limit, right.agent};
              });

    auto allocate = [clearing_tick, best_matched](std::vector<Order>& orders,
                                                  bool buys_side) {
      std::uint64_t remaining = best_matched;
      for (auto& order : orders) {
        const bool eligible = buys_side ? order.limit >= clearing_tick
                                        : order.limit <= clearing_tick;
        if (!eligible || remaining == 0) {
          continue;
        }
        order.fill = static_cast<std::uint8_t>(
            std::min<std::uint64_t>(order.quantity, remaining));
        remaining -= order.fill;
      }
      return remaining == 0;
    };
    if (!allocate(buys, true) || !allocate(sells, false)) {
      return market_failure(ErrorCode::invalid_format);
    }

    auto updated = accounts_;
    std::vector<Fill> fills;
    fills.reserve(buys.size() + sells.size());
    auto apply = [&](const Order& order) -> bool {
      if (order.fill == 0) {
        return true;
      }
      const cash_t notional = static_cast<cash_t>(order.fill) *
                              static_cast<cash_t>(clearing_tick) *
                              cash_units_per_tick;
      auto& account = updated[order.account_index];
      cash_t cash = 0;
      if (order.side == ActionKind::buy_yes) {
        if (!checked_add_cash(account.cash, -notional, cash) ||
            account.yes_shares >
                std::numeric_limits<std::uint32_t>::max() - order.fill) {
          return false;
        }
        account.cash = cash;
        account.yes_shares += order.fill;
      } else {
        if (!checked_add_cash(account.cash, notional, cash) ||
            account.yes_shares < order.fill) {
          return false;
        }
        account.cash = cash;
        account.yes_shares -= order.fill;
      }
      fills.push_back(Fill{order.agent, order.side, order.fill});
      return true;
    };
    for (const auto& order : buys) {
      if (!apply(order)) {
        return market_failure(ErrorCode::arithmetic_overflow);
      }
    }
    for (const auto& order : sells) {
      if (!apply(order)) {
        return market_failure(ErrorCode::arithmetic_overflow);
      }
    }
    std::sort(fills.begin(), fills.end(),
              [](const Fill& left, const Fill& right) {
                return left.agent < right.agent;
              });

    accounts_ = std::move(updated);
    last_traded_tick_ = clearing_tick;
    return Result<BatchResult>::success(BatchResult{
        true, clearing_tick, static_cast<std::uint32_t>(best_matched),
        std::move(fills)});
  } catch (const std::bad_alloc&) {
    return market_failure(ErrorCode::allocation_failed);
  }
}

} // namespace marketforge
