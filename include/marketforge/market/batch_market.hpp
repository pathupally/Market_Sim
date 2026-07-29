#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "marketforge/core/result.hpp"
#include "marketforge/grammar/action.hpp"

namespace marketforge {

using agent_id_t = std::uint32_t;
using cash_t = std::int64_t;

struct Account {
  agent_id_t id{0};
  cash_t cash{0};
  std::uint32_t yes_shares{0};

  friend constexpr bool operator==(const Account&, const Account&) = default;
};

struct AgentAction {
  agent_id_t agent{0};
  Action action;
};

struct Fill {
  agent_id_t agent{0};
  ActionKind side{ActionKind::hold};
  std::uint8_t quantity{0};

  friend constexpr bool operator==(const Fill&, const Fill&) = default;
};

struct BatchResult {
  bool traded{false};
  std::uint8_t clearing_tick{0};
  std::uint32_t matched_quantity{0};
  std::vector<Fill> fills;

  friend bool operator==(const BatchResult&, const BatchResult&) = default;
};

struct ClearingCandidate {
  std::uint64_t matched_quantity{0};
  std::uint64_t imbalance{0};
  std::uint32_t reference_distance{0};
  std::uint8_t tick{0};
};

[[nodiscard]] bool
better_clearing_candidate(ClearingCandidate candidate,
                          ClearingCandidate incumbent) noexcept;

class BatchMarket {
public:
  static constexpr cash_t cash_units_per_tick = 10000;

  [[nodiscard]] static Result<BatchMarket>
  create(std::span<const Account> accounts,
         std::uint32_t reference_tick) noexcept;

  [[nodiscard]] Result<BatchResult>
  clear(std::span<const AgentAction> actions) noexcept;

  [[nodiscard]] std::span<const Account> accounts() const noexcept {
    return accounts_;
  }
  [[nodiscard]] std::uint8_t last_traded_tick() const noexcept {
    return last_traded_tick_;
  }

private:
  BatchMarket(std::vector<Account> accounts,
              std::uint8_t last_traded_tick) noexcept
      : accounts_(std::move(accounts)), last_traded_tick_(last_traded_tick) {}

  std::vector<Account> accounts_;
  std::uint8_t last_traded_tick_;
};

} // namespace marketforge
