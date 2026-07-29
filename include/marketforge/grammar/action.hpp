#pragma once

#include <cstdint>

#include "marketforge/core/result.hpp"

namespace marketforge {

enum class ActionKind : std::uint8_t {
  hold = 0,
  buy_yes,
  sell_yes,
};

class Action {
public:
  [[nodiscard]] static constexpr Action hold() noexcept {
    return Action(ActionKind::hold, 0, 0);
  }

  [[nodiscard]] static Result<Action>
  buy_yes(std::uint32_t quantity, std::uint32_t price_tick) noexcept;

  [[nodiscard]] static Result<Action>
  sell_yes(std::uint32_t quantity, std::uint32_t price_tick) noexcept;

  [[nodiscard]] constexpr ActionKind kind() const noexcept { return kind_; }
  [[nodiscard]] constexpr std::uint8_t quantity() const noexcept {
    return quantity_;
  }
  [[nodiscard]] constexpr std::uint8_t price_tick() const noexcept {
    return price_tick_;
  }

  friend constexpr bool operator==(const Action&, const Action&) = default;

private:
  constexpr Action(ActionKind kind, std::uint8_t quantity,
                   std::uint8_t price_tick) noexcept
      : kind_(kind), quantity_(quantity), price_tick_(price_tick) {}

  ActionKind kind_;
  std::uint8_t quantity_;
  std::uint8_t price_tick_;
};

} // namespace marketforge
