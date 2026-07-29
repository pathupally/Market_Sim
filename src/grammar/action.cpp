#include "marketforge/grammar/action.hpp"

namespace marketforge {

Result<Action> Action::buy_yes(std::uint32_t quantity,
                               std::uint32_t price_tick) noexcept {
  if (quantity < 1 || quantity > 8 || price_tick < 1 || price_tick > 99) {
    return Result<Action>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return Result<Action>::success(Action(ActionKind::buy_yes,
                                        static_cast<std::uint8_t>(quantity),
                                        static_cast<std::uint8_t>(price_tick)));
}

Result<Action> Action::sell_yes(std::uint32_t quantity,
                                std::uint32_t price_tick) noexcept {
  if (quantity < 1 || quantity > 8 || price_tick < 1 || price_tick > 99) {
    return Result<Action>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return Result<Action>::success(Action(ActionKind::sell_yes,
                                        static_cast<std::uint8_t>(quantity),
                                        static_cast<std::uint8_t>(price_tick)));
}

} // namespace marketforge
