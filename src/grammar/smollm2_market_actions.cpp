#include "marketforge/grammar/smollm2_market_actions.hpp"

#include <iterator>
#include <limits>

namespace marketforge {

namespace {

struct GeneratedCatalogEntry {
  ActionKind kind;
  std::uint8_t quantity;
  std::uint8_t tick;
  std::uint32_t token_offset;
  std::uint8_t token_count;
};

#include "generated/smollm2_market_action_v1.inc"

static_assert(std::size(kGeneratedEntries) == 1585);
static_assert(std::size(kGeneratedTokens) < UINT32_MAX);

Result<Action> generated_action(const GeneratedCatalogEntry& entry) noexcept {
  if (entry.kind == ActionKind::hold) {
    if (entry.quantity != 0 || entry.tick != 0) {
      return Result<Action>::failure(
          Status::failure(ErrorCode::invalid_format));
    }
    return Result<Action>::success(Action::hold());
  }
  if (entry.kind == ActionKind::buy_yes) {
    return Action::buy_yes(entry.quantity, entry.tick);
  }
  if (entry.kind == ActionKind::sell_yes) {
    return Action::sell_yes(entry.quantity, entry.tick);
  }
  return Result<Action>::failure(Status::failure(ErrorCode::invalid_format));
}

} // namespace

Result<SmolLm2MarketActionCatalog>
SmolLm2MarketActionCatalog::create() noexcept {
  try {
    std::vector<EncodedAction> entries;
    entries.reserve(std::size(kGeneratedEntries));
    for (const auto& generated : kGeneratedEntries) {
      const auto action = generated_action(generated);
      const auto end = static_cast<std::uint64_t>(generated.token_offset) +
                       static_cast<std::uint64_t>(generated.token_count);
      if (!action || generated.token_count == 0 ||
          end > std::size(kGeneratedTokens)) {
        return Result<SmolLm2MarketActionCatalog>::failure(
            Status::failure(ErrorCode::invalid_format));
      }
      const auto begin = kGeneratedTokens + generated.token_offset;
      entries.push_back(EncodedAction{
          action.value(),
          std::vector<token_id_t>(begin, begin + generated.token_count),
      });
    }
    auto dfa = ActionDfa::build(
        entries, ActionDfaConfig{vocabulary_size(), 12, 4096, 65536, 65536});
    if (!dfa) {
      return Result<SmolLm2MarketActionCatalog>::failure(dfa.status());
    }
    return Result<SmolLm2MarketActionCatalog>::success(
        SmolLm2MarketActionCatalog(std::move(dfa).value(), std::move(entries)));
  } catch (const std::bad_alloc&) {
    return Result<SmolLm2MarketActionCatalog>::failure(
        Status::failure(ErrorCode::allocation_failed));
  }
}

Result<std::span<const token_id_t>>
SmolLm2MarketActionCatalog::tokens(Action action) const noexcept {
  for (const auto& entry : entries_) {
    if (entry.action == action) {
      return Result<std::span<const token_id_t>>::success(entry.tokens);
    }
  }
  return Result<std::span<const token_id_t>>::failure(
      Status::failure(ErrorCode::invalid_argument));
}

} // namespace marketforge
