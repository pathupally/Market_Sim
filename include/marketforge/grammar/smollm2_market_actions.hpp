#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "marketforge/grammar/action_dfa.hpp"

namespace marketforge {

class SmolLm2MarketActionCatalog {
public:
  [[nodiscard]] static Result<SmolLm2MarketActionCatalog> create() noexcept;

  [[nodiscard]] const ActionDfa& dfa() const noexcept { return dfa_; }
  [[nodiscard]] std::span<const EncodedAction> entries() const noexcept {
    return entries_;
  }
  [[nodiscard]] Result<std::span<const token_id_t>>
  tokens(Action action) const noexcept;

  [[nodiscard]] static constexpr std::string_view model_id() noexcept {
    return "smollm2-135m";
  }
  [[nodiscard]] static constexpr std::string_view model_revision() noexcept {
    return "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
  }
  [[nodiscard]] static constexpr std::string_view tokenizer_sha256() noexcept {
    return "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c";
  }
  [[nodiscard]] static constexpr std::uint32_t vocabulary_size() noexcept {
    return 49152;
  }

private:
  SmolLm2MarketActionCatalog(ActionDfa dfa,
                             std::vector<EncodedAction> entries) noexcept
      : dfa_(std::move(dfa)), entries_(std::move(entries)) {}

  ActionDfa dfa_;
  std::vector<EncodedAction> entries_;
};

} // namespace marketforge
