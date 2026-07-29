#pragma once

#include <cstdint>

namespace marketforge {

enum class ErrorCode : std::uint16_t {
  ok = 0,
  invalid_argument,
  unsupported_dtype,
  invalid_shape,
  invalid_model,
  arithmetic_overflow,
  invalid_alignment,
  allocation_failed,
  insufficient_memory,
  insufficient_resources,
  io_error,
  invalid_format,
  invalid_utf8,
  invalid_json,
  duplicate_key,
  resource_limit,
  truncated_data,
  invalid_tensor,
  missing_tensor,
  unexpected_tensor,
  hash_mismatch,
};

struct Status {
  ErrorCode code{ErrorCode::ok};
  std::uint32_t detail{0};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ErrorCode::ok;
  }

  [[nodiscard]] static constexpr Status success() noexcept { return {}; }

  [[nodiscard]] static constexpr Status
  failure(ErrorCode error, std::uint32_t error_detail = 0) noexcept {
    return Status{error, error_detail};
  }

  friend constexpr bool operator==(const Status&, const Status&) = default;
};

} // namespace marketforge
