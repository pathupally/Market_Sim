#pragma once

#include <cstdint>
#include <limits>

#include "marketforge/core/result.hpp"

namespace marketforge {

[[nodiscard]] inline Result<std::uint64_t>
checked_add(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  return Result<std::uint64_t>::success(lhs + rhs);
}

[[nodiscard]] inline Result<std::uint64_t>
checked_multiply(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  return Result<std::uint64_t>::success(lhs * rhs);
}

} // namespace marketforge
