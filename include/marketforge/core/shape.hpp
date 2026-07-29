#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "marketforge/core/dtype.hpp"

namespace marketforge {

struct Shape {
  static constexpr std::size_t max_rank = 6;

  std::array<std::uint64_t, max_rank> extents{};
  std::uint8_t rank{0};

  friend constexpr bool operator==(const Shape&, const Shape&) = default;
};

[[nodiscard]] Result<Shape>
make_shape(std::span<const std::uint64_t> extents) noexcept;

[[nodiscard]] Status validate_shape(const Shape& shape) noexcept;

[[nodiscard]] Result<std::uint64_t> checked_numel(const Shape& shape) noexcept;

[[nodiscard]] Result<std::uint64_t> checked_nbytes(const Shape& shape,
                                                   DType dtype) noexcept;

[[nodiscard]] Result<std::uint64_t>
row_major_offset(const Shape& shape,
                 std::span<const std::uint64_t> indices) noexcept;

} // namespace marketforge
