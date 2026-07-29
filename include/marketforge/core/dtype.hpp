#pragma once

#include <cstdint>

#include "marketforge/core/result.hpp"

namespace marketforge {

enum class DType : std::uint8_t {
  f32,
  f16,
  bf16,
  i8,
  i32,
  u32,
};

[[nodiscard]] inline Result<std::uint32_t> dtype_size(DType dtype) noexcept {
  switch (dtype) {
  case DType::f32:
  case DType::i32:
  case DType::u32:
    return Result<std::uint32_t>::success(4);
  case DType::f16:
  case DType::bf16:
    return Result<std::uint32_t>::success(2);
  case DType::i8:
    return Result<std::uint32_t>::success(1);
  }
  return Result<std::uint32_t>::failure(
      Status::failure(ErrorCode::unsupported_dtype));
}

} // namespace marketforge
