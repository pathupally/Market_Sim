#pragma once

#include <cstddef>
#include <type_traits>

#include "marketforge/core/shape.hpp"

namespace marketforge {

enum class MemoryKind : std::uint8_t {
  host,
  device,
};

struct TensorView {
  void* data{nullptr};
  Shape shape{};
  DType dtype{DType::f32};
  MemoryKind memory{MemoryKind::host};
};

struct ConstTensorView {
  const void* data{nullptr};
  Shape shape{};
  DType dtype{DType::f32};
  MemoryKind memory{MemoryKind::host};
};

static_assert(std::is_standard_layout_v<TensorView>);
static_assert(std::is_trivially_copyable_v<TensorView>);
static_assert(std::is_standard_layout_v<ConstTensorView>);
static_assert(std::is_trivially_copyable_v<ConstTensorView>);

[[nodiscard]] inline Status
validate_tensor_view(const ConstTensorView& view) noexcept {
  const auto bytes = checked_nbytes(view.shape, view.dtype);
  if (!bytes) {
    return bytes.status();
  }
  if (bytes.value() != 0 && view.data == nullptr) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  return Status::success();
}

[[nodiscard]] inline Status
validate_tensor_view(const TensorView& view) noexcept {
  return validate_tensor_view(ConstTensorView{
      view.data,
      view.shape,
      view.dtype,
      view.memory,
  });
}

} // namespace marketforge
