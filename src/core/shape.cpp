#include "marketforge/core/shape.hpp"

#include <algorithm>
#include <limits>

#include "marketforge/core/checked_math.hpp"

namespace marketforge {

Result<Shape>
make_shape(const std::span<const std::uint64_t> extents) noexcept {
  if (extents.size() > Shape::max_rank) {
    return Result<Shape>::failure(Status::failure(ErrorCode::invalid_shape));
  }

  Shape shape{};
  shape.rank = static_cast<std::uint8_t>(extents.size());
  std::copy(extents.begin(), extents.end(), shape.extents.begin());
  return Result<Shape>::success(shape);
}

Status validate_shape(const Shape& shape) noexcept {
  const auto rank = static_cast<std::size_t>(shape.rank);
  if (rank > Shape::max_rank) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  for (std::size_t index = rank; index < Shape::max_rank; ++index) {
    if (shape.extents[index] != 0) {
      return Status::failure(ErrorCode::invalid_shape,
                             static_cast<std::uint32_t>(index));
    }
  }
  return Status::success();
}

Result<std::uint64_t> checked_numel(const Shape& shape) noexcept {
  const auto validation = validate_shape(shape);
  if (!validation.ok()) {
    return Result<std::uint64_t>::failure(validation);
  }

  std::uint64_t elements = 1;
  const auto rank = static_cast<std::size_t>(shape.rank);
  for (std::size_t index = 0; index < rank; ++index) {
    const auto product = checked_multiply(elements, shape.extents[index]);
    if (!product) {
      return product;
    }
    elements = product.value();
  }
  return Result<std::uint64_t>::success(elements);
}

Result<std::uint64_t> checked_nbytes(const Shape& shape,
                                     const DType dtype) noexcept {
  const auto elements = checked_numel(shape);
  if (!elements) {
    return Result<std::uint64_t>::failure(elements.status());
  }
  const auto bytes_per_element = dtype_size(dtype);
  if (!bytes_per_element) {
    return Result<std::uint64_t>::failure(bytes_per_element.status());
  }
  return checked_multiply(elements.value(), bytes_per_element.value());
}

Result<std::uint64_t>
row_major_offset(const Shape& shape,
                 const std::span<const std::uint64_t> indices) noexcept {
  const auto validation = validate_shape(shape);
  if (!validation.ok() ||
      indices.size() != static_cast<std::size_t>(shape.rank)) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::invalid_shape));
  }

  std::uint64_t offset = 0;
  const auto rank = static_cast<std::size_t>(shape.rank);
  for (std::size_t axis = 0; axis < rank; ++axis) {
    if (indices[axis] >= shape.extents[axis]) {
      return Result<std::uint64_t>::failure(Status::failure(
          ErrorCode::invalid_argument, static_cast<std::uint32_t>(axis)));
    }
    const auto scaled = checked_multiply(offset, shape.extents[axis]);
    if (!scaled) {
      return scaled;
    }
    const auto next = checked_add(scaled.value(), indices[axis]);
    if (!next) {
      return next;
    }
    offset = next.value();
  }
  return Result<std::uint64_t>::success(offset);
}

} // namespace marketforge
