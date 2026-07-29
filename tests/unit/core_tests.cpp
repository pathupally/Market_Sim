#include "test_support.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "marketforge/core/checked_math.hpp"
#include "marketforge/core/host_buffer.hpp"
#include "marketforge/core/shape.hpp"
#include "marketforge/core/tensor_view.hpp"

namespace {

using marketforge::ConstTensorView;
using marketforge::DType;
using marketforge::ErrorCode;
using marketforge::HostBuffer;
using marketforge::MemoryKind;
using marketforge::Shape;

MF_TEST(dtype_sizes_are_explicit) {
  MF_CHECK_EQ(marketforge::dtype_size(DType::f32).value(), 4U);
  MF_CHECK_EQ(marketforge::dtype_size(DType::f16).value(), 2U);
  MF_CHECK_EQ(marketforge::dtype_size(DType::bf16).value(), 2U);
  MF_CHECK_EQ(marketforge::dtype_size(DType::i8).value(), 1U);
  MF_CHECK_EQ(marketforge::dtype_size(DType::i32).value(), 4U);
  MF_CHECK_EQ(marketforge::dtype_size(DType::u32).value(), 4U);

  const auto invalid = marketforge::dtype_size(static_cast<DType>(255));
  MF_CHECK(!invalid);
  MF_CHECK_EQ(invalid.status().code, ErrorCode::unsupported_dtype);
}

MF_TEST(checked_arithmetic_reports_overflow) {
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  MF_CHECK_EQ(marketforge::checked_add(20, 22).value(), 42ULL);
  MF_CHECK_EQ(marketforge::checked_multiply(6, 7).value(), 42ULL);
  MF_CHECK(!marketforge::checked_add(maximum, 1));
  MF_CHECK(!marketforge::checked_multiply(maximum, 2));
  MF_CHECK_EQ(marketforge::checked_multiply(0, maximum).value(), 0ULL);
}

MF_TEST(shape_supports_scalars_empty_tensors_and_rank_six) {
  const auto scalar = marketforge::make_shape({});
  MF_CHECK(scalar);
  MF_CHECK_EQ(marketforge::checked_numel(scalar.value()).value(), 1ULL);

  const std::array<std::uint64_t, 3> empty_extents{2, 0, 4};
  const auto empty = marketforge::make_shape(empty_extents);
  MF_CHECK(empty);
  MF_CHECK_EQ(marketforge::checked_numel(empty.value()).value(), 0ULL);

  const std::array<std::uint64_t, 6> rank_six_extents{1, 2, 3, 4, 5, 6};
  const auto rank_six = marketforge::make_shape(rank_six_extents);
  MF_CHECK(rank_six);
  MF_CHECK_EQ(marketforge::checked_numel(rank_six.value()).value(), 720ULL);

  const std::array<std::uint64_t, 7> too_many{1, 1, 1, 1, 1, 1, 1};
  MF_CHECK(!marketforge::make_shape(too_many));
}

MF_TEST(shape_rejects_noncanonical_and_overflowing_values) {
  Shape noncanonical{};
  noncanonical.rank = 1;
  noncanonical.extents[0] = 3;
  noncanonical.extents[5] = 9;
  MF_CHECK_EQ(marketforge::validate_shape(noncanonical).code,
              ErrorCode::invalid_shape);

  const std::array<std::uint64_t, 2> overflow_extents{
      std::numeric_limits<std::uint64_t>::max(), 2};
  const auto overflow_shape = marketforge::make_shape(overflow_extents).value();
  const auto elements = marketforge::checked_numel(overflow_shape);
  MF_CHECK(!elements);
  MF_CHECK_EQ(elements.status().code, ErrorCode::arithmetic_overflow);

  const std::array<std::uint64_t, 1> byte_overflow_extents{
      std::numeric_limits<std::uint64_t>::max() / 2 + 1};
  const auto byte_overflow_shape =
      marketforge::make_shape(byte_overflow_extents).value();
  MF_CHECK(!marketforge::checked_nbytes(byte_overflow_shape, DType::f32));
}

MF_TEST(row_major_offsets_are_checked) {
  const std::array<std::uint64_t, 3> extents{2, 3, 4};
  const auto shape = marketforge::make_shape(extents).value();

  const std::array<std::uint64_t, 3> first{0, 0, 0};
  const std::array<std::uint64_t, 3> middle{1, 1, 2};
  const std::array<std::uint64_t, 3> last{1, 2, 3};
  MF_CHECK_EQ(marketforge::row_major_offset(shape, first).value(), 0ULL);
  MF_CHECK_EQ(marketforge::row_major_offset(shape, middle).value(), 18ULL);
  MF_CHECK_EQ(marketforge::row_major_offset(shape, last).value(), 23ULL);

  const std::array<std::uint64_t, 3> out_of_bounds{0, 3, 0};
  MF_CHECK(!marketforge::row_major_offset(shape, out_of_bounds));
  const std::array<std::uint64_t, 2> wrong_rank{0, 0};
  MF_CHECK(!marketforge::row_major_offset(shape, wrong_rank));
}

MF_TEST(random_small_shapes_match_direct_products) {
  std::uint64_t state = 0x1234'5678'9abc'def0ULL;
  const auto next = [&state]() {
    state = state * 6'364'136'223'846'793'005ULL + 1ULL;
    return state;
  };

  for (std::size_t iteration = 0; iteration < 10'000; ++iteration) {
    const auto rank = static_cast<std::size_t>(next() % 7);
    std::array<std::uint64_t, Shape::max_rank> extents{};
    std::uint64_t expected = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      extents[axis] = next() % 33;
      expected *= extents[axis];
    }
    const auto shape = marketforge::make_shape(
        std::span<const std::uint64_t>(extents.data(), rank));
    MF_CHECK(shape);
    MF_CHECK_EQ(marketforge::checked_numel(shape.value()).value(), expected);
  }
}

MF_TEST(host_buffer_is_aligned_move_only_and_zero_safe) {
  static_assert(!std::is_copy_constructible_v<HostBuffer>);
  static_assert(!std::is_copy_assignable_v<HostBuffer>);
  static_assert(std::is_nothrow_move_constructible_v<HostBuffer>);
  static_assert(std::is_nothrow_move_assignable_v<HostBuffer>);

  for (const std::uint64_t alignment : {16ULL, 64ULL, 256ULL}) {
    auto result = HostBuffer::allocate(1'003, alignment);
    MF_CHECK(result);
    auto buffer = std::move(result).value();
    MF_CHECK_EQ(buffer.bytes().size(), 1'003U);
    const auto address =
        reinterpret_cast<std::uintptr_t>(buffer.bytes().data());
    MF_CHECK_EQ(address % alignment, 0ULL);
    MF_CHECK_EQ(buffer.alignment(), alignment);

    HostBuffer moved = std::move(buffer);
    MF_CHECK_EQ(buffer.bytes().size(), 0U);
    MF_CHECK_EQ(moved.bytes().size(), 1'003U);
  }

  auto empty = HostBuffer::allocate(0, 64);
  MF_CHECK(empty);
  MF_CHECK_EQ(empty.value().bytes().size(), 0U);

  MF_CHECK(!HostBuffer::allocate(32, 0));
  MF_CHECK(!HostBuffer::allocate(32, 3));
}

MF_TEST(tensor_views_validate_shape_dtype_and_pointer) {
  const std::array<std::uint64_t, 2> extents{2, 3};
  const auto shape = marketforge::make_shape(extents).value();

  const ConstTensorView missing_data{
      nullptr,
      shape,
      DType::f32,
      MemoryKind::host,
  };
  MF_CHECK_EQ(marketforge::validate_tensor_view(missing_data).code,
              ErrorCode::invalid_argument);

  const std::array<std::uint64_t, 2> empty_extents{0, 3};
  const ConstTensorView empty{
      nullptr,
      marketforge::make_shape(empty_extents).value(),
      DType::f32,
      MemoryKind::host,
  };
  MF_CHECK(marketforge::validate_tensor_view(empty).ok());

  std::array<float, 6> values{};
  const ConstTensorView valid{
      values.data(),
      shape,
      DType::f32,
      MemoryKind::host,
  };
  MF_CHECK(marketforge::validate_tensor_view(valid).ok());
}

} // namespace
