#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"
#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "test_support.hpp"

namespace {

using marketforge::ErrorCode;
using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;
using marketforge::cuda::StreamHandle;

static_assert(!std::is_copy_constructible_v<CudaStream>);
static_assert(!std::is_copy_assignable_v<CudaStream>);
static_assert(std::is_nothrow_move_constructible_v<CudaStream>);
static_assert(std::is_nothrow_move_assignable_v<CudaStream>);
static_assert(std::is_nothrow_destructible_v<CudaStream>);
static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
static_assert(!std::is_copy_assignable_v<DeviceBuffer>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer>);
static_assert(std::is_nothrow_destructible_v<DeviceBuffer>);

template <typename T> [[nodiscard]] T&& indirect_move(T& value) noexcept {
  return static_cast<T&&>(value);
}

MF_TEST(cuda_error_classification_preserves_numeric_detail) {
  const auto runtime =
      marketforge::cuda::detail::runtime_status(cudaErrorInvalidValue);
  MF_CHECK_EQ(runtime.code, ErrorCode::cuda_runtime_failure);
  MF_CHECK_EQ(runtime.detail,
              static_cast<std::uint32_t>(cudaErrorInvalidValue));

  const auto launch =
      marketforge::cuda::detail::launch_status(cudaErrorInvalidConfiguration);
  MF_CHECK_EQ(launch.code, ErrorCode::cuda_backend_failure);
  MF_CHECK_EQ(launch.detail,
              static_cast<std::uint32_t>(cudaErrorInvalidConfiguration));
}

MF_TEST(cuda_empty_states_and_zero_copy_are_safe) {
  CudaStream empty_stream;
  MF_CHECK(!empty_stream.handle().valid());
  MF_CHECK_EQ(empty_stream.synchronize().code, ErrorCode::invalid_argument);

  auto stream_result = CudaStream::create();
  MF_CHECK(stream_result);
  CudaStream stream = std::move(stream_result).value();
  auto zero_result = DeviceBuffer::allocate(0);
  MF_CHECK(zero_result);
  DeviceBuffer zero = std::move(zero_result).value();
  MF_CHECK_EQ(zero.size_bytes(), 0);
  MF_CHECK(!zero.address().valid());
  MF_CHECK(zero.copy_from_host_async(nullptr, 0, 0, stream.handle()).ok());
  MF_CHECK(zero.copy_to_host_async(nullptr, 0, 0, stream.handle()).ok());
  MF_CHECK_EQ(zero.copy_from_host_async(nullptr, 0, 0, {}).code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(zero.copy_from_host_async(nullptr, 0, 1, stream.handle()).code,
              ErrorCode::invalid_argument);
}

MF_TEST(cuda_move_and_copy_validation) {
  auto stream_result = CudaStream::create();
  MF_CHECK(stream_result);
  CudaStream source_stream = std::move(stream_result).value();
  auto live_stream_result = CudaStream::create();
  MF_CHECK(live_stream_result);
  CudaStream stream = std::move(live_stream_result).value();
  stream = std::move(source_stream);
  MF_CHECK(!source_stream.handle().valid());
  stream = indirect_move(stream);
  MF_CHECK(stream.handle().valid());

  auto first_result = DeviceBuffer::allocate(16);
  auto second_result = DeviceBuffer::allocate(32);
  MF_CHECK(first_result);
  MF_CHECK(second_result);
  DeviceBuffer first = std::move(first_result).value();
  DeviceBuffer second = std::move(second_result).value();
  second = std::move(first);
  MF_CHECK(!first.address().valid());
  MF_CHECK_EQ(first.size_bytes(), 0);
  MF_CHECK_EQ(second.size_bytes(), 16);
  second = indirect_move(second);
  MF_CHECK_EQ(second.size_bytes(), 16);

  std::uint32_t value = 0;
  MF_CHECK_EQ(
      second.copy_from_host_async(nullptr, sizeof(value), 0, stream.handle())
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      second.copy_from_host_async(&value, sizeof(value), 0, StreamHandle{})
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(second
                  .copy_from_host_async(
                      &value, 2, std::numeric_limits<std::uint64_t>::max(),
                      stream.handle())
                  .code,
              ErrorCode::arithmetic_overflow);
  MF_CHECK_EQ(
      second.copy_from_host_async(&value, sizeof(value), 15, stream.handle())
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      second.copy_to_host_async(nullptr, sizeof(value), 0, stream.handle())
          .code,
      ErrorCode::invalid_argument);
}

} // namespace
