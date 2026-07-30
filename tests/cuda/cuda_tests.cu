#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "cublas_internal.hpp"
#include "cuda_internal.hpp"
#include "marketforge/core/dtype.hpp"
#include "marketforge/core/shape.hpp"
#include "marketforge/core/status.hpp"
#include "marketforge/core/tensor_view.hpp"
#include "marketforge/cpu/operators.hpp"
#include "marketforge/cuda/cublas_handle.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/linear.hpp"
#include "marketforge/cuda/rms_norm.hpp"
#include "test_support.hpp"

namespace {

using marketforge::ErrorCode;
using marketforge::MemoryKind;
using marketforge::TensorView;
using marketforge::cuda::CublasHandle;
using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;
using marketforge::cuda::StreamHandle;

static_assert(!std::is_copy_constructible_v<CublasHandle>);
static_assert(!std::is_copy_assignable_v<CublasHandle>);
static_assert(std::is_nothrow_move_constructible_v<CublasHandle>);
static_assert(std::is_nothrow_move_assignable_v<CublasHandle>);
static_assert(std::is_nothrow_destructible_v<CublasHandle>);
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

void run_rms_norm_parity_case(const std::uint64_t rows,
                              const std::uint64_t hidden_size,
                              const bool in_place) {
  const auto elements = static_cast<std::size_t>(rows * hidden_size);
  std::vector<float> input(elements);
  std::vector<float> weight(static_cast<std::size_t>(hidden_size));
  for (std::size_t index = 0; index < input.size(); ++index) {
    const auto centered = static_cast<std::int32_t>(index % 37) - 18;
    input[index] = static_cast<float>(centered) * 0.0625F;
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] = 0.75F + static_cast<float>(index % 17) * 0.03125F;
  }
  std::vector<float> expected(elements);
  std::vector<float> observed(elements);
  const std::array<std::uint64_t, 2> input_extents{rows, hidden_size};
  const std::array<std::uint64_t, 1> weight_extents{hidden_size};
  const auto input_shape = marketforge::make_shape(input_extents);
  const auto weight_shape = marketforge::make_shape(weight_extents);
  MF_CHECK(input_shape);
  MF_CHECK(weight_shape);
  MF_CHECK(marketforge::rms_norm_f32(
               {
                   input.data(),
                   input_shape.value(),
                   marketforge::DType::f32,
                   MemoryKind::host,
               },
               {
                   weight.data(),
                   weight_shape.value(),
                   marketforge::DType::f32,
                   MemoryKind::host,
               },
               1.0e-5F,
               TensorView{
                   expected.data(),
                   input_shape.value(),
                   marketforge::DType::f32,
                   MemoryKind::host,
               })
               .ok());

  auto stream_result = CudaStream::create();
  auto input_result = DeviceBuffer::allocate(elements * sizeof(float));
  auto weight_result = DeviceBuffer::allocate(weight.size() * sizeof(float));
  auto output_result =
      DeviceBuffer::allocate(in_place ? 0 : elements * sizeof(float));
  MF_CHECK(stream_result);
  MF_CHECK(input_result);
  MF_CHECK(weight_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer input_device = std::move(input_result).value();
  DeviceBuffer weight_device = std::move(weight_result).value();
  DeviceBuffer output_device =
      in_place ? DeviceBuffer{} : std::move(output_result).value();
  MF_CHECK(input_device
               .copy_from_host_async(input.data(), elements * sizeof(float), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(weight_device
               .copy_from_host_async(weight.data(),
                                     weight.size() * sizeof(float), 0,
                                     stream.handle())
               .ok());
  DeviceBuffer& destination = in_place ? input_device : output_device;
  MF_CHECK(marketforge::cuda::rms_norm_f32(input_device, weight_device,
                                           destination, rows, hidden_size,
                                           1.0e-5F, stream.handle())
               .ok());
  MF_CHECK(destination
               .copy_to_host_async(observed.data(), elements * sizeof(float), 0,
                                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t index = 0; index < elements; ++index) {
    MF_CHECK_NEAR(observed[index], expected[index], 3.0e-5F);
  }
}

void run_linear_parity_case(const std::uint64_t rows,
                            const std::uint64_t input_features,
                            const std::uint64_t output_features) {
  const auto input_elements =
      static_cast<std::size_t>(rows * input_features);
  const auto weight_elements =
      static_cast<std::size_t>(output_features * input_features);
  const auto output_elements =
      static_cast<std::size_t>(rows * output_features);
  std::vector<__half> input(input_elements);
  std::vector<__half> weight(weight_elements);
  std::vector<__half> observed(output_elements);
  std::vector<float> expected(output_elements, 0.0F);

  for (std::size_t index = 0; index < input.size(); ++index) {
    const auto value =
        static_cast<float>(static_cast<std::int32_t>(index % 17) - 8) /
        32.0F;
    input[index] = __float2half_rn(value);
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    const auto value =
        static_cast<float>(static_cast<std::int32_t>(index % 13) - 6) /
        64.0F;
    weight[index] = __float2half_rn(value);
  }
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t output_feature = 0;
         output_feature < output_features; ++output_feature) {
      float sum = 0.0F;
      for (std::uint64_t input_feature = 0;
           input_feature < input_features; ++input_feature) {
        const auto input_index =
            static_cast<std::size_t>(row * input_features + input_feature);
        const auto weight_index = static_cast<std::size_t>(
            output_feature * input_features + input_feature);
        sum = std::fma(__half2float(input[input_index]),
                       __half2float(weight[weight_index]), sum);
      }
      expected[static_cast<std::size_t>(row * output_features +
                                        output_feature)] = sum;
    }
  }

  auto stream_result = CudaStream::create();
  auto handle_result = CublasHandle::create();
  auto input_result =
      DeviceBuffer::allocate(input_elements * sizeof(__half));
  auto weight_result =
      DeviceBuffer::allocate(weight_elements * sizeof(__half));
  auto output_result =
      DeviceBuffer::allocate(output_elements * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(handle_result);
  MF_CHECK(input_result);
  MF_CHECK(weight_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  CublasHandle handle = std::move(handle_result).value();
  DeviceBuffer input_device = std::move(input_result).value();
  DeviceBuffer weight_device = std::move(weight_result).value();
  DeviceBuffer output_device = std::move(output_result).value();

  MF_CHECK(input_device
               .copy_from_host_async(input.data(),
                                     input_elements * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(weight_device
               .copy_from_host_async(weight.data(),
                                     weight_elements * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::linear_f16(
               input_device, weight_device, output_device, rows,
               input_features, output_features, handle, stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(observed.data(),
                                   output_elements * sizeof(__half), 0,
                                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t index = 0; index < observed.size(); ++index) {
    MF_CHECK_NEAR(__half2float(observed[index]), expected[index], 0.015F);
  }
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

  const auto cublas =
      marketforge::cuda::detail::cublas_status(CUBLAS_STATUS_INVALID_VALUE);
  MF_CHECK_EQ(cublas.code, ErrorCode::cuda_backend_failure);
  MF_CHECK_EQ(cublas.detail,
              static_cast<std::uint32_t>(CUBLAS_STATUS_INVALID_VALUE));
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

  auto first_handle_result = CublasHandle::create();
  auto second_handle_result = CublasHandle::create();
  MF_CHECK(first_handle_result);
  MF_CHECK(second_handle_result);
  CublasHandle first_handle = std::move(first_handle_result).value();
  CublasHandle second_handle = std::move(second_handle_result).value();
  second_handle = std::move(first_handle);
  MF_CHECK(!first_handle.handle().valid());
  MF_CHECK(second_handle.handle().valid());
  second_handle = indirect_move(second_handle);
  MF_CHECK(second_handle.handle().valid());
}

MF_TEST(cuda_rms_norm_matches_cpu_for_smollm2_and_boundary_shapes) {
  run_rms_norm_parity_case(2, 1, false);
  run_rms_norm_parity_case(3, 255, false);
  run_rms_norm_parity_case(3, 256, false);
  run_rms_norm_parity_case(3, 257, false);
  run_rms_norm_parity_case(7, 576, false);
  run_rms_norm_parity_case(2, 1024, true);
}

MF_TEST(cuda_rms_norm_rejects_invalid_metadata_before_launch) {
  auto stream_result = CudaStream::create();
  auto input_result = DeviceBuffer::allocate(8 * sizeof(float));
  auto weight_result = DeviceBuffer::allocate(4 * sizeof(float));
  auto output_result = DeviceBuffer::allocate(8 * sizeof(float));
  MF_CHECK(stream_result);
  MF_CHECK(input_result);
  MF_CHECK(weight_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer input = std::move(input_result).value();
  DeviceBuffer weight = std::move(weight_result).value();
  DeviceBuffer output = std::move(output_result).value();

  MF_CHECK_EQ(marketforge::cuda::rms_norm_f32(input, weight, output, 0, 4,
                                              1.0e-5F, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::rms_norm_f32(
                  input, weight, output, 2, 4,
                  std::numeric_limits<float>::quiet_NaN(), stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::rms_norm_f32(input, weight, output, 2, 4, 1.0e-5F, {})
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::rms_norm_f32(input, weight, output, 1, 4,
                                              1.0e-5F, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
}

MF_TEST(cuda_linear_f16_matches_reference_for_tiny_and_smollm2_shapes) {
  run_linear_parity_case(2, 3, 2);
  run_linear_parity_case(3, 576, 1'536);
}

MF_TEST(cuda_linear_f16_rejects_invalid_metadata_before_cublas) {
  auto stream_result = CudaStream::create();
  auto handle_result = CublasHandle::create();
  auto input_result = DeviceBuffer::allocate(6 * sizeof(__half));
  auto weight_result = DeviceBuffer::allocate(6 * sizeof(__half));
  auto output_result = DeviceBuffer::allocate(4 * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(handle_result);
  MF_CHECK(input_result);
  MF_CHECK(weight_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  CublasHandle handle = std::move(handle_result).value();
  DeviceBuffer input = std::move(input_result).value();
  DeviceBuffer weight = std::move(weight_result).value();
  DeviceBuffer output = std::move(output_result).value();

  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, output, 0, 3, 2, handle, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, output, 2, 3, 2, handle, {})
                  .code,
              ErrorCode::invalid_argument);
  CublasHandle empty_handle;
  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, output, 2, 3, 2, empty_handle,
                  stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, output, 1, 3, 2, handle, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, input, 2, 3, 2, handle, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::linear_f16(
          input, weight, output, std::numeric_limits<std::uint64_t>::max(), 2,
          2, handle, stream.handle())
          .code,
      ErrorCode::arithmetic_overflow);
}

} // namespace
