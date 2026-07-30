#include "marketforge/cuda/rms_norm.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;

[[nodiscard]] bool checked_f32_bytes(const std::uint64_t elements,
                                     std::uint64_t& bytes) noexcept {
  if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
    return false;
  }
  bytes = elements * sizeof(float);
  return true;
}

[[nodiscard]] bool checked_elements(const std::uint64_t rows,
                                    const std::uint64_t hidden_size,
                                    std::uint64_t& elements) noexcept {
  if (rows > std::numeric_limits<std::uint64_t>::max() / hidden_size) {
    return false;
  }
  elements = rows * hidden_size;
  return true;
}

__global__ void rms_norm_kernel(const float* input, const float* weight,
                                float* output, const std::uint64_t hidden_size,
                                const float epsilon) {
  extern __shared__ float reductions[];
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  const auto row_offset = row * hidden_size;

  float sum_squares = 0.0F;
  for (std::uint64_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const float value = input[row_offset + column];
    sum_squares = fmaf(value, value, sum_squares);
  }
  reductions[threadIdx.x] = sum_squares;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(reductions[0] / static_cast<float>(hidden_size) + epsilon);
  for (std::uint64_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    output[row_offset + column] =
        input[row_offset + column] * inverse_rms * weight[column];
  }
}

__global__ void rms_norm_f16_kernel(
    const __half* input, const __half* weight, __half* output,
    const std::uint64_t hidden_size, const float epsilon) {
  extern __shared__ float reductions[];
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  const auto row_offset = row * hidden_size;

  float sum_squares = 0.0F;
  for (std::uint64_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    const float value = __half2float(input[row_offset + column]);
    sum_squares = fmaf(value, value, sum_squares);
  }
  reductions[threadIdx.x] = sum_squares;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(reductions[0] / static_cast<float>(hidden_size) + epsilon);
  for (std::uint64_t column = threadIdx.x; column < hidden_size;
       column += blockDim.x) {
    output[row_offset + column] = __float2half_rn(
        __half2float(input[row_offset + column]) * inverse_rms *
        __half2float(weight[column]));
  }
}

} // namespace

Status rms_norm_f32(const DeviceBuffer& input, const DeviceBuffer& weight,
                    DeviceBuffer& output, const std::uint64_t rows,
                    const std::uint64_t hidden_size, const float epsilon,
                    const StreamHandle stream) noexcept {
  if (rows == 0 || hidden_size == 0 || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  std::uint64_t elements = 0;
  std::uint64_t tensor_bytes = 0;
  std::uint64_t weight_bytes = 0;
  if (!checked_elements(rows, hidden_size, elements) ||
      !checked_f32_bytes(elements, tensor_bytes) ||
      !checked_f32_bytes(hidden_size, weight_bytes)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  if (input.size_bytes() != tensor_bytes ||
      output.size_bytes() != tensor_bytes ||
      weight.size_bytes() != weight_bytes || !input.address().valid() ||
      !output.address().valid() || !weight.address().valid() ||
      weight.address() == input.address() ||
      weight.address() == output.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Status::failure(ErrorCode::resource_limit);
  }

  rms_norm_kernel<<<static_cast<unsigned int>(rows), threads_per_block,
                    threads_per_block * sizeof(float),
                    detail::native_stream(stream)>>>(
      static_cast<const float*>(detail::native_address(input.address())),
      static_cast<const float*>(detail::native_address(weight.address())),
      static_cast<float*>(detail::native_address(output.address())),
      hidden_size, epsilon);
  return detail::launch_status(cudaGetLastError());
}

Status rms_norm_f16(const DeviceBuffer& input, const DeviceBuffer& weight,
                    DeviceBuffer& output, const std::uint64_t rows,
                    const std::uint64_t hidden_size, const float epsilon,
                    const StreamHandle stream) noexcept {
  if (rows == 0 || hidden_size == 0 || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (rows > std::numeric_limits<std::uint64_t>::max() / hidden_size) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto elements = rows * hidden_size;
  if (elements >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half) ||
      hidden_size >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto tensor_bytes = elements * sizeof(__half);
  const auto weight_bytes = hidden_size * sizeof(__half);
  if (input.size_bytes() != tensor_bytes ||
      output.size_bytes() != tensor_bytes ||
      weight.size_bytes() != weight_bytes || !input.address().valid() ||
      !output.address().valid() || !weight.address().valid() ||
      weight.address() == input.address() ||
      weight.address() == output.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Status::failure(ErrorCode::resource_limit);
  }
  rms_norm_f16_kernel<<<static_cast<unsigned int>(rows), threads_per_block,
                        threads_per_block * sizeof(float),
                        detail::native_stream(stream)>>>(
      static_cast<const __half*>(detail::native_address(input.address())),
      static_cast<const __half*>(detail::native_address(weight.address())),
      static_cast<__half*>(detail::native_address(output.address())),
      hidden_size, epsilon);
  return detail::launch_status(cudaGetLastError());
}

} // namespace marketforge::cuda
