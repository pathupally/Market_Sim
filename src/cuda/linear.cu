#include "marketforge/cuda/linear.hpp"

#include <cstdint>
#include <limits>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "cublas_internal.hpp"
#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

[[nodiscard]] Status checked_f16_matrix_bytes(
    const std::uint64_t rows, const std::uint64_t columns,
    std::uint64_t& bytes) noexcept {
  if (rows == 0 || columns == 0) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (rows > std::numeric_limits<std::uint64_t>::max() / columns) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto elements = rows * columns;
  constexpr std::uint64_t f16_bytes = 2;
  if (elements > std::numeric_limits<std::uint64_t>::max() / f16_bytes) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  bytes = elements * f16_bytes;
  return Status::success();
}

[[nodiscard]] bool fits_cublas_dimension(const std::uint64_t value) noexcept {
  return value <=
         static_cast<std::uint64_t>(std::numeric_limits<int>::max());
}

} // namespace

Status linear_f16(const DeviceBuffer& input, const DeviceBuffer& weight,
                  DeviceBuffer& output, const std::uint64_t rows,
                  const std::uint64_t input_features,
                  const std::uint64_t output_features, CublasHandle& handle,
                  const StreamHandle stream) noexcept {
  if (!handle.handle().valid() || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  std::uint64_t input_bytes = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t output_bytes = 0;
  const auto input_status =
      checked_f16_matrix_bytes(rows, input_features, input_bytes);
  if (!input_status.ok()) {
    return input_status;
  }
  const auto weight_status = checked_f16_matrix_bytes(
      output_features, input_features, weight_bytes);
  if (!weight_status.ok()) {
    return weight_status;
  }
  const auto output_status =
      checked_f16_matrix_bytes(rows, output_features, output_bytes);
  if (!output_status.ok()) {
    return output_status;
  }

  if (!fits_cublas_dimension(rows) ||
      !fits_cublas_dimension(input_features) ||
      !fits_cublas_dimension(output_features)) {
    return Status::failure(ErrorCode::resource_limit);
  }
  if (input.size_bytes() != input_bytes ||
      weight.size_bytes() != weight_bytes ||
      output.size_bytes() != output_bytes || !input.address().valid() ||
      !weight.address().valid() || !output.address().valid() ||
      input.address() == weight.address() ||
      input.address() == output.address() ||
      weight.address() == output.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  auto* native_handle = detail::native_cublas_handle(handle.handle());
  const auto stream_status =
      cublasSetStream(native_handle, detail::native_stream(stream));
  if (stream_status != CUBLAS_STATUS_SUCCESS) {
    return detail::cublas_status(stream_status);
  }

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const auto gemm_status = cublasGemmEx(
      native_handle, CUBLAS_OP_T, CUBLAS_OP_N,
      static_cast<int>(output_features), static_cast<int>(rows),
      static_cast<int>(input_features), &alpha,
      detail::native_address(weight.address()), CUDA_R_16F,
      static_cast<int>(input_features),
      detail::native_address(input.address()), CUDA_R_16F,
      static_cast<int>(input_features), &beta,
      detail::native_address(output.address()), CUDA_R_16F,
      static_cast<int>(output_features), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  return detail::cublas_status(gemm_status);
}

} // namespace marketforge::cuda
