#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cublas_handle.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Computes output[rows, output_features] =
//   input[rows, input_features] * weight[output_features, input_features]^T.
//
// Input, weight, and output are packed IEEE FP16. cuBLAS accumulates in FP32
// and rounds the result to FP16. The operation is asynchronous on the explicit
// stream, and all allocation sizes must match their matrix shapes exactly.
[[nodiscard]] Status
linear_f16(const DeviceBuffer& input, const DeviceBuffer& weight,
           DeviceBuffer& output, std::uint64_t rows,
           std::uint64_t input_features, std::uint64_t output_features,
           CublasHandle& handle, StreamHandle stream) noexcept;

} // namespace marketforge::cuda
