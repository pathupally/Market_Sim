#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Applies row-wise FP32 RMSNorm:
// output[row, column] =
//   input[row, column] * weight[column] /
//   sqrt(mean(input[row, :] ^ 2) + epsilon).
//
// The operation is asynchronous on the explicit stream. Input and output may
// be the same allocation. All buffers must have exact packed FP32 sizes.
[[nodiscard]] Status rms_norm_f32(const DeviceBuffer& input,
                                  const DeviceBuffer& weight,
                                  DeviceBuffer& output, std::uint64_t rows,
                                  std::uint64_t hidden_size, float epsilon,
                                  StreamHandle stream) noexcept;

} // namespace marketforge::cuda
