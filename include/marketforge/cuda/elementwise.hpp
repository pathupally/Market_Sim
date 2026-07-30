#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Adds packed FP16 buffers. Output may exactly alias either input.
[[nodiscard]] Status add_f16(const DeviceBuffer& left,
                             const DeviceBuffer& right,
                             DeviceBuffer& output,
                             std::uint64_t elements,
                             StreamHandle stream) noexcept;

} // namespace marketforge::cuda
