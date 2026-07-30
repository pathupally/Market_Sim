#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Computes output[index] = SiLU(gate[index]) * up[index] for packed FP16
// buffers. Output may exactly alias gate; all other aliasing is rejected.
[[nodiscard]] Status swiglu_f16(const DeviceBuffer& gate,
                                const DeviceBuffer& up,
                                DeviceBuffer& output,
                                std::uint64_t elements,
                                StreamHandle stream) noexcept;

} // namespace marketforge::cuda
