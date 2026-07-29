#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

[[nodiscard]] Status launch_lifecycle_probe(const DeviceBuffer& input,
                                            DeviceBuffer& guarded_output,
                                            std::uint64_t element_count,
                                            StreamHandle stream) noexcept;

} // namespace marketforge::cuda
