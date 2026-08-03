#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Gathers packed uint32 token IDs from an FP16
// embedding[vocabulary_size, hidden_size] into output[tokens, hidden_size].
// Out-of-vocabulary IDs produce an all-zero row and remain host-validation
// errors at the public model boundary.
[[nodiscard]] Status embedding_lookup_f16(
    const DeviceBuffer& embedding, const DeviceBuffer& token_ids,
    DeviceBuffer& output, std::uint64_t tokens,
    std::uint64_t vocabulary_size, std::uint64_t hidden_size,
    StreamHandle stream) noexcept;

} // namespace marketforge::cuda
