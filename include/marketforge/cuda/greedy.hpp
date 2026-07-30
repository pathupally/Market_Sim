#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Selects one token ID per row from packed FP16 logits[rows, vocabulary_size].
// Equal logits resolve to the lowest token ID. NaNs never win a comparison.
// token_ids is a packed uint32 output buffer with one element per row.
[[nodiscard]] Status greedy_select_f16(const DeviceBuffer& logits,
                                       DeviceBuffer& token_ids,
                                       std::uint64_t rows,
                                       std::uint64_t vocabulary_size,
                                       StreamHandle stream) noexcept;

} // namespace marketforge::cuda
