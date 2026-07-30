#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

inline constexpr std::uint32_t restricted_greedy_invalid_token_id =
    UINT32_MAX;

// Selects one token ID per row from packed FP16 logits[rows, vocabulary_size].
// Equal logits resolve to the lowest token ID. NaNs never win a comparison.
// token_ids is a packed uint32 output buffer with one element per row.
[[nodiscard]] Status greedy_select_f16(const DeviceBuffer& logits,
                                       DeviceBuffer& token_ids,
                                       std::uint64_t rows,
                                       std::uint64_t vocabulary_size,
                                       StreamHandle stream) noexcept;

// Selects one token ID per row from a fixed-width packed candidate table.
// allowed_token_ids is uint32[rows, maximum_allowed_tokens] and
// allowed_token_counts is uint32[rows]. Candidate order is irrelevant. Equal
// logits resolve to the lowest token ID, and an all-NaN valid set falls back to
// its lowest token ID. A row with zero/oversized count or an out-of-vocabulary
// candidate emits restricted_greedy_invalid_token_id.
[[nodiscard]] Status restricted_greedy_select_f16(
    const DeviceBuffer& logits, const DeviceBuffer& allowed_token_ids,
    const DeviceBuffer& allowed_token_counts, DeviceBuffer& token_ids,
    std::uint64_t rows, std::uint64_t vocabulary_size,
    std::uint64_t maximum_allowed_tokens, StreamHandle stream) noexcept;

} // namespace marketforge::cuda
