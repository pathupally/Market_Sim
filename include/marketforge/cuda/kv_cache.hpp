#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Writes packed FP16 key/value tensors into contiguous FP16 caches:
// source[batch, tokens, key_value_heads, head_dim]
// cache[batch, maximum_context, key_value_heads, head_dim]
//
// positions contains packed uint32 values for [batch, tokens]. Position values
// are trusted scheduler input and must be unique within each batch. Out-of-range
// positions are skipped to preserve memory safety.
[[nodiscard]] Status append_kv_f16(
    const DeviceBuffer& key, const DeviceBuffer& value,
    const DeviceBuffer& positions, DeviceBuffer& key_cache,
    DeviceBuffer& value_cache, std::uint64_t batch, std::uint64_t tokens,
    std::uint64_t maximum_context, std::uint64_t key_value_heads,
    std::uint64_t head_dim, StreamHandle stream) noexcept;

} // namespace marketforge::cuda
