#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Computes causal grouped-query attention over contiguous FP16 caches.
// query[batch, tokens, query_heads, head_dim]
// cache[batch, maximum_context, key_value_heads, head_dim]
// output has the same packed shape as query.
//
// positions is packed uint32[batch, tokens] scheduler state. Each position must
// identify the query token's contiguous cache slot.
[[nodiscard]] Status attention_f16(
    const DeviceBuffer& query, const DeviceBuffer& key_cache,
    const DeviceBuffer& value_cache, const DeviceBuffer& positions,
    DeviceBuffer& output, std::uint64_t batch, std::uint64_t tokens,
    std::uint64_t maximum_context, std::uint64_t query_heads,
    std::uint64_t key_value_heads, std::uint64_t head_dim,
    StreamHandle stream) noexcept;

} // namespace marketforge::cuda
