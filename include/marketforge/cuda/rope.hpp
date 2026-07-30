#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

// Applies Llama half-rotation RoPE in place to packed FP16 tensors:
// query[batch, tokens, query_heads, head_dim]
// key[batch, tokens, key_value_heads, head_dim]
//
// positions contains packed uint32 values for [batch, tokens]. Position values
// are trusted scheduler input; shape, allocation, theta, alias, and launch
// limits are validated synchronously before the kernel is launched.
[[nodiscard]] Status apply_rope_f16(
    DeviceBuffer& query, DeviceBuffer& key, const DeviceBuffer& positions,
    std::uint64_t batch, std::uint64_t tokens, std::uint64_t query_heads,
    std::uint64_t key_value_heads, std::uint64_t head_dim, float theta,
    StreamHandle stream) noexcept;

} // namespace marketforge::cuda
