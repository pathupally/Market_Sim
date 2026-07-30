#include "marketforge/cuda/kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;

[[nodiscard]] bool checked_product(
    const std::initializer_list<std::uint64_t> factors,
    std::uint64_t& result) noexcept {
  result = 1;
  for (const auto factor : factors) {
    if (factor == 0 ||
        result > std::numeric_limits<std::uint64_t>::max() / factor) {
      return false;
    }
    result *= factor;
  }
  return true;
}

[[nodiscard]] bool checked_bytes(const std::uint64_t elements,
                                 const std::uint64_t element_bytes,
                                 std::uint64_t& bytes) noexcept {
  if (elements > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
    return false;
  }
  bytes = elements * element_bytes;
  return true;
}

__global__ void append_kv_kernel(
    const __half* key, const __half* value,
    const std::uint32_t* positions, __half* key_cache, __half* value_cache,
    const std::uint64_t tokens, const std::uint64_t maximum_context,
    const std::uint64_t token_width, const std::uint64_t source_elements) {
  const auto source_index = static_cast<std::uint64_t>(blockIdx.x) *
                                static_cast<std::uint64_t>(blockDim.x) +
                            threadIdx.x;
  if (source_index >= source_elements) {
    return;
  }
  const auto packed_token = source_index / token_width;
  const auto position = static_cast<std::uint64_t>(positions[packed_token]);
  if (position >= maximum_context) {
    return;
  }
  const auto batch_index = packed_token / tokens;
  const auto element = source_index % token_width;
  const auto destination_index =
      (batch_index * maximum_context + position) * token_width + element;
  key_cache[destination_index] = key[source_index];
  value_cache[destination_index] = value[source_index];
}

} // namespace

Status append_kv_f16(
    const DeviceBuffer& key, const DeviceBuffer& value,
    const DeviceBuffer& positions, DeviceBuffer& key_cache,
    DeviceBuffer& value_cache, const std::uint64_t batch,
    const std::uint64_t tokens, const std::uint64_t maximum_context,
    const std::uint64_t key_value_heads, const std::uint64_t head_dim,
    const StreamHandle stream) noexcept {
  if (batch == 0 || tokens == 0 || maximum_context == 0 ||
      key_value_heads == 0 || head_dim == 0 || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  std::uint64_t packed_tokens = 0;
  std::uint64_t token_width = 0;
  std::uint64_t source_elements = 0;
  std::uint64_t cache_elements = 0;
  std::uint64_t source_bytes = 0;
  std::uint64_t cache_bytes = 0;
  std::uint64_t position_bytes = 0;
  if (!checked_product({batch, tokens}, packed_tokens) ||
      !checked_product({key_value_heads, head_dim}, token_width) ||
      !checked_product(
          {batch, tokens, key_value_heads, head_dim}, source_elements) ||
      !checked_product({batch, maximum_context, key_value_heads, head_dim},
                       cache_elements) ||
      !checked_bytes(source_elements, sizeof(__half), source_bytes) ||
      !checked_bytes(cache_elements, sizeof(__half), cache_bytes) ||
      !checked_bytes(
          packed_tokens, sizeof(std::uint32_t), position_bytes)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  if (key.size_bytes() != source_bytes ||
      value.size_bytes() != source_bytes ||
      positions.size_bytes() != position_bytes ||
      key_cache.size_bytes() != cache_bytes ||
      value_cache.size_bytes() != cache_bytes || !key.address().valid() ||
      !value.address().valid() || !positions.address().valid() ||
      !key_cache.address().valid() || !value_cache.address().valid() ||
      key.address() == value.address() ||
      key_cache.address() == value_cache.address() ||
      key.address() == key_cache.address() ||
      key.address() == value_cache.address() ||
      value.address() == key_cache.address() ||
      value.address() == value_cache.address() ||
      positions.address() == key.address() ||
      positions.address() == value.address() ||
      positions.address() == key_cache.address() ||
      positions.address() == value_cache.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  const auto blocks =
      (source_elements + threads_per_block - 1) / threads_per_block;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return Status::failure(ErrorCode::resource_limit);
  }
  append_kv_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0,
                     detail::native_stream(stream)>>>(
      static_cast<const __half*>(detail::native_address(key.address())),
      static_cast<const __half*>(detail::native_address(value.address())),
      static_cast<const std::uint32_t*>(
          detail::native_address(positions.address())),
      static_cast<__half*>(detail::native_address(key_cache.address())),
      static_cast<__half*>(detail::native_address(value_cache.address())),
      tokens, maximum_context, token_width, source_elements);
  return detail::launch_status(cudaGetLastError());
}

} // namespace marketforge::cuda
