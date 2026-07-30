#include "marketforge/cuda/attention.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <cuda_fp16.h>
#include <math_constants.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;
constexpr std::uint64_t maximum_supported_context = 8'192;

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

[[nodiscard]] bool checked_f16_bytes(const std::uint64_t elements,
                                     std::uint64_t& bytes) noexcept {
  if (elements >
      std::numeric_limits<std::uint64_t>::max() / sizeof(__half)) {
    return false;
  }
  bytes = elements * sizeof(__half);
  return true;
}

__global__ void attention_kernel(
    const __half* query, const __half* key_cache,
    const __half* value_cache, const std::uint32_t* positions, __half* output,
    const std::uint64_t tokens, const std::uint64_t maximum_context,
    const std::uint64_t query_heads, const std::uint64_t key_value_heads,
    const std::uint64_t head_dim) {
  extern __shared__ float scratch[];
  float* const logits = scratch;
  float* const reductions = scratch + maximum_context;
  const auto vector = static_cast<std::uint64_t>(blockIdx.x);
  const auto query_head = vector % query_heads;
  const auto packed_token = vector / query_heads;
  const auto batch_index = packed_token / tokens;
  const auto groups = query_heads / key_value_heads;
  const auto key_value_head = query_head / groups;
  const auto valid_keys =
      static_cast<std::uint64_t>(positions[packed_token]) + 1;
  const auto output_offset = vector * head_dim;
  if (valid_keys > maximum_context) {
    for (std::uint64_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
      output[output_offset + dimension] = __float2half_rn(0.0F);
    }
    return;
  }

  float local_maximum = -CUDART_INF_F;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (std::uint64_t key_token = threadIdx.x; key_token < valid_keys;
       key_token += blockDim.x) {
    float dot = 0.0F;
    const auto key_offset =
        ((batch_index * maximum_context + key_token) * key_value_heads +
         key_value_head) *
        head_dim;
    for (std::uint64_t dimension = 0; dimension < head_dim; ++dimension) {
      dot = fmaf(__half2float(query[output_offset + dimension]),
                 __half2float(key_cache[key_offset + dimension]), dot);
    }
    const float logit = dot * scale;
    logits[key_token] = logit;
    local_maximum = fmaxf(local_maximum, logit);
  }
  reductions[threadIdx.x] = local_maximum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] =
          fmaxf(reductions[threadIdx.x],
                reductions[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float maximum = reductions[0];

  float local_sum = 0.0F;
  for (std::uint64_t key_token = threadIdx.x; key_token < valid_keys;
       key_token += blockDim.x) {
    const float probability = expf(logits[key_token] - maximum);
    logits[key_token] = probability;
    local_sum += probability;
  }
  reductions[threadIdx.x] = local_sum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_denominator = 1.0F / reductions[0];

  for (std::uint64_t dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    float weighted_value = 0.0F;
    for (std::uint64_t key_token = 0; key_token < valid_keys; ++key_token) {
      const auto value_offset =
          ((batch_index * maximum_context + key_token) * key_value_heads +
           key_value_head) *
              head_dim +
          dimension;
      weighted_value = fmaf(logits[key_token] * inverse_denominator,
                            __half2float(value_cache[value_offset]),
                            weighted_value);
    }
    output[output_offset + dimension] =
        __float2half_rn(weighted_value);
  }
}

} // namespace

Status attention_f16(
    const DeviceBuffer& query, const DeviceBuffer& key_cache,
    const DeviceBuffer& value_cache, const DeviceBuffer& positions,
    DeviceBuffer& output, const std::uint64_t batch,
    const std::uint64_t tokens, const std::uint64_t maximum_context,
    const std::uint64_t query_heads,
    const std::uint64_t key_value_heads, const std::uint64_t head_dim,
    const StreamHandle stream) noexcept {
  if (batch == 0 || tokens == 0 || maximum_context == 0 ||
      maximum_context > maximum_supported_context || query_heads == 0 ||
      key_value_heads == 0 || query_heads % key_value_heads != 0 ||
      head_dim == 0 || head_dim > threads_per_block || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  std::uint64_t query_elements = 0;
  std::uint64_t cache_elements = 0;
  std::uint64_t packed_tokens = 0;
  std::uint64_t query_bytes = 0;
  std::uint64_t cache_bytes = 0;
  std::uint64_t position_bytes = 0;
  if (!checked_product(
          {batch, tokens, query_heads, head_dim}, query_elements) ||
      !checked_product({batch, maximum_context, key_value_heads, head_dim},
                       cache_elements) ||
      !checked_product({batch, tokens}, packed_tokens) ||
      !checked_f16_bytes(query_elements, query_bytes) ||
      !checked_f16_bytes(cache_elements, cache_bytes) ||
      packed_tokens > std::numeric_limits<std::uint64_t>::max() /
                          sizeof(std::uint32_t)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  position_bytes = packed_tokens * sizeof(std::uint32_t);
  if (query.size_bytes() != query_bytes ||
      output.size_bytes() != query_bytes ||
      key_cache.size_bytes() != cache_bytes ||
      value_cache.size_bytes() != cache_bytes ||
      positions.size_bytes() != position_bytes || !query.address().valid() ||
      !output.address().valid() || !key_cache.address().valid() ||
      !value_cache.address().valid() || !positions.address().valid() ||
      query.address() == output.address() ||
      key_cache.address() == value_cache.address() ||
      output.address() == key_cache.address() ||
      output.address() == value_cache.address() ||
      output.address() == positions.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  const auto query_vectors = query_elements / head_dim;
  if (query_vectors >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Status::failure(ErrorCode::resource_limit);
  }
  const auto shared_bytes =
      (maximum_context + threads_per_block) * sizeof(float);
  attention_kernel<<<static_cast<unsigned int>(query_vectors),
                     threads_per_block,
                     static_cast<std::size_t>(shared_bytes),
                     detail::native_stream(stream)>>>(
      static_cast<const __half*>(detail::native_address(query.address())),
      static_cast<const __half*>(
          detail::native_address(key_cache.address())),
      static_cast<const __half*>(
          detail::native_address(value_cache.address())),
      static_cast<const std::uint32_t*>(
          detail::native_address(positions.address())),
      static_cast<__half*>(detail::native_address(output.address())), tokens,
      maximum_context, query_heads, key_value_heads, head_dim);
  return detail::launch_status(cudaGetLastError());
}

} // namespace marketforge::cuda
