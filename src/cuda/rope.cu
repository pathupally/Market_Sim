#include "marketforge/cuda/rope.hpp"

#include <cmath>
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

[[nodiscard]] Status checked_product(
    const std::initializer_list<std::uint64_t> factors,
    std::uint64_t& result) noexcept {
  result = 1;
  for (const auto factor : factors) {
    if (factor == 0) {
      return Status::failure(ErrorCode::invalid_argument);
    }
    if (result > std::numeric_limits<std::uint64_t>::max() / factor) {
      return Status::failure(ErrorCode::arithmetic_overflow);
    }
    result *= factor;
  }
  return Status::success();
}

[[nodiscard]] Status checked_bytes(const std::uint64_t elements,
                                   const std::uint64_t element_bytes,
                                   std::uint64_t& bytes) noexcept {
  if (elements > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  bytes = elements * element_bytes;
  return Status::success();
}

__global__ void rope_kernel(__half* query, __half* key,
                            const std::uint32_t* positions,
                            const std::uint64_t tokens_and_batch,
                            const std::uint64_t query_heads,
                            const std::uint64_t key_value_heads,
                            const std::uint64_t head_dim,
                            const float theta) {
  const auto pair_index = static_cast<std::uint64_t>(blockIdx.x) *
                              static_cast<std::uint64_t>(blockDim.x) +
                          threadIdx.x;
  const auto half = head_dim / 2;
  const auto pair_count = tokens_and_batch * half;
  if (pair_index >= pair_count) {
    return;
  }
  const auto token = pair_index / half;
  const auto dimension = pair_index % half;
  const auto exponent =
      static_cast<float>(2 * dimension) / static_cast<float>(head_dim);
  const auto angle =
      static_cast<float>(positions[token]) / powf(theta, exponent);
  float sine = 0.0F;
  float cosine = 0.0F;
  sincosf(angle, &sine, &cosine);

  for (std::uint64_t head = 0; head < query_heads; ++head) {
    const auto offset = (token * query_heads + head) * head_dim + dimension;
    const float first = __half2float(query[offset]);
    const float second = __half2float(query[offset + half]);
    query[offset] = __float2half_rn(first * cosine - second * sine);
    query[offset + half] =
        __float2half_rn(second * cosine + first * sine);
  }
  for (std::uint64_t head = 0; head < key_value_heads; ++head) {
    const auto offset =
        (token * key_value_heads + head) * head_dim + dimension;
    const float first = __half2float(key[offset]);
    const float second = __half2float(key[offset + half]);
    key[offset] = __float2half_rn(first * cosine - second * sine);
    key[offset + half] =
        __float2half_rn(second * cosine + first * sine);
  }
}

} // namespace

Status apply_rope_f16(
    DeviceBuffer& query, DeviceBuffer& key, const DeviceBuffer& positions,
    const std::uint64_t batch, const std::uint64_t tokens,
    const std::uint64_t query_heads, const std::uint64_t key_value_heads,
    const std::uint64_t head_dim, const float theta,
    const StreamHandle stream) noexcept {
  if (!stream.valid() || !std::isfinite(theta) || theta <= 0.0F ||
      head_dim % 2 != 0) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  std::uint64_t tokens_and_batch = 0;
  std::uint64_t query_elements = 0;
  std::uint64_t key_elements = 0;
  std::uint64_t query_bytes = 0;
  std::uint64_t key_bytes = 0;
  std::uint64_t position_bytes = 0;
  for (const auto status : {
           checked_product({batch, tokens}, tokens_and_batch),
           checked_product(
               {batch, tokens, query_heads, head_dim}, query_elements),
           checked_product(
               {batch, tokens, key_value_heads, head_dim}, key_elements),
           checked_bytes(query_elements, sizeof(__half), query_bytes),
           checked_bytes(key_elements, sizeof(__half), key_bytes),
           checked_bytes(
               tokens_and_batch, sizeof(std::uint32_t), position_bytes),
       }) {
    if (!status.ok()) {
      return status;
    }
  }

  if (query.size_bytes() != query_bytes || key.size_bytes() != key_bytes ||
      positions.size_bytes() != position_bytes ||
      !query.address().valid() || !key.address().valid() ||
      !positions.address().valid() || query.address() == key.address() ||
      query.address() == positions.address() ||
      key.address() == positions.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  const auto pair_count = tokens_and_batch * (head_dim / 2);
  const auto blocks =
      (pair_count + threads_per_block - 1) / threads_per_block;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return Status::failure(ErrorCode::resource_limit);
  }
  rope_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0,
                detail::native_stream(stream)>>>(
      static_cast<__half*>(detail::native_address(query.address())),
      static_cast<__half*>(detail::native_address(key.address())),
      static_cast<const std::uint32_t*>(
          detail::native_address(positions.address())),
      tokens_and_batch, query_heads, key_value_heads, head_dim, theta);
  return detail::launch_status(cudaGetLastError());
}

} // namespace marketforge::cuda
