#include "marketforge/cuda/greedy.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>

#include <cuda_fp16.h>
#include <math_constants.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;
constexpr std::uint32_t invalid_token_id =
    restricted_greedy_invalid_token_id;

__global__ void greedy_kernel(const __half* logits,
                              std::uint32_t* token_ids,
                              const std::uint64_t vocabulary_size) {
  __shared__ float maxima[threads_per_block];
  __shared__ std::uint32_t indices[threads_per_block];
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  const auto row_offset = row * vocabulary_size;
  float maximum = -CUDART_INF_F;
  std::uint32_t token_id = invalid_token_id;
  for (std::uint64_t column = threadIdx.x; column < vocabulary_size;
       column += blockDim.x) {
    const float value = __half2float(logits[row_offset + column]);
    const auto candidate = static_cast<std::uint32_t>(column);
    if (value > maximum || (value == maximum && candidate < token_id)) {
      maximum = value;
      token_id = candidate;
    }
  }
  maxima[threadIdx.x] = maximum;
  indices[threadIdx.x] = token_id;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float candidate_value = maxima[threadIdx.x + stride];
      const auto candidate_index = indices[threadIdx.x + stride];
      if (candidate_value > maxima[threadIdx.x] ||
          (candidate_value == maxima[threadIdx.x] &&
           candidate_index < indices[threadIdx.x])) {
        maxima[threadIdx.x] = candidate_value;
        indices[threadIdx.x] = candidate_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    token_ids[row] = indices[0] == invalid_token_id ? 0 : indices[0];
  }
}

__global__ void restricted_greedy_kernel(
    const __half* logits, const std::uint32_t* allowed_token_ids,
    const std::uint32_t* allowed_token_counts, std::uint32_t* token_ids,
    const std::uint64_t vocabulary_size,
    const std::uint64_t maximum_allowed_tokens) {
  __shared__ float maxima[threads_per_block];
  __shared__ std::uint32_t indices[threads_per_block];
  __shared__ std::uint32_t fallbacks[threads_per_block];
  __shared__ unsigned int row_invalid;

  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  const auto allowed_count = allowed_token_counts[row];
  if (threadIdx.x == 0) {
    row_invalid =
        allowed_count == 0 || allowed_count > maximum_allowed_tokens ? 1U : 0U;
  }
  __syncthreads();

  const auto logits_offset = row * vocabulary_size;
  const auto allowed_offset = row * maximum_allowed_tokens;
  float maximum = -CUDART_INF_F;
  std::uint32_t token_id = invalid_token_id;
  std::uint32_t fallback = invalid_token_id;
  if (row_invalid == 0U) {
    for (std::uint64_t candidate_index = threadIdx.x;
         candidate_index < allowed_count;
         candidate_index += blockDim.x) {
      const auto candidate =
          allowed_token_ids[allowed_offset + candidate_index];
      if (candidate >= vocabulary_size) {
        atomicExch(&row_invalid, 1U);
        continue;
      }
      fallback = candidate < fallback ? candidate : fallback;
      const float value = __half2float(logits[logits_offset + candidate]);
      if (value > maximum ||
          (value == maximum && candidate < token_id)) {
        maximum = value;
        token_id = candidate;
      }
    }
  }
  maxima[threadIdx.x] = maximum;
  indices[threadIdx.x] = token_id;
  fallbacks[threadIdx.x] = fallback;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float candidate_value = maxima[threadIdx.x + stride];
      const auto candidate_index = indices[threadIdx.x + stride];
      if (candidate_value > maxima[threadIdx.x] ||
          (candidate_value == maxima[threadIdx.x] &&
           candidate_index < indices[threadIdx.x])) {
        maxima[threadIdx.x] = candidate_value;
        indices[threadIdx.x] = candidate_index;
      }
      fallbacks[threadIdx.x] =
          fallbacks[threadIdx.x + stride] < fallbacks[threadIdx.x]
              ? fallbacks[threadIdx.x + stride]
              : fallbacks[threadIdx.x];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    token_ids[row] =
        row_invalid != 0U
            ? invalid_token_id
            : (indices[0] == invalid_token_id ? fallbacks[0] : indices[0]);
  }
}

__global__ void restricted_output_head_kernel(
    const __half* hidden, const __half* embedding,
    const std::uint32_t* allowed_token_ids,
    const std::uint32_t* allowed_token_counts, std::uint32_t* token_ids,
    const std::uint64_t hidden_size,
    const std::uint64_t vocabulary_size,
    const std::uint64_t maximum_allowed_tokens) {
  __shared__ float maxima[threads_per_block];
  __shared__ std::uint32_t indices[threads_per_block];
  __shared__ std::uint32_t fallbacks[threads_per_block];
  __shared__ unsigned int row_invalid;

  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  const auto allowed_count = allowed_token_counts[row];
  if (threadIdx.x == 0) {
    row_invalid =
        allowed_count == 0 || allowed_count > maximum_allowed_tokens ? 1U : 0U;
  }
  __syncthreads();

  const auto hidden_offset = row * hidden_size;
  const auto allowed_offset = row * maximum_allowed_tokens;
  float maximum = -CUDART_INF_F;
  std::uint32_t token_id = invalid_token_id;
  std::uint32_t fallback = invalid_token_id;
  if (row_invalid == 0U) {
    for (std::uint64_t candidate_index = threadIdx.x;
         candidate_index < allowed_count;
         candidate_index += blockDim.x) {
      const auto candidate =
          allowed_token_ids[allowed_offset + candidate_index];
      if (candidate >= vocabulary_size) {
        atomicExch(&row_invalid, 1U);
        continue;
      }
      fallback = candidate < fallback ? candidate : fallback;
      const auto embedding_offset =
          static_cast<std::uint64_t>(candidate) * hidden_size;
      float score = 0.0F;
      for (std::uint64_t feature = 0; feature < hidden_size; ++feature) {
        score = fmaf(__half2float(hidden[hidden_offset + feature]),
                     __half2float(embedding[embedding_offset + feature]),
                     score);
      }
      if (score > maximum ||
          (score == maximum && candidate < token_id)) {
        maximum = score;
        token_id = candidate;
      }
    }
  }
  maxima[threadIdx.x] = maximum;
  indices[threadIdx.x] = token_id;
  fallbacks[threadIdx.x] = fallback;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float candidate_value = maxima[threadIdx.x + stride];
      const auto candidate_index = indices[threadIdx.x + stride];
      if (candidate_value > maxima[threadIdx.x] ||
          (candidate_value == maxima[threadIdx.x] &&
           candidate_index < indices[threadIdx.x])) {
        maxima[threadIdx.x] = candidate_value;
        indices[threadIdx.x] = candidate_index;
      }
      fallbacks[threadIdx.x] =
          fallbacks[threadIdx.x + stride] < fallbacks[threadIdx.x]
              ? fallbacks[threadIdx.x + stride]
              : fallbacks[threadIdx.x];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    token_ids[row] =
        row_invalid != 0U
            ? invalid_token_id
            : (indices[0] == invalid_token_id ? fallbacks[0] : indices[0]);
  }
}

bool buffers_are_distinct(const DeviceBuffer& first,
                          const DeviceBuffer& second,
                          const DeviceBuffer& third,
                          const DeviceBuffer& fourth,
                          const DeviceBuffer& fifth) noexcept {
  const DeviceAddress addresses[] = {
      first.address(), second.address(), third.address(), fourth.address(),
      fifth.address()};
  for (std::size_t left = 0; left < std::size(addresses); ++left) {
    if (!addresses[left].valid()) {
      return false;
    }
    for (std::size_t right = left + 1; right < std::size(addresses); ++right) {
      if (addresses[left] == addresses[right]) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

Status greedy_select_f16(const DeviceBuffer& logits, DeviceBuffer& token_ids,
                         const std::uint64_t rows,
                         const std::uint64_t vocabulary_size,
                         const StreamHandle stream) noexcept {
  if (rows == 0 || vocabulary_size == 0 || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (vocabulary_size >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::uint32_t>::max())) {
    return Status::failure(ErrorCode::resource_limit);
  }
  if (rows >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Status::failure(ErrorCode::resource_limit);
  }
  if (rows > std::numeric_limits<std::uint64_t>::max() / vocabulary_size) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto elements = rows * vocabulary_size;
  if (elements >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half) ||
      rows > std::numeric_limits<std::uint64_t>::max() /
                 sizeof(std::uint32_t)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto logits_bytes = elements * sizeof(__half);
  const auto token_bytes = rows * sizeof(std::uint32_t);
  if (logits.size_bytes() != logits_bytes ||
      token_ids.size_bytes() != token_bytes || !logits.address().valid() ||
      !token_ids.address().valid() ||
      logits.address() == token_ids.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  greedy_kernel<<<static_cast<unsigned int>(rows), threads_per_block, 0,
                  detail::native_stream(stream)>>>(
      static_cast<const __half*>(detail::native_address(logits.address())),
      static_cast<std::uint32_t*>(
          detail::native_address(token_ids.address())),
      vocabulary_size);
  return detail::launch_status(cudaGetLastError());
}

Status restricted_greedy_select_f16(
    const DeviceBuffer& logits, const DeviceBuffer& allowed_token_ids,
    const DeviceBuffer& allowed_token_counts, DeviceBuffer& token_ids,
    const std::uint64_t rows, const std::uint64_t vocabulary_size,
    const std::uint64_t maximum_allowed_tokens,
    const StreamHandle stream) noexcept {
  if (rows == 0 || vocabulary_size == 0 ||
      maximum_allowed_tokens == 0 || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (vocabulary_size >
          std::numeric_limits<std::uint32_t>::max() ||
      maximum_allowed_tokens >
          std::numeric_limits<std::uint32_t>::max() ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Status::failure(ErrorCode::resource_limit);
  }
  if (rows > std::numeric_limits<std::uint64_t>::max() / vocabulary_size ||
      rows > std::numeric_limits<std::uint64_t>::max() /
                 maximum_allowed_tokens) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto logits_elements = rows * vocabulary_size;
  const auto allowed_elements = rows * maximum_allowed_tokens;
  if (logits_elements >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half) ||
      allowed_elements >
          std::numeric_limits<std::uint64_t>::max() /
              sizeof(std::uint32_t) ||
      rows > std::numeric_limits<std::uint64_t>::max() /
                 sizeof(std::uint32_t)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto logits_bytes = logits_elements * sizeof(__half);
  const auto allowed_bytes =
      allowed_elements * sizeof(std::uint32_t);
  const auto row_bytes = rows * sizeof(std::uint32_t);
  if (logits.size_bytes() != logits_bytes ||
      allowed_token_ids.size_bytes() != allowed_bytes ||
      allowed_token_counts.size_bytes() != row_bytes ||
      token_ids.size_bytes() != row_bytes || !logits.address().valid() ||
      !allowed_token_ids.address().valid() ||
      !allowed_token_counts.address().valid() ||
      !token_ids.address().valid() ||
      logits.address() == allowed_token_ids.address() ||
      logits.address() == allowed_token_counts.address() ||
      logits.address() == token_ids.address() ||
      allowed_token_ids.address() == allowed_token_counts.address() ||
      allowed_token_ids.address() == token_ids.address() ||
      allowed_token_counts.address() == token_ids.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  restricted_greedy_kernel<<<static_cast<unsigned int>(rows),
                             threads_per_block, 0,
                             detail::native_stream(stream)>>>(
      static_cast<const __half*>(
          detail::native_address(logits.address())),
      static_cast<const std::uint32_t*>(
          detail::native_address(allowed_token_ids.address())),
      static_cast<const std::uint32_t*>(
          detail::native_address(allowed_token_counts.address())),
      static_cast<std::uint32_t*>(
          detail::native_address(token_ids.address())),
      vocabulary_size, maximum_allowed_tokens);
  return detail::launch_status(cudaGetLastError());
}

Status restricted_output_head_f16(
    const DeviceBuffer& hidden, const DeviceBuffer& embedding,
    const DeviceBuffer& allowed_token_ids,
    const DeviceBuffer& allowed_token_counts, DeviceBuffer& token_ids,
    const std::uint64_t rows, const std::uint64_t hidden_size,
    const std::uint64_t vocabulary_size,
    const std::uint64_t maximum_allowed_tokens,
    const StreamHandle stream) noexcept {
  if (rows == 0 || hidden_size == 0 || vocabulary_size == 0 ||
      maximum_allowed_tokens == 0 || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (vocabulary_size > std::numeric_limits<std::uint32_t>::max() ||
      maximum_allowed_tokens > std::numeric_limits<std::uint32_t>::max() ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Status::failure(ErrorCode::resource_limit);
  }
  if (rows > std::numeric_limits<std::uint64_t>::max() / hidden_size ||
      vocabulary_size >
          std::numeric_limits<std::uint64_t>::max() / hidden_size ||
      rows > std::numeric_limits<std::uint64_t>::max() /
                 maximum_allowed_tokens) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto hidden_elements = rows * hidden_size;
  const auto embedding_elements = vocabulary_size * hidden_size;
  const auto allowed_elements = rows * maximum_allowed_tokens;
  if (hidden_elements >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half) ||
      embedding_elements >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half) ||
      allowed_elements >
          std::numeric_limits<std::uint64_t>::max() /
              sizeof(std::uint32_t) ||
      rows > std::numeric_limits<std::uint64_t>::max() /
                 sizeof(std::uint32_t)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto hidden_bytes = hidden_elements * sizeof(__half);
  const auto embedding_bytes = embedding_elements * sizeof(__half);
  const auto allowed_bytes = allowed_elements * sizeof(std::uint32_t);
  const auto row_bytes = rows * sizeof(std::uint32_t);
  if (hidden.size_bytes() != hidden_bytes ||
      embedding.size_bytes() != embedding_bytes ||
      allowed_token_ids.size_bytes() != allowed_bytes ||
      allowed_token_counts.size_bytes() != row_bytes ||
      token_ids.size_bytes() != row_bytes ||
      !buffers_are_distinct(hidden, embedding, allowed_token_ids,
                            allowed_token_counts, token_ids)) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  restricted_output_head_kernel<<<static_cast<unsigned int>(rows),
                                  threads_per_block, 0,
                                  detail::native_stream(stream)>>>(
      static_cast<const __half*>(detail::native_address(hidden.address())),
      static_cast<const __half*>(detail::native_address(embedding.address())),
      static_cast<const std::uint32_t*>(
          detail::native_address(allowed_token_ids.address())),
      static_cast<const std::uint32_t*>(
          detail::native_address(allowed_token_counts.address())),
      static_cast<std::uint32_t*>(
          detail::native_address(token_ids.address())),
      hidden_size, vocabulary_size, maximum_allowed_tokens);
  return detail::launch_status(cudaGetLastError());
}

} // namespace marketforge::cuda
