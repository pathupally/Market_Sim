#include "marketforge/cuda/greedy.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;

__global__ void greedy_kernel(const __half* logits,
                              std::uint32_t* token_ids,
                              const std::uint64_t vocabulary_size) {
  __shared__ float maxima[threads_per_block];
  __shared__ std::uint32_t indices[threads_per_block];
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  const auto row_offset = row * vocabulary_size;
  float maximum = -std::numeric_limits<float>::infinity();
  std::uint32_t token_id = std::numeric_limits<std::uint32_t>::max();
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
    token_ids[row] =
        indices[0] == std::numeric_limits<std::uint32_t>::max()
            ? 0
            : indices[0];
  }
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

} // namespace marketforge::cuda
