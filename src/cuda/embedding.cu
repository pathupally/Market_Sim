#include "marketforge/cuda/embedding.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;

__global__ void embedding_kernel(
    const __half* embedding, const std::uint32_t* token_ids, __half* output,
    const std::uint64_t elements, const std::uint64_t vocabulary_size,
    const std::uint64_t hidden_size) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) *
                         static_cast<std::uint64_t>(blockDim.x) +
                     threadIdx.x;
  if (index >= elements) {
    return;
  }
  const auto token = index / hidden_size;
  const auto column = index % hidden_size;
  const auto token_id = static_cast<std::uint64_t>(token_ids[token]);
  output[index] = token_id < vocabulary_size
                      ? embedding[token_id * hidden_size + column]
                      : __float2half_rn(0.0F);
}

} // namespace

Status embedding_lookup_f16(
    const DeviceBuffer& embedding, const DeviceBuffer& token_ids,
    DeviceBuffer& output, const std::uint64_t tokens,
    const std::uint64_t vocabulary_size, const std::uint64_t hidden_size,
    const StreamHandle stream) noexcept {
  if (tokens == 0 || vocabulary_size == 0 || hidden_size == 0 ||
      !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (tokens > std::numeric_limits<std::uint64_t>::max() / hidden_size ||
      vocabulary_size >
          std::numeric_limits<std::uint64_t>::max() / hidden_size) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto output_elements = tokens * hidden_size;
  const auto embedding_elements = vocabulary_size * hidden_size;
  if (output_elements >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half) ||
      embedding_elements >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half) ||
      tokens > std::numeric_limits<std::uint64_t>::max() /
                   sizeof(std::uint32_t)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto output_bytes = output_elements * sizeof(__half);
  const auto embedding_bytes = embedding_elements * sizeof(__half);
  const auto token_bytes = tokens * sizeof(std::uint32_t);
  if (embedding.size_bytes() != embedding_bytes ||
      token_ids.size_bytes() != token_bytes ||
      output.size_bytes() != output_bytes || !embedding.address().valid() ||
      !token_ids.address().valid() || !output.address().valid() ||
      embedding.address() == output.address() ||
      token_ids.address() == output.address() ||
      embedding.address() == token_ids.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  const auto blocks =
      (output_elements + threads_per_block - 1) / threads_per_block;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return Status::failure(ErrorCode::resource_limit);
  }
  embedding_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0,
                     detail::native_stream(stream)>>>(
      static_cast<const __half*>(
          detail::native_address(embedding.address())),
      static_cast<const std::uint32_t*>(
          detail::native_address(token_ids.address())),
      static_cast<__half*>(detail::native_address(output.address())),
      output_elements, vocabulary_size, hidden_size);
  return detail::launch_status(cudaGetLastError());
}

} // namespace marketforge::cuda
