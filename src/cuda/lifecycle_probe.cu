#include "marketforge/cuda/lifecycle_probe.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

__global__ void lifecycle_kernel(const std::uint32_t* input,
                                 std::uint32_t* output,
                                 const std::uint64_t element_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < element_count) {
    output[index] = input[index] * UINT32_C(3) + UINT32_C(7);
  }
}

} // namespace

Status launch_lifecycle_probe(const DeviceBuffer& input,
                              DeviceBuffer& guarded_output,
                              const std::uint64_t element_count,
                              const StreamHandle stream) noexcept {
  if (element_count == 0) {
    return Status::success();
  }
  if (!stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  constexpr std::uint64_t word_bytes = sizeof(std::uint32_t);
  if (element_count >
      (std::numeric_limits<std::uint64_t>::max() / word_bytes) - 2) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto input_bytes = element_count * word_bytes;
  const auto output_bytes = (element_count + 2) * word_bytes;
  if (input.size_bytes() < input_bytes ||
      guarded_output.size_bytes() < output_bytes || !input.address().valid() ||
      !guarded_output.address().valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  constexpr unsigned int threads = 256;
  const auto block_count_u64 = (element_count + threads - 1) / threads;
  if (block_count_u64 > std::numeric_limits<unsigned int>::max()) {
    return Status::failure(ErrorCode::insufficient_resources);
  }

  const auto* input_words = static_cast<const std::uint32_t*>(
      detail::native_address(input.address()));
  auto* guarded_words = static_cast<std::uint32_t*>(
      detail::native_address(guarded_output.address()));
  lifecycle_kernel<<<static_cast<unsigned int>(block_count_u64), threads, 0,
                     detail::native_stream(stream)>>>(
      input_words, guarded_words + 1, element_count);
  return detail::launch_status(cudaPeekAtLastError());
}

} // namespace marketforge::cuda
