#include "marketforge/cuda/elementwise.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;

__global__ void add_kernel(const __half* left, const __half* right,
                           __half* output, const std::uint64_t elements) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) *
                         static_cast<std::uint64_t>(blockDim.x) +
                     threadIdx.x;
  if (index < elements) {
    output[index] =
        __hadd_rn(left[index], right[index]);
  }
}

} // namespace

Status add_f16(const DeviceBuffer& left, const DeviceBuffer& right,
               DeviceBuffer& output, const std::uint64_t elements,
               const StreamHandle stream) noexcept {
  if (elements == 0 || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (elements >
      std::numeric_limits<std::uint64_t>::max() / sizeof(__half)) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto bytes = elements * sizeof(__half);
  if (left.size_bytes() != bytes || right.size_bytes() != bytes ||
      output.size_bytes() != bytes || !left.address().valid() ||
      !right.address().valid() || !output.address().valid() ||
      left.address() == right.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  const auto blocks =
      (elements + threads_per_block - 1) / threads_per_block;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return Status::failure(ErrorCode::resource_limit);
  }
  add_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0,
               detail::native_stream(stream)>>>(
      static_cast<const __half*>(detail::native_address(left.address())),
      static_cast<const __half*>(detail::native_address(right.address())),
      static_cast<__half*>(detail::native_address(output.address())), elements);
  return detail::launch_status(cudaGetLastError());
}

} // namespace marketforge::cuda
