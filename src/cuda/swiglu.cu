#include "marketforge/cuda/swiglu.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

constexpr std::uint32_t threads_per_block = 256;

__global__ void swiglu_kernel(const __half* gate, const __half* up,
                              __half* output,
                              const std::uint64_t elements) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) *
                         static_cast<std::uint64_t>(blockDim.x) +
                     threadIdx.x;
  if (index >= elements) {
    return;
  }
  const float gate_value = __half2float(gate[index]);
  const float up_value = __half2float(up[index]);
  const float silu = gate_value / (1.0F + expf(-gate_value));
  output[index] = __float2half_rn(silu * up_value);
}

} // namespace

Status swiglu_f16(const DeviceBuffer& gate, const DeviceBuffer& up,
                  DeviceBuffer& output, const std::uint64_t elements,
                  const StreamHandle stream) noexcept {
  if (elements == 0 || !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  constexpr std::uint64_t element_bytes = sizeof(__half);
  if (elements >
      std::numeric_limits<std::uint64_t>::max() / element_bytes) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto bytes = elements * element_bytes;
  if (gate.size_bytes() != bytes || up.size_bytes() != bytes ||
      output.size_bytes() != bytes || !gate.address().valid() ||
      !up.address().valid() || !output.address().valid() ||
      gate.address() == up.address() || up.address() == output.address()) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  const auto blocks =
      (elements + threads_per_block - 1) / threads_per_block;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return Status::failure(ErrorCode::resource_limit);
  }
  swiglu_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, 0,
                  detail::native_stream(stream)>>>(
      static_cast<const __half*>(detail::native_address(gate.address())),
      static_cast<const __half*>(detail::native_address(up.address())),
      static_cast<__half*>(detail::native_address(output.address())), elements);
  return detail::launch_status(cudaGetLastError());
}

} // namespace marketforge::cuda
