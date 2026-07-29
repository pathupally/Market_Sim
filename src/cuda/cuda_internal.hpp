#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda::detail {

[[nodiscard]] inline cudaStream_t native_stream(StreamHandle handle) noexcept {
  return reinterpret_cast<cudaStream_t>(handle.value);
}

[[nodiscard]] inline void* native_address(DeviceAddress address) noexcept {
  return reinterpret_cast<void*>(address.value);
}

[[nodiscard]] inline Status runtime_status(cudaError_t error) noexcept {
  if (error == cudaSuccess) {
    return Status::success();
  }
  return Status::failure(ErrorCode::cuda_runtime_failure,
                         static_cast<std::uint32_t>(error));
}

[[nodiscard]] inline Status launch_status(cudaError_t error) noexcept {
  if (error == cudaSuccess) {
    return Status::success();
  }
  return Status::failure(ErrorCode::cuda_backend_failure,
                         static_cast<std::uint32_t>(error));
}

} // namespace marketforge::cuda::detail
