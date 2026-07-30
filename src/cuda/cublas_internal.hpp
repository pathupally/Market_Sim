#pragma once

#include <cstdint>

#include <cublas_v2.h>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cublas_handle.hpp"

namespace marketforge::cuda::detail {

[[nodiscard]] inline cublasHandle_t
native_cublas_handle(CublasHandleValue handle) noexcept {
  return reinterpret_cast<cublasHandle_t>(handle.value);
}

[[nodiscard]] inline Status cublas_status(cublasStatus_t status) noexcept {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::success();
  }
  return Status::failure(ErrorCode::cuda_backend_failure,
                         static_cast<std::uint32_t>(status));
}

} // namespace marketforge::cuda::detail
