#include "marketforge/cuda/cublas_handle.hpp"

#include <utility>

#include <cublas_v2.h>

#include "cublas_internal.hpp"

namespace marketforge::cuda {

Result<CublasHandle> CublasHandle::create() noexcept {
  cublasHandle_t handle = nullptr;
  const auto status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return Result<CublasHandle>::failure(detail::cublas_status(status));
  }
  const auto pointer_mode =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (pointer_mode != CUBLAS_STATUS_SUCCESS) {
    (void)cublasDestroy(handle);
    return Result<CublasHandle>::failure(
        detail::cublas_status(pointer_mode));
  }
  return Result<CublasHandle>::success(CublasHandle{
      CublasHandleValue{reinterpret_cast<std::uintptr_t>(handle)}});
}

CublasHandle::CublasHandle(CublasHandle&& other) noexcept
    : handle_(std::exchange(other.handle_, {})) {}

CublasHandle& CublasHandle::operator=(CublasHandle&& other) noexcept {
  if (this != &other) {
    release();
    handle_ = std::exchange(other.handle_, {});
  }
  return *this;
}

CublasHandle::~CublasHandle() noexcept { release(); }

void CublasHandle::release() noexcept {
  if (!handle_.valid()) {
    return;
  }
  const auto handle = detail::native_cublas_handle(handle_);
  handle_ = {};
  (void)cublasDestroy(handle);
}

} // namespace marketforge::cuda
