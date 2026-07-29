#include "marketforge/cuda/cuda_stream.hpp"

#include <utility>

#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {

Result<CudaStream> CudaStream::create() noexcept {
  cudaStream_t stream = nullptr;
  const auto status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    return Result<CudaStream>::failure(detail::runtime_status(status));
  }
  return Result<CudaStream>::success(
      CudaStream(StreamHandle{reinterpret_cast<std::uintptr_t>(stream)}));
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : handle_(std::exchange(other.handle_, {})) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    release();
    handle_ = std::exchange(other.handle_, {});
  }
  return *this;
}

CudaStream::~CudaStream() noexcept { release(); }

Status CudaStream::synchronize() const noexcept {
  if (!handle_.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  return detail::runtime_status(
      cudaStreamSynchronize(detail::native_stream(handle_)));
}

void CudaStream::release() noexcept {
  if (!handle_.valid()) {
    return;
  }
  const auto stream = detail::native_stream(handle_);
  handle_ = {};
  (void)cudaStreamDestroy(stream);
}

} // namespace marketforge::cuda
