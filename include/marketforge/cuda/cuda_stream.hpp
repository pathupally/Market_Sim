#pragma once

#include <cstdint>

#include "marketforge/core/result.hpp"
#include "marketforge/core/status.hpp"

namespace marketforge::cuda {

struct StreamHandle {
  std::uintptr_t value{0};

  [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }

  friend constexpr bool operator==(const StreamHandle&,
                                   const StreamHandle&) = default;
};

class CudaStream {
public:
  CudaStream() noexcept = default;

  [[nodiscard]] static Result<CudaStream> create() noexcept;

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;

  ~CudaStream() noexcept;

  [[nodiscard]] StreamHandle handle() const noexcept { return handle_; }
  [[nodiscard]] Status synchronize() const noexcept;

private:
  explicit CudaStream(StreamHandle handle) noexcept : handle_(handle) {}
  void release() noexcept;

  StreamHandle handle_{};
};

} // namespace marketforge::cuda
