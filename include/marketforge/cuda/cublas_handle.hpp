#pragma once

#include <cstdint>

#include "marketforge/core/result.hpp"

namespace marketforge::cuda {

struct CublasHandleValue {
  std::uintptr_t value{0};

  [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }

  friend constexpr bool operator==(const CublasHandleValue&,
                                   const CublasHandleValue&) = default;
};

class CublasHandle {
public:
  CublasHandle() noexcept = default;

  [[nodiscard]] static Result<CublasHandle> create() noexcept;

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  CublasHandle(CublasHandle&& other) noexcept;
  CublasHandle& operator=(CublasHandle&& other) noexcept;

  ~CublasHandle() noexcept;

  [[nodiscard]] CublasHandleValue handle() const noexcept { return handle_; }

private:
  explicit CublasHandle(CublasHandleValue handle) noexcept : handle_(handle) {}
  void release() noexcept;

  CublasHandleValue handle_{};
};

} // namespace marketforge::cuda
