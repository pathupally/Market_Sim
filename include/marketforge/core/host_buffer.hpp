#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "marketforge/core/result.hpp"

namespace marketforge {

class HostBuffer {
public:
  [[nodiscard]] static Result<HostBuffer>
  allocate(std::uint64_t bytes, std::uint64_t alignment) noexcept;

  HostBuffer() = default;
  ~HostBuffer();

  HostBuffer(HostBuffer&& other) noexcept;
  HostBuffer& operator=(HostBuffer&& other) noexcept;

  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;

  [[nodiscard]] std::span<std::byte> bytes() noexcept;
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
  [[nodiscard]] std::uint64_t alignment() const noexcept;

private:
  HostBuffer(std::byte* data, std::size_t size, std::size_t alignment) noexcept;

  void reset() noexcept;

  std::byte* data_{nullptr};
  std::size_t size_{0};
  std::size_t alignment_{alignof(std::max_align_t)};
};

} // namespace marketforge
