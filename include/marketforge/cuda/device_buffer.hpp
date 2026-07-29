#pragma once

#include <cstdint>

#include "marketforge/core/result.hpp"
#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"

namespace marketforge::cuda {

struct DeviceAddress {
  std::uintptr_t value{0};

  [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }

  friend constexpr bool operator==(const DeviceAddress&,
                                   const DeviceAddress&) = default;
};

class DeviceBuffer {
public:
  DeviceBuffer() noexcept = default;

  [[nodiscard]] static Result<DeviceBuffer>
  allocate(std::uint64_t byte_count) noexcept;

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  ~DeviceBuffer() noexcept;

  [[nodiscard]] std::uint64_t size_bytes() const noexcept {
    return size_bytes_;
  }
  [[nodiscard]] DeviceAddress address() const noexcept { return address_; }

  [[nodiscard]] Status copy_from_host_async(const void* source,
                                            std::uint64_t byte_count,
                                            std::uint64_t offset,
                                            StreamHandle stream) noexcept;
  [[nodiscard]] Status copy_to_host_async(void* destination,
                                          std::uint64_t byte_count,
                                          std::uint64_t offset,
                                          StreamHandle stream) const noexcept;

private:
  DeviceBuffer(DeviceAddress address, std::uint64_t byte_count) noexcept
      : address_(address), size_bytes_(byte_count) {}
  void release() noexcept;

  DeviceAddress address_{};
  std::uint64_t size_bytes_{0};
};

} // namespace marketforge::cuda
