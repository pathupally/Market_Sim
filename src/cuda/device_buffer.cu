#include "marketforge/cuda/device_buffer.hpp"

#include <cstddef>
#include <limits>
#include <utility>

#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"

namespace marketforge::cuda {
namespace {

[[nodiscard]] Status validate_copy(const void* host_pointer,
                                   std::uint64_t byte_count,
                                   std::uint64_t offset,
                                   std::uint64_t allocation_size,
                                   StreamHandle stream) noexcept {
  if (!stream.valid() || (byte_count != 0 && host_pointer == nullptr)) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (offset > std::numeric_limits<std::uint64_t>::max() - byte_count) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  if (offset + byte_count > allocation_size) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (byte_count > std::numeric_limits<std::size_t>::max()) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  return Status::success();
}

} // namespace

Result<DeviceBuffer>
DeviceBuffer::allocate(const std::uint64_t byte_count) noexcept {
  if (byte_count == 0) {
    return Result<DeviceBuffer>::success(DeviceBuffer{});
  }
  if (byte_count > std::numeric_limits<std::size_t>::max()) {
    return Result<DeviceBuffer>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }

  void* allocation = nullptr;
  const auto status =
      cudaMalloc(&allocation, static_cast<std::size_t>(byte_count));
  if (status != cudaSuccess) {
    return Result<DeviceBuffer>::failure(detail::runtime_status(status));
  }
  return Result<DeviceBuffer>::success(DeviceBuffer(
      DeviceAddress{reinterpret_cast<std::uintptr_t>(allocation)}, byte_count));
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : address_(std::exchange(other.address_, {})),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    address_ = std::exchange(other.address_, {});
    size_bytes_ = std::exchange(other.size_bytes_, 0);
  }
  return *this;
}

DeviceBuffer::~DeviceBuffer() noexcept { release(); }

Status DeviceBuffer::copy_from_host_async(const void* source,
                                          const std::uint64_t byte_count,
                                          const std::uint64_t offset,
                                          const StreamHandle stream) noexcept {
  const auto validation =
      validate_copy(source, byte_count, offset, size_bytes_, stream);
  if (!validation.ok() || byte_count == 0) {
    return validation;
  }

  auto* destination = static_cast<std::byte*>(detail::native_address(address_));
  return detail::runtime_status(
      cudaMemcpyAsync(destination + static_cast<std::size_t>(offset), source,
                      static_cast<std::size_t>(byte_count),
                      cudaMemcpyHostToDevice, detail::native_stream(stream)));
}

Status DeviceBuffer::copy_to_host_async(
    void* destination, const std::uint64_t byte_count,
    const std::uint64_t offset, const StreamHandle stream) const noexcept {
  const auto validation =
      validate_copy(destination, byte_count, offset, size_bytes_, stream);
  if (!validation.ok() || byte_count == 0) {
    return validation;
  }

  const auto* source =
      static_cast<const std::byte*>(detail::native_address(address_));
  return detail::runtime_status(
      cudaMemcpyAsync(destination, source + static_cast<std::size_t>(offset),
                      static_cast<std::size_t>(byte_count),
                      cudaMemcpyDeviceToHost, detail::native_stream(stream)));
}

void DeviceBuffer::release() noexcept {
  if (!address_.valid()) {
    size_bytes_ = 0;
    return;
  }
  auto* allocation = detail::native_address(address_);
  address_ = {};
  size_bytes_ = 0;
  (void)cudaFree(allocation);
}

} // namespace marketforge::cuda
