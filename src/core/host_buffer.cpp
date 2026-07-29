#include "marketforge/core/host_buffer.hpp"

#include <limits>
#include <new>
#include <utility>

namespace marketforge {
namespace {

[[nodiscard]] constexpr bool
is_power_of_two(const std::uint64_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

Result<HostBuffer>
HostBuffer::allocate(const std::uint64_t bytes,
                     const std::uint64_t alignment) noexcept {
  if (!is_power_of_two(alignment) || alignment < alignof(void*) ||
      alignment > std::numeric_limits<std::size_t>::max()) {
    return Result<HostBuffer>::failure(
        Status::failure(ErrorCode::invalid_alignment));
  }
  if (bytes > std::numeric_limits<std::size_t>::max()) {
    return Result<HostBuffer>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }

  const auto native_size = static_cast<std::size_t>(bytes);
  const auto native_alignment = static_cast<std::size_t>(alignment);
  if (native_size == 0) {
    return Result<HostBuffer>::success(
        HostBuffer(nullptr, 0, native_alignment));
  }

  void* allocation = ::operator new(
      native_size, std::align_val_t(native_alignment), std::nothrow);
  if (allocation == nullptr) {
    return Result<HostBuffer>::failure(
        Status::failure(ErrorCode::allocation_failed));
  }

  return Result<HostBuffer>::success(HostBuffer(
      static_cast<std::byte*>(allocation), native_size, native_alignment));
}

HostBuffer::HostBuffer(std::byte* data, const std::size_t size,
                       const std::size_t alignment) noexcept
    : data_(data), size_(size), alignment_(alignment) {}

HostBuffer::~HostBuffer() { reset(); }

HostBuffer::HostBuffer(HostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      alignment_(std::exchange(other.alignment_, alignof(std::max_align_t))) {}

HostBuffer& HostBuffer::operator=(HostBuffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  data_ = std::exchange(other.data_, nullptr);
  size_ = std::exchange(other.size_, 0);
  alignment_ = std::exchange(other.alignment_, alignof(std::max_align_t));
  return *this;
}

std::span<std::byte> HostBuffer::bytes() noexcept { return {data_, size_}; }

std::span<const std::byte> HostBuffer::bytes() const noexcept {
  return {data_, size_};
}

std::uint64_t HostBuffer::alignment() const noexcept { return alignment_; }

void HostBuffer::reset() noexcept {
  if (data_ != nullptr) {
    ::operator delete(data_, std::align_val_t(alignment_));
  }
  data_ = nullptr;
  size_ = 0;
  alignment_ = alignof(std::max_align_t);
}

} // namespace marketforge
