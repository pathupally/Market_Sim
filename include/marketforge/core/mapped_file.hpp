#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include "marketforge/core/result.hpp"

namespace marketforge {

class MappedFile {
public:
  [[nodiscard]] static Result<MappedFile>
  open_read_only(const std::filesystem::path& path) noexcept;

  MappedFile() noexcept = default;
  ~MappedFile();

  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return {data_, size_};
  }

private:
  MappedFile(const std::byte* data, std::size_t size) noexcept
      : data_(data), size_(size) {}

  void reset() noexcept;

  const std::byte* data_{nullptr};
  std::size_t size_{0};
};

} // namespace marketforge
