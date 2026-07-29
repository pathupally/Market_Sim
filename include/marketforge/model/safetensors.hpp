#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "marketforge/core/mapped_file.hpp"
#include "marketforge/core/tensor_view.hpp"

namespace marketforge {

inline constexpr std::uint64_t default_max_safetensors_header_bytes =
    16ULL * 1024ULL * 1024ULL;

struct TensorRecord {
  std::string name;
  DType dtype{DType::f32};
  Shape shape{};
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

struct SafeTensorMetadata {
  std::uint64_t data_start{0};
  std::vector<TensorRecord> records;
};

[[nodiscard]] Result<SafeTensorMetadata> parse_safetensors_metadata(
    std::span<const std::byte> bytes,
    std::uint64_t maximum_header_bytes = default_max_safetensors_header_bytes);

class SafeTensorFile {
public:
  [[nodiscard]] static Result<SafeTensorFile>
  parse(MappedFile mapping, std::uint64_t maximum_header_bytes =
                                default_max_safetensors_header_bytes);

  SafeTensorFile(SafeTensorFile&&) noexcept = default;
  SafeTensorFile& operator=(SafeTensorFile&&) noexcept = default;
  SafeTensorFile(const SafeTensorFile&) = delete;
  SafeTensorFile& operator=(const SafeTensorFile&) = delete;

  [[nodiscard]] Result<ConstTensorView>
  tensor(std::string_view name) const noexcept;

  [[nodiscard]] const TensorRecord*
  record(std::string_view name) const noexcept;

  [[nodiscard]] std::span<const TensorRecord> records() const noexcept {
    return metadata_.records;
  }

private:
  SafeTensorFile(MappedFile mapping, SafeTensorMetadata metadata) noexcept
      : mapping_(std::move(mapping)), metadata_(std::move(metadata)) {}

  MappedFile mapping_;
  SafeTensorMetadata metadata_;
};

} // namespace marketforge
