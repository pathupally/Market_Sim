#include "marketforge/model/safetensors.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "json_parser.hpp"
#include "marketforge/core/shape.hpp"

namespace marketforge {
namespace {

Result<SafeTensorMetadata> fail(const ErrorCode code,
                                const std::uint32_t detail = 0) {
  return Result<SafeTensorMetadata>::failure(Status::failure(code, detail));
}

std::uint64_t
load_u64_little_endian(const std::span<const std::byte, 8> bytes) noexcept {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    value |=
        static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[index]))
        << (index * 8U);
  }
  return value;
}

Result<std::uint64_t> parse_unsigned(const detail::JsonValue& value) {
  if (value.kind != detail::JsonKind::number || value.text.empty() ||
      value.text.front() == '-' ||
      value.text.find_first_of(".eE") != std::string::npos) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::invalid_tensor));
  }
  std::uint64_t parsed = 0;
  const auto conversion = std::from_chars(
      value.text.data(), value.text.data() + value.text.size(), parsed);
  if (conversion.ec != std::errc{} ||
      conversion.ptr != value.text.data() + value.text.size()) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  return Result<std::uint64_t>::success(parsed);
}

Result<DType> parse_dtype(const detail::JsonValue& value) {
  if (value.kind != detail::JsonKind::string) {
    return Result<DType>::failure(Status::failure(ErrorCode::invalid_tensor));
  }
  if (value.text == "F32") {
    return Result<DType>::success(DType::f32);
  }
  if (value.text == "F16") {
    return Result<DType>::success(DType::f16);
  }
  if (value.text == "BF16") {
    return Result<DType>::success(DType::bf16);
  }
  if (value.text == "I8") {
    return Result<DType>::success(DType::i8);
  }
  if (value.text == "I32") {
    return Result<DType>::success(DType::i32);
  }
  if (value.text == "U32") {
    return Result<DType>::success(DType::u32);
  }
  return Result<DType>::failure(Status::failure(ErrorCode::unsupported_dtype));
}

Status validate_metadata_object(const detail::JsonValue& value) {
  if (value.kind != detail::JsonKind::object) {
    return Status::failure(ErrorCode::invalid_tensor);
  }
  for (const auto& [key, member] : value.object) {
    static_cast<void>(key);
    if (member->kind != detail::JsonKind::string) {
      return Status::failure(ErrorCode::invalid_tensor);
    }
  }
  return Status::success();
}

Result<TensorRecord> parse_tensor_record(const std::string& name,
                                         const detail::JsonValue& value,
                                         const std::uint64_t data_size) {
  if (name.empty() || value.kind != detail::JsonKind::object ||
      value.object.size() != 3) {
    return Result<TensorRecord>::failure(
        Status::failure(ErrorCode::invalid_tensor));
  }
  const auto* const dtype_value = value.find("dtype");
  const auto* const shape_value = value.find("shape");
  const auto* const offsets_value = value.find("data_offsets");
  if (dtype_value == nullptr || shape_value == nullptr ||
      offsets_value == nullptr ||
      shape_value->kind != detail::JsonKind::array ||
      offsets_value->kind != detail::JsonKind::array ||
      offsets_value->array.size() != 2 ||
      shape_value->array.size() > Shape::max_rank) {
    return Result<TensorRecord>::failure(
        Status::failure(ErrorCode::invalid_tensor));
  }

  const auto dtype = parse_dtype(*dtype_value);
  if (!dtype) {
    return Result<TensorRecord>::failure(dtype.status());
  }

  std::array<std::uint64_t, Shape::max_rank> extents{};
  for (std::size_t axis = 0; axis < shape_value->array.size(); ++axis) {
    const auto extent = parse_unsigned(shape_value->array[axis]);
    if (!extent) {
      return Result<TensorRecord>::failure(extent.status());
    }
    extents[axis] = extent.value();
  }
  const auto shape = make_shape(std::span<const std::uint64_t>(
      extents.data(), shape_value->array.size()));
  if (!shape) {
    return Result<TensorRecord>::failure(shape.status());
  }

  const auto begin = parse_unsigned(offsets_value->array[0]);
  const auto end = parse_unsigned(offsets_value->array[1]);
  if (!begin || !end) {
    return Result<TensorRecord>::failure(!begin ? begin.status()
                                                : end.status());
  }
  if (end.value() < begin.value() || end.value() > data_size) {
    return Result<TensorRecord>::failure(
        Status::failure(ErrorCode::invalid_tensor));
  }

  const auto expected_bytes = checked_nbytes(shape.value(), dtype.value());
  if (!expected_bytes) {
    return Result<TensorRecord>::failure(expected_bytes.status());
  }
  if (expected_bytes.value() != end.value() - begin.value()) {
    return Result<TensorRecord>::failure(
        Status::failure(ErrorCode::invalid_tensor));
  }

  return Result<TensorRecord>::success(TensorRecord{
      name,
      dtype.value(),
      shape.value(),
      begin.value(),
      end.value(),
  });
}

Status validate_exact_coverage(const std::vector<TensorRecord>& records,
                               const std::uint64_t data_size) {
  std::vector<const TensorRecord*> ordered;
  ordered.reserve(records.size());
  for (const auto& record : records) {
    ordered.push_back(&record);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const TensorRecord* left, const TensorRecord* right) {
              if (left->begin != right->begin) {
                return left->begin < right->begin;
              }
              return left->end < right->end;
            });

  std::uint64_t expected_begin = 0;
  for (const auto* const record : ordered) {
    if (record->begin != expected_begin) {
      return Status::failure(ErrorCode::invalid_tensor);
    }
    expected_begin = record->end;
  }
  if (expected_begin != data_size) {
    return Status::failure(ErrorCode::invalid_tensor);
  }
  return Status::success();
}

} // namespace

Result<SafeTensorMetadata>
parse_safetensors_metadata(const std::span<const std::byte> bytes,
                           const std::uint64_t maximum_header_bytes) {
  if (bytes.size() < 8) {
    return fail(ErrorCode::truncated_data);
  }
  const auto prefix = std::span<const std::byte, 8>(bytes.first<8>());
  const std::uint64_t header_size = load_u64_little_endian(prefix);
  if (header_size > maximum_header_bytes ||
      header_size > std::numeric_limits<std::size_t>::max()) {
    return fail(ErrorCode::resource_limit);
  }

  const auto host_header_size = static_cast<std::size_t>(header_size);
  if (host_header_size > bytes.size() - 8) {
    return fail(ErrorCode::truncated_data);
  }
  const std::uint64_t data_start = 8 + header_size;
  const std::uint64_t data_size =
      static_cast<std::uint64_t>(bytes.size()) - data_start;
  const auto header_bytes = bytes.subspan(8, host_header_size);
  const std::string_view header(
      reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());
  if (header.empty() || header.front() != '{') {
    return fail(ErrorCode::invalid_json);
  }

  const auto parsed = detail::parse_json(header);
  if (!parsed) {
    return fail(parsed.status().code, parsed.status().detail);
  }
  if (parsed.value().kind != detail::JsonKind::object) {
    return fail(ErrorCode::invalid_json);
  }

  SafeTensorMetadata metadata;
  metadata.data_start = data_start;
  metadata.records.reserve(parsed.value().object.size());
  for (const auto& [name, value] : parsed.value().object) {
    if (name == "__metadata__") {
      const auto status = validate_metadata_object(*value);
      if (!status.ok()) {
        return fail(status.code, status.detail);
      }
      continue;
    }
    auto record = parse_tensor_record(name, *value, data_size);
    if (!record) {
      return fail(record.status().code, record.status().detail);
    }
    metadata.records.push_back(std::move(record).value());
  }

  const auto coverage = validate_exact_coverage(metadata.records, data_size);
  if (!coverage.ok()) {
    return fail(coverage.code, coverage.detail);
  }
  return Result<SafeTensorMetadata>::success(std::move(metadata));
}

Result<SafeTensorFile>
SafeTensorFile::parse(MappedFile mapping,
                      const std::uint64_t maximum_header_bytes) {
  auto metadata =
      parse_safetensors_metadata(mapping.bytes(), maximum_header_bytes);
  if (!metadata) {
    return Result<SafeTensorFile>::failure(metadata.status());
  }
  return Result<SafeTensorFile>::success(
      SafeTensorFile{std::move(mapping), std::move(metadata).value()});
}

const TensorRecord*
SafeTensorFile::record(const std::string_view name) const noexcept {
  for (const auto& candidate : metadata_.records) {
    if (candidate.name == name) {
      return &candidate;
    }
  }
  return nullptr;
}

Result<ConstTensorView>
SafeTensorFile::tensor(const std::string_view name) const noexcept {
  const TensorRecord* const found = record(name);
  if (found == nullptr) {
    return Result<ConstTensorView>::failure(
        Status::failure(ErrorCode::missing_tensor));
  }
  const auto mapping = mapping_.bytes();
  const std::uint64_t absolute_offset = metadata_.data_start + found->begin;
  if (absolute_offset > mapping.size()) {
    return Result<ConstTensorView>::failure(
        Status::failure(ErrorCode::invalid_tensor));
  }
  const auto* const data =
      mapping.data() + static_cast<std::size_t>(absolute_offset);
  return Result<ConstTensorView>::success(ConstTensorView{
      data,
      found->shape,
      found->dtype,
      MemoryKind::host,
  });
}

} // namespace marketforge
