#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "marketforge/core/result.hpp"

namespace marketforge::detail {

enum class JsonKind {
  null_value,
  boolean,
  number,
  string,
  array,
  object,
};

struct JsonValue {
  JsonKind kind{JsonKind::null_value};
  bool boolean{false};
  std::string text;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, std::unique_ptr<JsonValue>>> object;

  [[nodiscard]] const JsonValue* find(std::string_view key) const noexcept;
};

[[nodiscard]] Result<JsonValue> parse_json(std::string_view input);
[[nodiscard]] bool is_valid_utf8(std::string_view input) noexcept;

} // namespace marketforge::detail
