#include "json_parser.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

namespace marketforge::detail {
namespace {

constexpr std::size_t maximum_depth = 64;

class Parser {
public:
  explicit Parser(std::string_view input) : input_(input) {}

  Result<JsonValue> parse() {
    if (!is_valid_utf8(input_)) {
      return fail(ErrorCode::invalid_utf8);
    }
    auto value = parse_value(0);
    if (!value) {
      return value;
    }
    skip_whitespace();
    if (position_ != input_.size()) {
      return fail(ErrorCode::invalid_json);
    }
    return value;
  }

private:
  Result<JsonValue> fail(const ErrorCode code) const {
    return Result<JsonValue>::failure(Status::failure(
        code, static_cast<std::uint32_t>(position_ > UINT32_MAX ? UINT32_MAX
                                                                : position_)));
  }

  void skip_whitespace() noexcept {
    while (position_ < input_.size()) {
      const char value = input_[position_];
      if (value != ' ' && value != '\n' && value != '\r' && value != '\t') {
        return;
      }
      ++position_;
    }
  }

  bool consume(const char expected) noexcept {
    skip_whitespace();
    if (position_ >= input_.size() || input_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  bool consume_literal(const std::string_view literal) noexcept {
    if (input_.substr(position_, literal.size()) != literal) {
      return false;
    }
    position_ += literal.size();
    return true;
  }

  Result<JsonValue> parse_value(const std::size_t depth) {
    if (depth > maximum_depth) {
      return fail(ErrorCode::resource_limit);
    }
    skip_whitespace();
    if (position_ >= input_.size()) {
      return fail(ErrorCode::invalid_json);
    }

    switch (input_[position_]) {
    case '{':
      return parse_object(depth + 1);
    case '[':
      return parse_array(depth + 1);
    case '"': {
      auto value = parse_string();
      if (!value) {
        return Result<JsonValue>::failure(value.status());
      }
      JsonValue result;
      result.kind = JsonKind::string;
      result.text = std::move(value).value();
      return Result<JsonValue>::success(std::move(result));
    }
    case 't':
      if (consume_literal("true")) {
        JsonValue value;
        value.kind = JsonKind::boolean;
        value.boolean = true;
        return Result<JsonValue>::success(std::move(value));
      }
      return fail(ErrorCode::invalid_json);
    case 'f':
      if (consume_literal("false")) {
        JsonValue value;
        value.kind = JsonKind::boolean;
        return Result<JsonValue>::success(std::move(value));
      }
      return fail(ErrorCode::invalid_json);
    case 'n':
      if (consume_literal("null")) {
        return Result<JsonValue>::success(JsonValue{});
      }
      return fail(ErrorCode::invalid_json);
    default:
      return parse_number();
    }
  }

  Result<JsonValue> parse_object(const std::size_t depth) {
    if (!consume('{')) {
      return fail(ErrorCode::invalid_json);
    }
    JsonValue value;
    value.kind = JsonKind::object;
    std::unordered_set<std::string> keys;

    skip_whitespace();
    if (consume('}')) {
      return Result<JsonValue>::success(std::move(value));
    }
    while (true) {
      skip_whitespace();
      if (position_ >= input_.size() || input_[position_] != '"') {
        return fail(ErrorCode::invalid_json);
      }
      auto key = parse_string();
      if (!key) {
        return Result<JsonValue>::failure(key.status());
      }
      if (!keys.insert(key.value()).second) {
        return fail(ErrorCode::duplicate_key);
      }
      if (!consume(':')) {
        return fail(ErrorCode::invalid_json);
      }
      auto member = parse_value(depth);
      if (!member) {
        return member;
      }
      value.object.emplace_back(
          std::move(key).value(),
          std::make_unique<JsonValue>(std::move(member).value()));
      skip_whitespace();
      if (consume('}')) {
        return Result<JsonValue>::success(std::move(value));
      }
      if (!consume(',')) {
        return fail(ErrorCode::invalid_json);
      }
    }
  }

  Result<JsonValue> parse_array(const std::size_t depth) {
    if (!consume('[')) {
      return fail(ErrorCode::invalid_json);
    }
    JsonValue value;
    value.kind = JsonKind::array;
    skip_whitespace();
    if (consume(']')) {
      return Result<JsonValue>::success(std::move(value));
    }
    while (true) {
      auto element = parse_value(depth);
      if (!element) {
        return element;
      }
      value.array.push_back(std::move(element).value());
      skip_whitespace();
      if (consume(']')) {
        return Result<JsonValue>::success(std::move(value));
      }
      if (!consume(',')) {
        return fail(ErrorCode::invalid_json);
      }
    }
  }

  Result<JsonValue> parse_number() {
    const std::size_t start = position_;
    if (position_ < input_.size() && input_[position_] == '-') {
      ++position_;
    }
    if (position_ >= input_.size()) {
      return fail(ErrorCode::invalid_json);
    }
    if (input_[position_] == '0') {
      ++position_;
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') {
        return fail(ErrorCode::invalid_json);
      }
    } else if (input_[position_] >= '1' && input_[position_] <= '9') {
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    } else {
      return fail(ErrorCode::invalid_json);
    }

    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      const std::size_t fraction_start = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (fraction_start == position_) {
        return fail(ErrorCode::invalid_json);
      }
    }

    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent_start = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (exponent_start == position_) {
        return fail(ErrorCode::invalid_json);
      }
    }

    JsonValue value;
    value.kind = JsonKind::number;
    value.text = std::string(input_.substr(start, position_ - start));
    return Result<JsonValue>::success(std::move(value));
  }

  static bool is_hex(const char value) noexcept {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
           (value >= 'A' && value <= 'F');
  }

  static std::uint32_t hex_value(const char value) noexcept {
    if (value >= '0' && value <= '9') {
      return static_cast<std::uint32_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
      return static_cast<std::uint32_t>(value - 'a' + 10);
    }
    return static_cast<std::uint32_t>(value - 'A' + 10);
  }

  bool parse_hex_quad(std::uint32_t& value) noexcept {
    if (input_.size() - position_ < 4) {
      return false;
    }
    value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
      const char digit = input_[position_ + index];
      if (!is_hex(digit)) {
        return false;
      }
      value = value * 16 + hex_value(digit);
    }
    position_ += 4;
    return true;
  }

  static void append_utf8(std::string& output, const std::uint32_t code_point) {
    if (code_point <= 0x7fU) {
      output.push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7ffU) {
      output.push_back(static_cast<char>(0xc0U | (code_point >> 6U)));
      output.push_back(static_cast<char>(0x80U | (code_point & 0x3fU)));
    } else if (code_point <= 0xffffU) {
      output.push_back(static_cast<char>(0xe0U | (code_point >> 12U)));
      output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (code_point & 0x3fU)));
    } else {
      output.push_back(static_cast<char>(0xf0U | (code_point >> 18U)));
      output.push_back(
          static_cast<char>(0x80U | ((code_point >> 12U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (code_point & 0x3fU)));
    }
  }

  Result<std::string> parse_string() {
    if (position_ >= input_.size() || input_[position_] != '"') {
      return Result<std::string>::failure(
          Status::failure(ErrorCode::invalid_json));
    }
    ++position_;
    std::string output;
    while (position_ < input_.size()) {
      const unsigned char value =
          static_cast<unsigned char>(input_[position_++]);
      if (value == '"') {
        return Result<std::string>::success(std::move(output));
      }
      if (value < 0x20U) {
        return Result<std::string>::failure(
            Status::failure(ErrorCode::invalid_json,
                            static_cast<std::uint32_t>(position_ - 1)));
      }
      if (value != '\\') {
        output.push_back(static_cast<char>(value));
        continue;
      }
      if (position_ >= input_.size()) {
        break;
      }
      const char escape = input_[position_++];
      switch (escape) {
      case '"':
      case '\\':
      case '/':
        output.push_back(escape);
        break;
      case 'b':
        output.push_back('\b');
        break;
      case 'f':
        output.push_back('\f');
        break;
      case 'n':
        output.push_back('\n');
        break;
      case 'r':
        output.push_back('\r');
        break;
      case 't':
        output.push_back('\t');
        break;
      case 'u': {
        std::uint32_t code_point = 0;
        if (!parse_hex_quad(code_point)) {
          return Result<std::string>::failure(
              Status::failure(ErrorCode::invalid_json));
        }
        if (code_point >= 0xd800U && code_point <= 0xdbffU) {
          if (input_.size() - position_ < 6 || input_[position_] != '\\' ||
              input_[position_ + 1] != 'u') {
            return Result<std::string>::failure(
                Status::failure(ErrorCode::invalid_json));
          }
          position_ += 2;
          std::uint32_t low = 0;
          if (!parse_hex_quad(low) || low < 0xdc00U || low > 0xdfffU) {
            return Result<std::string>::failure(
                Status::failure(ErrorCode::invalid_json));
          }
          code_point =
              0x10000U + ((code_point - 0xd800U) << 10U) + (low - 0xdc00U);
        } else if (code_point >= 0xdc00U && code_point <= 0xdfffU) {
          return Result<std::string>::failure(
              Status::failure(ErrorCode::invalid_json));
        }
        append_utf8(output, code_point);
        break;
      }
      default:
        return Result<std::string>::failure(
            Status::failure(ErrorCode::invalid_json));
      }
    }
    return Result<std::string>::failure(
        Status::failure(ErrorCode::invalid_json));
  }

  std::string_view input_;
  std::size_t position_{0};
};

} // namespace

const JsonValue* JsonValue::find(const std::string_view key) const noexcept {
  if (kind != JsonKind::object) {
    return nullptr;
  }
  for (const auto& [name, value] : object) {
    if (name == key) {
      return value.get();
    }
  }
  return nullptr;
}

Result<JsonValue> parse_json(const std::string_view input) {
  return Parser(input).parse();
}

bool is_valid_utf8(const std::string_view input) noexcept {
  std::size_t index = 0;
  while (index < input.size()) {
    const auto first = static_cast<unsigned char>(input[index]);
    if (first <= 0x7fU) {
      ++index;
      continue;
    }

    std::size_t continuation_count = 0;
    std::uint32_t code_point = 0;
    if (first >= 0xc2U && first <= 0xdfU) {
      continuation_count = 1;
      code_point = first & 0x1fU;
    } else if (first >= 0xe0U && first <= 0xefU) {
      continuation_count = 2;
      code_point = first & 0x0fU;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      continuation_count = 3;
      code_point = first & 0x07U;
    } else {
      return false;
    }

    if (input.size() - index - 1 < continuation_count) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
      const auto next = static_cast<unsigned char>(input[index + offset]);
      if ((next & 0xc0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (next & 0x3fU);
    }

    if ((continuation_count == 2 &&
         (code_point < 0x800U ||
          (code_point >= 0xd800U && code_point <= 0xdfffU))) ||
        (continuation_count == 3 &&
         (code_point < 0x10000U || code_point > 0x10ffffU))) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

} // namespace marketforge::detail
