#include "marketforge/model/model_config.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>

#include "json_parser.hpp"
#include "marketforge/core/mapped_file.hpp"

namespace marketforge {
namespace {

template <typename T>
Result<T> config_failure(const ErrorCode code = ErrorCode::invalid_model) {
  return Result<T>::failure(Status::failure(code));
}

Result<std::uint32_t> required_u32(const detail::JsonValue& root,
                                   const std::string_view key) {
  const auto* const value = root.find(key);
  if (value == nullptr || value->kind != detail::JsonKind::number ||
      value->text.empty() || value->text.front() == '-' ||
      value->text.find_first_of(".eE") != std::string::npos) {
    return config_failure<std::uint32_t>();
  }
  std::uint64_t parsed = 0;
  const auto conversion = std::from_chars(
      value->text.data(), value->text.data() + value->text.size(), parsed);
  if (conversion.ec != std::errc{} ||
      conversion.ptr != value->text.data() + value->text.size() ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    return config_failure<std::uint32_t>(ErrorCode::arithmetic_overflow);
  }
  return Result<std::uint32_t>::success(static_cast<std::uint32_t>(parsed));
}

Result<float> required_float(const detail::JsonValue& root,
                             const std::string_view key) {
  const auto* const value = root.find(key);
  if (value == nullptr || value->kind != detail::JsonKind::number) {
    return config_failure<float>();
  }
  double parsed = 0.0;
  std::istringstream input(value->text);
  input.imbue(std::locale::classic());
  input >> parsed;
  if (!input || input.peek() != std::char_traits<char>::eof() ||
      !std::isfinite(parsed) || parsed > std::numeric_limits<float>::max() ||
      parsed < -std::numeric_limits<float>::max()) {
    return config_failure<float>();
  }
  return Result<float>::success(static_cast<float>(parsed));
}

Result<bool> required_bool(const detail::JsonValue& root,
                           const std::string_view key) {
  const auto* const value = root.find(key);
  if (value == nullptr || value->kind != detail::JsonKind::boolean) {
    return config_failure<bool>();
  }
  return Result<bool>::success(value->boolean);
}

Result<std::string_view> required_string(const detail::JsonValue& root,
                                         const std::string_view key) {
  const auto* const value = root.find(key);
  if (value == nullptr || value->kind != detail::JsonKind::string) {
    return config_failure<std::string_view>();
  }
  return Result<std::string_view>::success(value->text);
}

Status validate_architecture_name(const detail::JsonValue& root,
                                  const std::string_view expected) {
  const auto* const architectures = root.find("architectures");
  if (architectures == nullptr ||
      architectures->kind != detail::JsonKind::array ||
      architectures->array.size() != 1 ||
      architectures->array[0].kind != detail::JsonKind::string ||
      architectures->array[0].text != expected) {
    return Status::failure(ErrorCode::invalid_model);
  }
  return Status::success();
}

} // namespace

Result<ModelSpec> parse_model_config(const std::string_view json_text) {
  const auto root = detail::parse_json(json_text);
  if (!root) {
    return Result<ModelSpec>::failure(root.status());
  }
  if (root.value().kind != detail::JsonKind::object) {
    return config_failure<ModelSpec>();
  }

  const auto model_type = required_string(root.value(), "model_type");
  if (!model_type) {
    return config_failure<ModelSpec>();
  }

  Architecture architecture{};
  bool qkv_bias = false;
  if (model_type.value() == "llama") {
    architecture = Architecture::smollm2_llama;
    const auto architecture_status =
        validate_architecture_name(root.value(), "LlamaForCausalLM");
    const auto attention_bias = required_bool(root.value(), "attention_bias");
    if (!architecture_status.ok() || !attention_bias) {
      return config_failure<ModelSpec>();
    }
    qkv_bias = attention_bias.value();
  } else if (model_type.value() == "qwen2") {
    architecture = Architecture::qwen2;
    const auto architecture_status =
        validate_architecture_name(root.value(), "Qwen2ForCausalLM");
    if (!architecture_status.ok()) {
      return config_failure<ModelSpec>();
    }
    qkv_bias = true;
  } else {
    return config_failure<ModelSpec>();
  }

  const auto layers = required_u32(root.value(), "num_hidden_layers");
  const auto hidden = required_u32(root.value(), "hidden_size");
  const auto intermediate = required_u32(root.value(), "intermediate_size");
  const auto query_heads = required_u32(root.value(), "num_attention_heads");
  const auto kv_heads = required_u32(root.value(), "num_key_value_heads");
  const auto vocabulary = required_u32(root.value(), "vocab_size");
  const auto max_positions =
      required_u32(root.value(), "max_position_embeddings");
  const auto epsilon = required_float(root.value(), "rms_norm_eps");
  const auto rope_theta = required_float(root.value(), "rope_theta");
  const auto tied = required_bool(root.value(), "tie_word_embeddings");

  if (!layers || !hidden || !intermediate || !query_heads || !kv_heads ||
      !vocabulary || !max_positions || !epsilon || !rope_theta || !tied ||
      query_heads.value() == 0 || hidden.value() % query_heads.value() != 0) {
    return config_failure<ModelSpec>();
  }

  const ModelSpec spec{
      architecture,
      layers.value(),
      hidden.value(),
      intermediate.value(),
      query_heads.value(),
      kv_heads.value(),
      hidden.value() / query_heads.value(),
      vocabulary.value(),
      max_positions.value(),
      epsilon.value(),
      rope_theta.value(),
      qkv_bias,
      tied.value(),
  };
  const auto validation = validate(spec);
  if (!validation.ok()) {
    return Result<ModelSpec>::failure(validation);
  }
  return Result<ModelSpec>::success(spec);
}

Result<ModelSpec>
load_model_config(const std::filesystem::path& path) noexcept {
  auto mapping = MappedFile::open_read_only(path);
  if (!mapping) {
    return Result<ModelSpec>::failure(mapping.status());
  }
  const auto bytes = mapping.value().bytes();
  const std::string_view text(reinterpret_cast<const char*>(bytes.data()),
                              bytes.size());
  return parse_model_config(text);
}

} // namespace marketforge
