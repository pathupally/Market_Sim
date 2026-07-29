#include "test_support.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "marketforge/core/mapped_file.hpp"
#include "marketforge/model/loaded_weights.hpp"
#include "marketforge/model/model_config.hpp"
#include "marketforge/model/safetensors.hpp"

namespace {

using marketforge::Architecture;
using marketforge::DType;
using marketforge::ErrorCode;
using marketforge::LoadedWeights;
using marketforge::MappedFile;
using marketforge::ModelSpec;
using marketforge::SafeTensorFile;

std::filesystem::path fixture_path(const std::string_view relative) {
  return std::filesystem::path(MARKETFORGE_SOURCE_DIR) / relative;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<std::byte> make_safetensors(std::string header,
                                        const std::size_t data_bytes) {
  while (header.size() % 8 != 0) {
    header.push_back(' ');
  }
  std::vector<std::byte> bytes(8 + header.size() + data_bytes);
  const auto header_size = static_cast<std::uint64_t>(header.size());
  for (std::size_t index = 0; index < 8; ++index) {
    bytes[index] =
        static_cast<std::byte>((header_size >> (index * 8U)) & 0xffU);
  }
  for (std::size_t index = 0; index < header.size(); ++index) {
    bytes[8 + index] =
        static_cast<std::byte>(static_cast<unsigned char>(header[index]));
  }
  for (std::size_t index = 0; index < data_bytes; ++index) {
    bytes[8 + header.size() + index] = static_cast<std::byte>(index & 0xffU);
  }
  return bytes;
}

class TemporaryFile {
public:
  explicit TemporaryFile(const std::span<const std::byte> bytes) {
    static std::uint64_t counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("marketforge-loader-" + std::to_string(++counter) + ".bin");
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }

  ~TemporaryFile() {
    std::error_code error;
    static_cast<void>(std::filesystem::remove(path_, error));
  }

  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

struct SyntheticTensor {
  std::string name;
  std::vector<std::uint64_t> shape;
};

std::uint64_t elements(const std::vector<std::uint64_t>& shape) {
  std::uint64_t count = 1;
  for (const auto extent : shape) {
    count *= extent;
  }
  return count;
}

std::vector<std::byte>
make_bf16_file(const std::vector<SyntheticTensor>& tensors) {
  std::ostringstream header;
  header << '{';
  std::uint64_t offset = 0;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto& tensor = tensors[index];
    const std::uint64_t end = offset + elements(tensor.shape) * 2;
    if (index != 0) {
      header << ',';
    }
    header << '"' << tensor.name << "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis) {
      if (axis != 0) {
        header << ',';
      }
      header << tensor.shape[axis];
    }
    header << "],\"data_offsets\":[" << offset << ',' << end << "]}";
    offset = end;
  }
  header << '}';
  return make_safetensors(header.str(), static_cast<std::size_t>(offset));
}

ModelSpec tiny_smollm_spec() {
  return ModelSpec{
      Architecture::smollm2_llama,
      1,
      4,
      8,
      2,
      1,
      2,
      8,
      16,
      1.0e-5F,
      10'000.0F,
      false,
      true,
  };
}

std::vector<SyntheticTensor> tiny_smollm_tensors() {
  return {
      {"model.embed_tokens.weight", {8, 4}},
      {"model.layers.0.input_layernorm.weight", {4}},
      {"model.layers.0.self_attn.q_proj.weight", {4, 4}},
      {"model.layers.0.self_attn.k_proj.weight", {2, 4}},
      {"model.layers.0.self_attn.v_proj.weight", {2, 4}},
      {"model.layers.0.self_attn.o_proj.weight", {4, 4}},
      {"model.layers.0.post_attention_layernorm.weight", {4}},
      {"model.layers.0.mlp.gate_proj.weight", {8, 4}},
      {"model.layers.0.mlp.up_proj.weight", {8, 4}},
      {"model.layers.0.mlp.down_proj.weight", {4, 8}},
      {"model.norm.weight", {4}},
  };
}

marketforge::Result<SafeTensorFile>
parse_owned_file(const std::vector<std::byte>& bytes, TemporaryFile& file) {
  static_cast<void>(bytes);
  auto mapping = MappedFile::open_read_only(file.path());
  if (!mapping) {
    return marketforge::Result<SafeTensorFile>::failure(mapping.status());
  }
  return SafeTensorFile::parse(std::move(mapping).value());
}

MF_TEST(mapped_file_is_move_only_and_safetensors_views_are_borrowed) {
  const std::string header =
      read_text(fixture_path("tests/fixtures/safetensors/valid-header.json"));
  const auto bytes = make_safetensors(header, 12);
  TemporaryFile temporary(bytes);

  auto mapping_result = MappedFile::open_read_only(temporary.path());
  MF_CHECK(mapping_result);
  MappedFile mapping = std::move(mapping_result).value();
  MF_CHECK_EQ(mapping.bytes().size(), bytes.size());

  MappedFile moved = std::move(mapping);
  MF_CHECK_EQ(mapping.bytes().size(), 0U);
  auto parsed = SafeTensorFile::parse(std::move(moved));
  MF_CHECK(parsed);
  MF_CHECK_EQ(parsed.value().records().size(), 4U);

  const auto scalar = parsed.value().tensor("scalar_f32");
  const auto pair = parsed.value().tensor("pair_f16");
  const auto empty = parsed.value().tensor("empty_bf16");
  const auto integer = parsed.value().tensor("one_i32");
  MF_CHECK(scalar && pair && empty && integer);
  MF_CHECK_EQ(scalar.value().dtype, DType::f32);
  MF_CHECK_EQ(scalar.value().shape.rank, 0U);
  MF_CHECK_EQ(pair.value().dtype, DType::f16);
  MF_CHECK_EQ(empty.value().dtype, DType::bf16);
  MF_CHECK_EQ(empty.value().shape.extents[0], 0ULL);
  MF_CHECK_EQ(integer.value().dtype, DType::i32);
  MF_CHECK(!parsed.value().tensor("missing"));
}

MF_TEST(safetensors_rejects_every_truncation_boundary) {
  const std::string header =
      read_text(fixture_path("tests/fixtures/safetensors/valid-header.json"));
  const auto bytes = make_safetensors(header, 12);
  for (std::size_t size = 0; size < bytes.size(); ++size) {
    const auto parsed =
        marketforge::parse_safetensors_metadata({bytes.data(), size});
    MF_CHECK(!parsed);
  }
  MF_CHECK(marketforge::parse_safetensors_metadata(bytes));
}

MF_TEST(safetensors_rejects_malformed_json_shapes_dtypes_and_offsets) {
  const auto rejected = [](const std::string& header,
                           const std::size_t data_bytes) {
    const auto bytes = make_safetensors(header, data_bytes);
    return !marketforge::parse_safetensors_metadata(bytes);
  };

  MF_CHECK(rejected("{", 0));
  MF_CHECK(rejected(" {\"x\":{\"dtype\":\"F32\",\"shape\":[],"
                    "\"data_offsets\":[0,4]}}",
                    4));
  MF_CHECK(rejected("{\"x\":{\"dtype\":\"F32\",\"dtype\":\"F32\",\"shape\":[],"
                    "\"data_offsets\":[0,4]}}",
                    4));
  MF_CHECK(rejected(
      "{\"x\":{\"dtype\":\"F64\",\"shape\":[],\"data_offsets\":[0,8]}}", 8));
  MF_CHECK(rejected("{\"x\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,1,1,1],"
                    "\"data_offsets\":[0,4]}}",
                    4));
  MF_CHECK(rejected("{\"x\":{\"dtype\":\"F32\",\"shape\":[-1],"
                    "\"data_offsets\":[0,4]}}",
                    4));
  MF_CHECK(rejected("{\"x\":{\"dtype\":\"F32\","
                    "\"shape\":[18446744073709551615,2],"
                    "\"data_offsets\":[0,4]}}",
                    4));
  MF_CHECK(rejected("{\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
                    "\"data_offsets\":[4,0]}}",
                    4));
  MF_CHECK(rejected("{\"x\":{\"dtype\":\"F32\",\"shape\":[2],"
                    "\"data_offsets\":[0,4]}}",
                    4));
  MF_CHECK(rejected("{\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
                    "\"data_offsets\":[1,5]}}",
                    5));
  MF_CHECK(rejected("{\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
                    "\"data_offsets\":[0,4]}}",
                    5));

  const auto overlap_header = read_text(
      fixture_path("tests/fixtures/safetensors/malformed-overlap-header.json"));
  MF_CHECK(rejected(overlap_header, 6));
}

MF_TEST(safetensors_rejects_invalid_utf8_and_oversize_headers) {
  auto bytes = make_safetensors("{\"x\":{\"dtype\":\"F32\",\"shape\":[],"
                                "\"data_offsets\":[0,4]}}",
                                4);
  bytes[10] = static_cast<std::byte>(0xffU);
  const auto invalid_utf8 = marketforge::parse_safetensors_metadata(bytes);
  MF_CHECK(!invalid_utf8);
  MF_CHECK_EQ(invalid_utf8.status().code, ErrorCode::invalid_utf8);

  const auto valid = make_safetensors("{}", 0);
  const auto too_large = marketforge::parse_safetensors_metadata(valid, 1);
  MF_CHECK(!too_large);
  MF_CHECK_EQ(too_large.status().code, ErrorCode::resource_limit);
}

MF_TEST(locked_model_configs_parse_to_exact_specs) {
  const auto smol_text =
      read_text(fixture_path("tests/fixtures/config/smollm2-135m.json"));
  const auto qwen_text = read_text(
      fixture_path("tests/fixtures/config/qwen2.5-0.5b-instruct.json"));
  const auto smol = marketforge::parse_model_config(smol_text);
  const auto qwen = marketforge::parse_model_config(qwen_text);
  MF_CHECK(smol && qwen);
  MF_CHECK_EQ(smol.value(), marketforge::smollm2_135m_profile().spec);
  MF_CHECK_EQ(qwen.value(), marketforge::qwen2_5_0_5b_profile().spec);

  const auto smol_bytes = std::as_bytes(std::span(smol_text));
  TemporaryFile config_file(smol_bytes);
  const auto loaded = marketforge::load_model_config(config_file.path());
  MF_CHECK(loaded);
  MF_CHECK_EQ(loaded.value(), smol.value());
}

MF_TEST(model_config_rejects_duplicate_missing_and_unknown_architecture) {
  const auto duplicate = marketforge::parse_model_config(
      "{\"model_type\":\"llama\",\"model_type\":\"qwen2\"}");
  MF_CHECK(!duplicate);
  MF_CHECK_EQ(duplicate.status().code, ErrorCode::duplicate_key);

  const auto missing = marketforge::parse_model_config(
      "{\"model_type\":\"llama\",\"architectures\":"
      "[\"LlamaForCausalLM\"]}");
  MF_CHECK(!missing);

  const auto unknown =
      marketforge::parse_model_config("{\"model_type\":\"other\"}");
  MF_CHECK(!unknown);
}

MF_TEST(smollm_weight_binder_checks_names_shapes_and_tied_alias) {
  const auto spec = tiny_smollm_spec();
  auto tensors = tiny_smollm_tensors();
  auto bytes = make_bf16_file(tensors);
  TemporaryFile valid_file(bytes);
  auto parsed = parse_owned_file(bytes, valid_file);
  MF_CHECK(parsed);
  auto bound = LoadedWeights::bind(std::move(parsed).value(), spec);
  MF_CHECK(bound);
  MF_CHECK_EQ(bound.value().layers().size(), 1U);
  MF_CHECK(bound.value().output_head_aliases_embedding());
  MF_CHECK_EQ(bound.value().embedding().data, bound.value().output_head().data);

  tensors.erase(tensors.begin() + 2);
  bytes = make_bf16_file(tensors);
  TemporaryFile missing_file(bytes);
  auto missing_parsed = parse_owned_file(bytes, missing_file);
  MF_CHECK(missing_parsed);
  const auto missing =
      LoadedWeights::bind(std::move(missing_parsed).value(), spec);
  MF_CHECK(!missing);
  MF_CHECK_EQ(missing.status().code, ErrorCode::missing_tensor);

  tensors = tiny_smollm_tensors();
  tensors[2].shape = {3, 4};
  bytes = make_bf16_file(tensors);
  TemporaryFile wrong_shape_file(bytes);
  auto wrong_shape_parsed = parse_owned_file(bytes, wrong_shape_file);
  MF_CHECK(wrong_shape_parsed);
  const auto wrong_shape =
      LoadedWeights::bind(std::move(wrong_shape_parsed).value(), spec);
  MF_CHECK(!wrong_shape);
  MF_CHECK_EQ(wrong_shape.status().code, ErrorCode::invalid_tensor);

  tensors = tiny_smollm_tensors();
  tensors.push_back({"unexpected.critical.weight", {1}});
  bytes = make_bf16_file(tensors);
  TemporaryFile unexpected_file(bytes);
  auto unexpected_parsed = parse_owned_file(bytes, unexpected_file);
  MF_CHECK(unexpected_parsed);
  const auto unexpected =
      LoadedWeights::bind(std::move(unexpected_parsed).value(), spec);
  MF_CHECK(!unexpected);
  MF_CHECK_EQ(unexpected.status().code, ErrorCode::unexpected_tensor);
}

MF_TEST(smollm_weight_binder_accepts_optional_explicit_output_head) {
  const auto spec = tiny_smollm_spec();
  auto tensors = tiny_smollm_tensors();
  tensors.push_back({"lm_head.weight", {8, 4}});
  const auto bytes = make_bf16_file(tensors);
  TemporaryFile temporary(bytes);
  auto parsed = parse_owned_file(bytes, temporary);
  MF_CHECK(parsed);
  const auto bound = LoadedWeights::bind(std::move(parsed).value(), spec);
  MF_CHECK(bound);
  MF_CHECK(!bound.value().output_head_aliases_embedding());
  MF_CHECK_NE(bound.value().embedding().data, bound.value().output_head().data);
}

MF_TEST(deterministic_parser_fuzz_corpus_handles_one_hundred_thousand_inputs) {
  std::array<std::byte, 128> bytes{};
  std::uint64_t state = 0x8c3c'010c'b475'4c9dULL;
  for (std::size_t iteration = 0; iteration < 100'000; ++iteration) {
    state = state * 6'364'136'223'846'793'005ULL + 1ULL;
    const std::size_t size = static_cast<std::size_t>(state % bytes.size());
    for (std::size_t index = 0; index < size; ++index) {
      state = state * 2'862'933'555'777'941'757ULL + 3'037'000'493ULL;
      bytes[index] = static_cast<std::byte>(state >> 56U);
    }
    static_cast<void>(
        marketforge::parse_safetensors_metadata({bytes.data(), size}, 1'024));
  }
  MF_CHECK(true);
}

} // namespace
