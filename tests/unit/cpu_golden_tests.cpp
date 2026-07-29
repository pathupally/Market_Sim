#include "test_support.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "marketforge/core/mapped_file.hpp"
#include "marketforge/core/shape.hpp"
#include "marketforge/cpu/operators.hpp"
#include "marketforge/model/safetensors.hpp"

namespace {

using marketforge::AttentionWeights;
using marketforge::ConstTensorView;
using marketforge::CpuKvView;
using marketforge::CpuWorkspace;
using marketforge::DecoderLayerSpec;
using marketforge::DecoderLayerTrace;
using marketforge::DType;
using marketforge::LayerWeights;
using marketforge::MemoryKind;
using marketforge::MlpWeights;
using marketforge::RopeSpec;
using marketforge::SafeTensorFile;
using marketforge::TensorView;

constexpr std::uint64_t batch = 2;
constexpr std::uint64_t tokens = 3;
constexpr std::uint64_t hidden = 8;
constexpr std::uint64_t intermediate = 12;
constexpr std::uint64_t query_heads = 2;
constexpr std::uint64_t kv_heads = 1;
constexpr std::uint64_t head_dim = 4;
constexpr std::uint64_t rows = batch * tokens;

template <std::size_t Rank>
TensorView view(std::span<float> values,
                const std::array<std::uint64_t, Rank>& extents) {
  return TensorView{
      values.data(),
      marketforge::make_shape(extents).value(),
      DType::f32,
      MemoryKind::host,
  };
}

template <std::size_t Rank>
ConstTensorView const_view(std::span<const float> values,
                           const std::array<std::uint64_t, Rank>& extents) {
  return ConstTensorView{
      values.data(),
      marketforge::make_shape(extents).value(),
      DType::f32,
      MemoryKind::host,
  };
}

ConstTensorView as_const(const TensorView value) {
  return ConstTensorView{
      value.data,
      value.shape,
      value.dtype,
      value.memory,
  };
}

ConstTensorView tensor(const SafeTensorFile& file,
                       const std::string_view name) {
  const auto result = file.tensor(name);
  MF_CHECK(result);
  return result.value();
}

std::vector<float> copy_f32(const ConstTensorView source) {
  MF_CHECK_EQ(source.dtype, DType::f32);
  const auto count = marketforge::checked_numel(source.shape);
  MF_CHECK(count);
  std::vector<float> result(static_cast<std::size_t>(count.value()));
  std::memcpy(result.data(), source.data, result.size() * sizeof(float));
  return result;
}

std::vector<std::uint32_t> copy_i32_as_u32(const ConstTensorView source) {
  MF_CHECK_EQ(source.dtype, DType::i32);
  const auto count = marketforge::checked_numel(source.shape);
  MF_CHECK(count);
  std::vector<std::int32_t> signed_values(
      static_cast<std::size_t>(count.value()));
  std::memcpy(signed_values.data(), source.data,
              signed_values.size() * sizeof(std::int32_t));
  std::vector<std::uint32_t> result;
  result.reserve(signed_values.size());
  for (const auto value : signed_values) {
    MF_CHECK(value >= 0);
    result.push_back(static_cast<std::uint32_t>(value));
  }
  return result;
}

void check_near(const std::span<const float> actual,
                const ConstTensorView expected, const float tolerance) {
  const auto expected_values = copy_f32(expected);
  MF_CHECK_EQ(actual.size(), expected_values.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    MF_CHECK_NEAR(actual[index], expected_values[index], tolerance);
  }
}

LayerWeights fixture_weights(const SafeTensorFile& file) {
  return LayerWeights{
      tensor(file, "weights.input_norm"),
      AttentionWeights{
          tensor(file, "weights.q_proj"),
          tensor(file, "weights.k_proj"),
          tensor(file, "weights.v_proj"),
          tensor(file, "weights.o_proj"),
      },
      tensor(file, "weights.post_attention_norm"),
      MlpWeights{
          tensor(file, "weights.gate_proj"),
          tensor(file, "weights.up_proj"),
          tensor(file, "weights.down_proj"),
      },
  };
}

marketforge::Result<SafeTensorFile> open_fixture() {
  const auto path = std::filesystem::path(MARKETFORGE_SOURCE_DIR) /
                    "tests/fixtures/golden/"
                    "smollm2-tiny-layer-f32.safetensors";
  auto mapping = marketforge::MappedFile::open_read_only(path);
  if (!mapping) {
    return marketforge::Result<SafeTensorFile>::failure(mapping.status());
  }
  return SafeTensorFile::parse(std::move(mapping).value());
}

MF_TEST(pytorch_fixture_matches_every_fp32_decoder_intermediate) {
  auto file_result = open_fixture();
  MF_CHECK(file_result);
  auto file = std::move(file_result).value();
  const auto weights = fixture_weights(file);
  const auto input = tensor(file, "input.hidden");
  const auto positions = copy_i32_as_u32(tensor(file, "position_ids"));
  const auto contexts = copy_i32_as_u32(tensor(file, "context_lengths"));

  const std::array<std::uint64_t, 3> hidden_shape{batch, tokens, hidden};
  const std::array<std::uint64_t, 2> flat_hidden_shape{rows, hidden};
  const std::array<std::uint64_t, 2> flat_kv_shape{rows, kv_heads * head_dim};
  const std::array<std::uint64_t, 4> query_shape{batch, tokens, query_heads,
                                                 head_dim};
  const std::array<std::uint64_t, 4> kv_shape{batch, tokens, kv_heads,
                                              head_dim};
  const std::array<std::uint64_t, 2> flat_intermediate_shape{rows,
                                                             intermediate};

  std::vector<float> normalized(rows * hidden);
  auto normalized_view = view(std::span<float>(normalized), hidden_shape);
  MF_CHECK(marketforge::rms_norm_f32(input, weights.input_norm, 1.0e-5F,
                                     normalized_view)
               .ok());
  check_near(normalized, tensor(file, "expected.input_norm"), 2.0e-6F);

  std::vector<float> query(rows * hidden);
  std::vector<float> key(rows * kv_heads * head_dim);
  std::vector<float> value(rows * kv_heads * head_dim);
  auto query_flat = view(std::span<float>(query), flat_hidden_shape);
  auto key_flat = view(std::span<float>(key), flat_kv_shape);
  auto value_flat = view(std::span<float>(value), flat_kv_shape);
  const auto normalized_flat =
      const_view(std::span<const float>(normalized), flat_hidden_shape);
  MF_CHECK(marketforge::linear_f32(normalized_flat, weights.attention.query,
                                   query_flat)
               .ok());
  MF_CHECK(
      marketforge::linear_f32(normalized_flat, weights.attention.key, key_flat)
          .ok());
  MF_CHECK(marketforge::linear_f32(normalized_flat, weights.attention.value,
                                   value_flat)
               .ok());

  auto query_view = view(std::span<float>(query), query_shape);
  auto key_view = view(std::span<float>(key), kv_shape);
  auto value_view = view(std::span<float>(value), kv_shape);
  MF_CHECK(marketforge::apply_rope_f32(query_view, key_view, positions,
                                       RopeSpec{10'000.0F, 16})
               .ok());
  check_near(query, tensor(file, "expected.query_rope"), 2.0e-6F);
  check_near(key, tensor(file, "expected.key_rope"), 2.0e-6F);
  check_near(value, tensor(file, "expected.value"), 2.0e-6F);

  std::vector<float> key_cache(rows * kv_heads * head_dim);
  std::vector<float> value_cache(rows * kv_heads * head_dim);
  auto key_cache_view = view(std::span<float>(key_cache), kv_shape);
  auto value_cache_view = view(std::span<float>(value_cache), kv_shape);
  MF_CHECK(marketforge::append_kv_f32(
               as_const(key_view), as_const(value_view), positions,
               CpuKvView{key_cache_view, value_cache_view})
               .ok());
  check_near(key_cache, tensor(file, "expected.key_cache"), 2.0e-6F);
  check_near(value_cache, tensor(file, "expected.value_cache"), 2.0e-6F);

  auto logits = copy_f32(tensor(file, "expected.attention_logits"));
  std::vector<float> probabilities(logits.size());
  std::vector<std::uint32_t> valid_lengths;
  valid_lengths.reserve(batch * query_heads * tokens);
  for (std::uint64_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::uint64_t head = 0; head < query_heads; ++head) {
      for (std::uint32_t token = 0; token < tokens; ++token) {
        valid_lengths.push_back(token + 1);
      }
    }
  }
  const std::array<std::uint64_t, 2> probability_shape{
      batch * query_heads * tokens, tokens};
  MF_CHECK(marketforge::softmax_f32(
               const_view(std::span<const float>(logits), probability_shape),
               valid_lengths,
               view(std::span<float>(probabilities), probability_shape))
               .ok());
  check_near(probabilities, tensor(file, "expected.attention_probabilities"),
             2.0e-6F);

  std::vector<float> attention(rows * hidden);
  const auto required_attention =
      marketforge::required_attention_workspace_floats(
          as_const(query_view), as_const(key_cache_view));
  MF_CHECK(required_attention);
  auto attention_workspace =
      CpuWorkspace::allocate(required_attention.value()).value();
  MF_CHECK(
      marketforge::attention_f32(as_const(query_view), as_const(key_cache_view),
                                 as_const(value_cache_view), contexts,
                                 view(std::span<float>(attention), query_shape),
                                 attention_workspace)
          .ok());
  check_near(attention, tensor(file, "expected.attention_heads"), 5.0e-6F);

  std::vector<float> projected(rows * hidden);
  MF_CHECK(marketforge::linear_f32(
               const_view(std::span<const float>(attention), flat_hidden_shape),
               weights.attention.output,
               view(std::span<float>(projected), flat_hidden_shape))
               .ok());
  check_near(projected, tensor(file, "expected.attention_projected"), 5.0e-6F);

  auto after_attention = copy_f32(input);
  for (std::size_t index = 0; index < after_attention.size(); ++index) {
    after_attention[index] += projected[index];
  }
  check_near(after_attention, tensor(file, "expected.after_attention"),
             5.0e-6F);

  std::vector<float> post_norm(rows * hidden);
  MF_CHECK(
      marketforge::rms_norm_f32(
          const_view(std::span<const float>(after_attention), hidden_shape),
          weights.post_attention_norm, 1.0e-5F,
          view(std::span<float>(post_norm), hidden_shape))
          .ok());
  check_near(post_norm, tensor(file, "expected.post_attention_norm"), 5.0e-6F);

  std::vector<float> gate(rows * intermediate);
  std::vector<float> up(rows * intermediate);
  const auto post_norm_flat =
      const_view(std::span<const float>(post_norm), flat_hidden_shape);
  MF_CHECK(marketforge::linear_f32(
               post_norm_flat, weights.mlp.gate,
               view(std::span<float>(gate), flat_intermediate_shape))
               .ok());
  MF_CHECK(marketforge::linear_f32(
               post_norm_flat, weights.mlp.up,
               view(std::span<float>(up), flat_intermediate_shape))
               .ok());
  check_near(gate, tensor(file, "expected.gate"), 5.0e-6F);
  check_near(up, tensor(file, "expected.up"), 5.0e-6F);

  for (std::size_t index = 0; index < gate.size(); ++index) {
    gate[index] = gate[index] / (1.0F + std::exp(-gate[index])) * up[index];
  }
  check_near(gate, tensor(file, "expected.swiglu"), 5.0e-6F);

  std::vector<float> down(rows * hidden);
  MF_CHECK(
      marketforge::linear_f32(
          const_view(std::span<const float>(gate), flat_intermediate_shape),
          weights.mlp.down, view(std::span<float>(down), flat_hidden_shape))
          .ok());
  check_near(down, tensor(file, "expected.mlp_down"), 5.0e-6F);

  for (std::size_t index = 0; index < after_attention.size(); ++index) {
    after_attention[index] += down[index];
  }
  check_near(after_attention, tensor(file, "expected.after_mlp"), 1.0e-5F);
}

MF_TEST(pytorch_fixture_matches_end_to_end_decoder_layer_trace) {
  auto file_result = open_fixture();
  MF_CHECK(file_result);
  auto file = std::move(file_result).value();
  const auto weights = fixture_weights(file);
  auto hidden_values = copy_f32(tensor(file, "input.hidden"));
  const auto original_hidden = hidden_values;
  const auto positions = copy_i32_as_u32(tensor(file, "position_ids"));
  const auto contexts = copy_i32_as_u32(tensor(file, "context_lengths"));
  const std::array<std::uint64_t, 3> hidden_shape{batch, tokens, hidden};
  const std::array<std::uint64_t, 4> cache_shape{batch, tokens, kv_heads,
                                                 head_dim};
  std::vector<float> key_cache(rows * kv_heads * head_dim, -3.0F);
  std::vector<float> value_cache(rows * kv_heads * head_dim, -4.0F);
  std::vector<float> after_attention(rows * hidden);
  std::vector<float> after_mlp(rows * hidden);
  const DecoderLayerSpec spec{
      hidden,
      intermediate,
      query_heads,
      kv_heads,
      head_dim,
      1.0e-5F,
      RopeSpec{10'000.0F, 16},
  };
  const auto required = marketforge::required_decoder_layer_workspace_floats(
      spec, batch, tokens, tokens);
  MF_CHECK(required);
  auto workspace = CpuWorkspace::allocate(required.value()).value();
  float* const workspace_address = workspace.floats().data();
  DecoderLayerTrace trace{
      view(std::span<float>(after_attention), hidden_shape),
      view(std::span<float>(after_mlp), hidden_shape),
  };
  MF_CHECK(marketforge::smollm2_decoder_layer_f32(
               weights, spec,
               view(std::span<float>(hidden_values), hidden_shape),
               CpuKvView{
                   view(std::span<float>(key_cache), cache_shape),
                   view(std::span<float>(value_cache), cache_shape),
               },
               positions, contexts, workspace, &trace)
               .ok());

  check_near(after_attention, tensor(file, "expected.after_attention"),
             5.0e-6F);
  check_near(after_mlp, tensor(file, "expected.after_mlp"), 1.0e-5F);
  check_near(hidden_values, tensor(file, "expected.after_mlp"), 1.0e-5F);
  check_near(key_cache, tensor(file, "expected.key_cache"), 2.0e-6F);
  check_near(value_cache, tensor(file, "expected.value_cache"), 2.0e-6F);
  MF_CHECK_EQ(workspace.floats().data(), workspace_address);

  const auto source_again = copy_f32(tensor(file, "input.hidden"));
  MF_CHECK_EQ(source_again, original_hidden);
}

} // namespace
