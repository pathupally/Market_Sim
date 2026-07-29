#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "marketforge/core/shape.hpp"
#include "marketforge/cpu/operators.hpp"

namespace {

using marketforge::AttentionWeights;
using marketforge::ConstTensorView;
using marketforge::CpuKvView;
using marketforge::CpuWorkspace;
using marketforge::DecoderLayerSpec;
using marketforge::DecoderLayerTrace;
using marketforge::DType;
using marketforge::ErrorCode;
using marketforge::LayerWeights;
using marketforge::MemoryKind;
using marketforge::MlpWeights;
using marketforge::RopeSpec;
using marketforge::TensorView;

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

void check_all_equal(const std::span<const float> values,
                     const float expected) {
  for (const float value : values) {
    MF_CHECK_EQ(value, expected);
  }
}

MF_TEST(rms_norm_and_linear_match_hand_computable_values) {
  std::array<float, 8> input{1.0F, 2.0F, 3.0F, 4.0F, -1.0F, 0.5F, 2.0F, -3.0F};
  const std::array<float, 4> weight{1.0F, 0.5F, 2.0F, 1.5F};
  std::array<float, 8> output{};
  const std::array<std::uint64_t, 2> input_shape{2, 4};
  const std::array<std::uint64_t, 1> weight_shape{4};

  const auto status = marketforge::rms_norm_f32(
      const_view(std::span<const float>(input), input_shape),
      const_view(std::span<const float>(weight), weight_shape), 1.0e-5F,
      view(std::span<float>(output), input_shape));
  MF_CHECK(status.ok());
  for (std::size_t row = 0; row < 2; ++row) {
    float squares = 0.0F;
    for (std::size_t column = 0; column < 4; ++column) {
      const float value = input[row * 4 + column];
      squares += value * value;
    }
    const float inverse = 1.0F / std::sqrt(squares / 4.0F + 1.0e-5F);
    for (std::size_t column = 0; column < 4; ++column) {
      MF_CHECK_NEAR(output[row * 4 + column],
                    input[row * 4 + column] * inverse * weight[column],
                    1.0e-6F);
    }
  }

  auto in_place = input;
  MF_CHECK(marketforge::rms_norm_f32(
               const_view(std::span<const float>(in_place), input_shape),
               const_view(std::span<const float>(weight), weight_shape),
               1.0e-5F, view(std::span<float>(in_place), input_shape))
               .ok());
  for (std::size_t index = 0; index < output.size(); ++index) {
    MF_CHECK_NEAR(in_place[index], output[index], 1.0e-6F);
  }

  const std::array<float, 3> linear_input{1.0F, 2.0F, 3.0F};
  const std::array<float, 6> linear_weight{1.0F, 0.0F, -1.0F, 0.5F, 0.5F, 0.5F};
  std::array<float, 2> linear_output{};
  const std::array<std::uint64_t, 2> one_by_three{1, 3};
  const std::array<std::uint64_t, 2> two_by_three{2, 3};
  const std::array<std::uint64_t, 2> one_by_two{1, 2};
  MF_CHECK(marketforge::linear_f32(
               const_view(std::span<const float>(linear_input), one_by_three),
               const_view(std::span<const float>(linear_weight), two_by_three),
               view(std::span<float>(linear_output), one_by_two))
               .ok());
  MF_CHECK_NEAR(linear_output[0], -2.0F, 1.0e-6F);
  MF_CHECK_NEAR(linear_output[1], 3.0F, 1.0e-6F);
}

MF_TEST(operator_validation_precedes_output_mutation) {
  const std::array<float, 4> input{1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 2> weight{1.0F, 1.0F};
  std::array<float, 4> output{77.0F, 77.0F, 77.0F, 77.0F};
  const std::array<std::uint64_t, 2> input_shape{2, 2};
  const std::array<std::uint64_t, 1> weight_shape{2};
  auto invalid_output = view(std::span<float>(output), input_shape);
  invalid_output.dtype = DType::f16;
  const auto status = marketforge::rms_norm_f32(
      const_view(std::span<const float>(input), input_shape),
      const_view(std::span<const float>(weight), weight_shape), 1.0e-5F,
      invalid_output);
  MF_CHECK(!status.ok());
  MF_CHECK_EQ(status.code, ErrorCode::unsupported_dtype);
  check_all_equal(output, 77.0F);

  auto non_host = const_view(std::span<const float>(input), input_shape);
  non_host.memory = MemoryKind::device;
  const auto non_host_status = marketforge::rms_norm_f32(
      non_host, const_view(std::span<const float>(weight), weight_shape),
      1.0e-5F, view(std::span<float>(output), input_shape));
  MF_CHECK(!non_host_status.ok());
  check_all_equal(output, 77.0F);

  const std::array<std::uint64_t, 2> wrong_output_shape{1, 4};
  const auto wrong_shape_status = marketforge::rms_norm_f32(
      const_view(std::span<const float>(input), input_shape),
      const_view(std::span<const float>(weight), weight_shape), 1.0e-5F,
      view(std::span<float>(output), wrong_output_shape));
  MF_CHECK(!wrong_shape_status.ok());
  MF_CHECK_EQ(wrong_shape_status.code, ErrorCode::invalid_shape);
  check_all_equal(output, 77.0F);
}

MF_TEST(softmax_is_stable_masks_tail_and_rejects_all_masked_rows) {
  const std::array<float, 8> input{
      1'000.0F, 1'001.0F, 999.0F, -50.0F, 5.0F, 6.0F, 7.0F, 8.0F,
  };
  std::array<float, 8> output{};
  const std::array<std::uint64_t, 2> shape{2, 4};
  const std::array<std::uint32_t, 2> lengths{3, 2};
  MF_CHECK(
      marketforge::softmax_f32(const_view(std::span<const float>(input), shape),
                               lengths, view(std::span<float>(output), shape))
          .ok());
  MF_CHECK_NEAR(output[0] + output[1] + output[2], 1.0F, 1.0e-6F);
  MF_CHECK_EQ(output[3], 0.0F);
  MF_CHECK_NEAR(output[4] + output[5], 1.0F, 1.0e-6F);
  MF_CHECK_EQ(output[6], 0.0F);
  MF_CHECK_EQ(output[7], 0.0F);
  MF_CHECK(output[1] > output[0]);
  MF_CHECK(output[0] > output[2]);

  std::array<float, 8> unchanged{};
  unchanged.fill(19.0F);
  const std::array<std::uint32_t, 2> invalid_lengths{3, 0};
  const auto rejected = marketforge::softmax_f32(
      const_view(std::span<const float>(input), shape), invalid_lengths,
      view(std::span<float>(unchanged), shape));
  MF_CHECK(!rejected.ok());
  check_all_equal(unchanged, 19.0F);
}

MF_TEST(rope_uses_llama_half_rotation_and_respects_positions) {
  std::array<float, 8> query{9.0F, 8.0F, 7.0F, 6.0F, 1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 8> key{4.0F, 3.0F, 2.0F, 1.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  const auto original_query = query;
  const auto original_key = key;
  const std::array<std::uint64_t, 4> shape{1, 2, 1, 4};
  const std::array<std::uint32_t, 2> positions{0, 1};
  const RopeSpec rope{10'000.0F, 16};
  MF_CHECK(marketforge::apply_rope_f32(view(std::span<float>(query), shape),
                                       view(std::span<float>(key), shape),
                                       positions, rope)
               .ok());
  for (std::size_t index = 0; index < 4; ++index) {
    MF_CHECK_EQ(query[index], original_query[index]);
    MF_CHECK_EQ(key[index], original_key[index]);
  }

  const float cosine_fast = std::cos(1.0F);
  const float sine_fast = std::sin(1.0F);
  const float cosine_slow = std::cos(0.01F);
  const float sine_slow = std::sin(0.01F);
  MF_CHECK_NEAR(query[4], 1.0F * cosine_fast - 3.0F * sine_fast, 1.0e-6F);
  MF_CHECK_NEAR(query[5], 2.0F * cosine_slow - 4.0F * sine_slow, 1.0e-6F);
  MF_CHECK_NEAR(query[6], 3.0F * cosine_fast + 1.0F * sine_fast, 1.0e-6F);
  MF_CHECK_NEAR(query[7], 4.0F * cosine_slow + 2.0F * sine_slow, 1.0e-6F);

  MF_CHECK(std::abs(query[4] - (1.0F * cosine_fast - 2.0F * sine_fast)) > 0.1F);

  auto invalid_query = original_query;
  auto invalid_key = original_key;
  const std::array<std::uint32_t, 2> invalid_positions{0, 16};
  MF_CHECK(
      !marketforge::apply_rope_f32(view(std::span<float>(invalid_query), shape),
                                   view(std::span<float>(invalid_key), shape),
                                   invalid_positions, rope)
           .ok());
  MF_CHECK_EQ(invalid_query, original_query);
  MF_CHECK_EQ(invalid_key, original_key);
}

MF_TEST(kv_append_changes_only_selected_positions) {
  const std::array<float, 4> key{1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> value{10.0F, 20.0F, 30.0F, 40.0F};
  std::array<float, 8> key_cache{};
  std::array<float, 8> value_cache{};
  key_cache.fill(-7.0F);
  value_cache.fill(-9.0F);
  const std::array<std::uint64_t, 4> input_shape{1, 2, 1, 2};
  const std::array<std::uint64_t, 4> cache_shape{1, 4, 1, 2};
  const std::array<std::uint32_t, 2> positions{1, 3};
  MF_CHECK(marketforge::append_kv_f32(
               const_view(std::span<const float>(key), input_shape),
               const_view(std::span<const float>(value), input_shape),
               positions,
               CpuKvView{
                   view(std::span<float>(key_cache), cache_shape),
                   view(std::span<float>(value_cache), cache_shape),
               })
               .ok());
  MF_CHECK_EQ(key_cache[0], -7.0F);
  MF_CHECK_EQ(key_cache[1], -7.0F);
  MF_CHECK_EQ(key_cache[2], 1.0F);
  MF_CHECK_EQ(key_cache[3], 2.0F);
  MF_CHECK_EQ(key_cache[4], -7.0F);
  MF_CHECK_EQ(key_cache[5], -7.0F);
  MF_CHECK_EQ(key_cache[6], 3.0F);
  MF_CHECK_EQ(key_cache[7], 4.0F);
  MF_CHECK_EQ(value_cache[2], 10.0F);
  MF_CHECK_EQ(value_cache[3], 20.0F);
  MF_CHECK_EQ(value_cache[6], 30.0F);
  MF_CHECK_EQ(value_cache[7], 40.0F);

  const auto before = key_cache;
  const std::array<std::uint32_t, 2> invalid_positions{1, 4};
  MF_CHECK(!marketforge::append_kv_f32(
                const_view(std::span<const float>(key), input_shape),
                const_view(std::span<const float>(value), input_shape),
                invalid_positions,
                CpuKvView{
                    view(std::span<float>(key_cache), cache_shape),
                    view(std::span<float>(value_cache), cache_shape),
                })
                .ok());
  MF_CHECK_EQ(key_cache, before);
}

MF_TEST(attention_is_causal_and_maps_grouped_query_heads) {
  const std::array<float, 8> query{};
  const std::array<float, 4> key_cache{};
  const std::array<float, 4> value_cache{1.0F, 10.0F, 3.0F, 30.0F};
  std::array<float, 8> output{};
  const std::array<std::uint64_t, 4> query_shape{1, 2, 2, 2};
  const std::array<std::uint64_t, 4> cache_shape{1, 2, 1, 2};
  const std::array<std::uint32_t, 1> contexts{2};
  const auto query_view =
      const_view(std::span<const float>(query), query_shape);
  const auto cache_view =
      const_view(std::span<const float>(key_cache), cache_shape);
  const auto required =
      marketforge::required_attention_workspace_floats(query_view, cache_view);
  auto workspace = CpuWorkspace::allocate(required.value()).value();
  MF_CHECK(marketforge::attention_f32(
               query_view, cache_view,
               const_view(std::span<const float>(value_cache), cache_shape),
               contexts, view(std::span<float>(output), query_shape), workspace)
               .ok());
  for (std::size_t head = 0; head < 2; ++head) {
    const std::size_t first = head * 2;
    MF_CHECK_NEAR(output[first], 1.0F, 1.0e-6F);
    MF_CHECK_NEAR(output[first + 1], 10.0F, 1.0e-6F);
    const std::size_t second = 4 + head * 2;
    MF_CHECK_NEAR(output[second], 2.0F, 1.0e-6F);
    MF_CHECK_NEAR(output[second + 1], 20.0F, 1.0e-6F);
  }

  const std::array<float, 4> gqa_query{};
  const std::array<float, 2> gqa_key{};
  const std::array<float, 2> gqa_value{2.0F, 20.0F};
  std::array<float, 4> gqa_output{};
  const std::array<std::uint64_t, 4> gqa_query_shape{1, 1, 4, 1};
  const std::array<std::uint64_t, 4> gqa_cache_shape{1, 1, 2, 1};
  const auto gqa_query_view =
      const_view(std::span<const float>(gqa_query), gqa_query_shape);
  const auto gqa_cache_view =
      const_view(std::span<const float>(gqa_key), gqa_cache_shape);
  auto gqa_workspace =
      CpuWorkspace::allocate(marketforge::required_attention_workspace_floats(
                                 gqa_query_view, gqa_cache_view)
                                 .value())
          .value();
  const std::array<std::uint32_t, 1> one_context{1};
  MF_CHECK(marketforge::attention_f32(
               gqa_query_view, gqa_cache_view,
               const_view(std::span<const float>(gqa_value), gqa_cache_shape),
               one_context, view(std::span<float>(gqa_output), gqa_query_shape),
               gqa_workspace)
               .ok());
  MF_CHECK_EQ(gqa_output[0], 2.0F);
  MF_CHECK_EQ(gqa_output[1], 2.0F);
  MF_CHECK_EQ(gqa_output[2], 20.0F);
  MF_CHECK_EQ(gqa_output[3], 20.0F);
}

struct TinyLayerStorage {
  std::vector<float> input_norm;
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> output;
  std::vector<float> post_norm;
  std::vector<float> gate;
  std::vector<float> up;
  std::vector<float> down;
};

LayerWeights tiny_layer_views(const TinyLayerStorage& storage) {
  const std::array<std::uint64_t, 1> norm{4};
  const std::array<std::uint64_t, 2> hidden_matrix{4, 4};
  const std::array<std::uint64_t, 2> kv_matrix{2, 4};
  const std::array<std::uint64_t, 2> gate_matrix{8, 4};
  const std::array<std::uint64_t, 2> down_matrix{4, 8};
  return LayerWeights{
      const_view(std::span<const float>(storage.input_norm), norm),
      AttentionWeights{
          const_view(std::span<const float>(storage.query), hidden_matrix),
          const_view(std::span<const float>(storage.key), kv_matrix),
          const_view(std::span<const float>(storage.value), kv_matrix),
          const_view(std::span<const float>(storage.output), hidden_matrix),
      },
      const_view(std::span<const float>(storage.post_norm), norm),
      MlpWeights{
          const_view(std::span<const float>(storage.gate), gate_matrix),
          const_view(std::span<const float>(storage.up), gate_matrix),
          const_view(std::span<const float>(storage.down), down_matrix),
      },
  };
}

MF_TEST(zero_projection_decoder_layer_is_identity_and_allocation_free) {
  TinyLayerStorage storage{
      std::vector<float>(4, 1.0F),  std::vector<float>(16, 0.0F),
      std::vector<float>(8, 0.0F),  std::vector<float>(8, 0.0F),
      std::vector<float>(16, 0.0F), std::vector<float>(4, 1.0F),
      std::vector<float>(32, 0.0F), std::vector<float>(32, 0.0F),
      std::vector<float>(32, 0.0F),
  };
  const auto weights = tiny_layer_views(storage);
  const DecoderLayerSpec spec{
      4, 8, 2, 1, 2, 1.0e-5F, RopeSpec{10'000.0F, 16},
  };
  const std::array<std::uint64_t, 3> hidden_shape{2, 2, 4};
  const std::array<std::uint64_t, 4> cache_shape{2, 4, 1, 2};
  std::array<float, 16> hidden{
      1.0F,  2.0F,  3.0F, 4.0F, -1.0F, 0.0F, 1.0F, 2.0F,
      -2.0F, -1.0F, 0.0F, 1.0F, 4.0F,  3.0F, 2.0F, 1.0F,
  };
  const auto original = hidden;
  std::array<float, 16> key_cache{};
  std::array<float, 16> value_cache{};
  std::array<float, 16> after_attention{};
  std::array<float, 16> after_mlp{};
  const std::array<std::uint32_t, 4> positions{0, 1, 0, 1};
  const std::array<std::uint32_t, 2> contexts{2, 2};
  const auto required =
      marketforge::required_decoder_layer_workspace_floats(spec, 2, 2, 4);
  auto workspace = CpuWorkspace::allocate(required.value()).value();
  float* const stable_workspace = workspace.floats().data();
  DecoderLayerTrace trace{
      view(std::span<float>(after_attention), hidden_shape),
      view(std::span<float>(after_mlp), hidden_shape),
  };
  for (std::size_t iteration = 0; iteration < 1'000; ++iteration) {
    hidden = original;
    MF_CHECK(marketforge::smollm2_decoder_layer_f32(
                 weights, spec, view(std::span<float>(hidden), hidden_shape),
                 CpuKvView{
                     view(std::span<float>(key_cache), cache_shape),
                     view(std::span<float>(value_cache), cache_shape),
                 },
                 positions, contexts, workspace, &trace)
                 .ok());
    MF_CHECK_EQ(workspace.floats().data(), stable_workspace);
    MF_CHECK_EQ(hidden, original);
    MF_CHECK_EQ(after_attention, original);
    MF_CHECK_EQ(after_mlp, original);
  }
}

MF_TEST(decoder_layer_faults_leave_hidden_unchanged) {
  TinyLayerStorage storage{
      std::vector<float>(4, 1.0F),  std::vector<float>(16, 0.0F),
      std::vector<float>(8, 0.0F),  std::vector<float>(8, 0.0F),
      std::vector<float>(16, 0.0F), std::vector<float>(4, 1.0F),
      std::vector<float>(32, 0.0F), std::vector<float>(32, 0.0F),
      std::vector<float>(32, 0.0F),
  };
  const auto weights = tiny_layer_views(storage);
  const DecoderLayerSpec spec{
      4, 8, 2, 1, 2, 1.0e-5F, RopeSpec{10'000.0F, 4},
  };
  const std::array<std::uint64_t, 3> hidden_shape{1, 2, 4};
  const std::array<std::uint64_t, 4> cache_shape{1, 2, 1, 2};
  std::array<float, 8> hidden{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  const auto original = hidden;
  std::array<float, 4> key_cache{};
  std::array<float, 4> value_cache{};
  const std::array<std::uint32_t, 2> positions{0, 1};
  const std::array<std::uint32_t, 1> contexts{2};
  const auto required =
      marketforge::required_decoder_layer_workspace_floats(spec, 1, 2, 2);
  auto insufficient = CpuWorkspace::allocate(required.value() - 1).value();
  const auto rejected = marketforge::smollm2_decoder_layer_f32(
      weights, spec, view(std::span<float>(hidden), hidden_shape),
      CpuKvView{
          view(std::span<float>(key_cache), cache_shape),
          view(std::span<float>(value_cache), cache_shape),
      },
      positions, contexts, insufficient);
  MF_CHECK(!rejected.ok());
  MF_CHECK_EQ(rejected.code, ErrorCode::insufficient_memory);
  MF_CHECK_EQ(hidden, original);
  check_all_equal(key_cache, 0.0F);
  check_all_equal(value_cache, 0.0F);

  auto workspace = CpuWorkspace::allocate(required.value()).value();
  const std::array<std::uint32_t, 2> invalid_positions{0, 2};
  MF_CHECK(!marketforge::smollm2_decoder_layer_f32(
                weights, spec, view(std::span<float>(hidden), hidden_shape),
                CpuKvView{
                    view(std::span<float>(key_cache), cache_shape),
                    view(std::span<float>(value_cache), cache_shape),
                },
                invalid_positions, contexts, workspace)
                .ok());
  MF_CHECK_EQ(hidden, original);

  DecoderLayerTrace illegal_trace{
      view(std::span<float>(hidden), hidden_shape),
      view(std::span<float>(hidden), hidden_shape),
  };
  MF_CHECK(!marketforge::smollm2_decoder_layer_f32(
                weights, spec, view(std::span<float>(hidden), hidden_shape),
                CpuKvView{
                    view(std::span<float>(key_cache), cache_shape),
                    view(std::span<float>(value_cache), cache_shape),
                },
                positions, contexts, workspace, &illegal_trace)
                .ok());
  MF_CHECK_EQ(hidden, original);
}

} // namespace
