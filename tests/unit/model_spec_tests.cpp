#include "test_support.hpp"

#include <cstdint>
#include <limits>

#include "marketforge/model/model_spec.hpp"

namespace {

using marketforge::Architecture;
using marketforge::DType;
using marketforge::ErrorCode;
using marketforge::ModelSpec;
using marketforge::RuntimeMemoryPlan;

constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t mib = 1024ULL * 1024ULL;

MF_TEST(locked_model_specs_validate) {
  const auto smol = marketforge::smollm2_135m_profile();
  const auto qwen = marketforge::qwen2_5_0_5b_profile();

  MF_CHECK(marketforge::validate(smol.spec).ok());
  MF_CHECK(marketforge::validate(qwen.spec).ok());
  MF_CHECK_EQ(smol.repository, "HuggingFaceTB/SmolLM2-135M");
  MF_CHECK_EQ(qwen.repository, "Qwen/Qwen2.5-0.5B-Instruct");
  MF_CHECK_EQ(smol.revision.size(), 40U);
  MF_CHECK_EQ(qwen.revision.size(), 40U);
}

MF_TEST(parameter_counts_match_locked_architectures) {
  const auto smol_profile = marketforge::smollm2_135m_profile();
  const auto qwen_profile = marketforge::qwen2_5_0_5b_profile();
  const auto smol = marketforge::estimate_parameter_count(smol_profile.spec);
  const auto qwen = marketforge::estimate_parameter_count(qwen_profile.spec);

  MF_CHECK(smol);
  MF_CHECK(qwen);
  MF_CHECK_EQ(smol.value(), 134'515'008ULL);
  MF_CHECK_EQ(qwen.value(), 494'032'768ULL);
}

MF_TEST(footprints_match_bf16_fp32_and_fp16_kv_math) {
  const auto smol = marketforge::estimate_model_footprint(
      marketforge::smollm2_135m_profile().spec, DType::bf16, DType::f32,
      DType::f16);
  const auto qwen = marketforge::estimate_model_footprint(
      marketforge::qwen2_5_0_5b_profile().spec, DType::bf16, DType::f32,
      DType::f16);

  MF_CHECK(smol);
  MF_CHECK(qwen);
  MF_CHECK_EQ(smol.value().source_weight_bytes, 269'030'016ULL);
  MF_CHECK_EQ(smol.value().materialized_weight_bytes, 538'060'032ULL);
  MF_CHECK_EQ(smol.value().peak_cpu_weight_bytes, 807'090'048ULL);
  MF_CHECK_EQ(smol.value().kv_bytes_per_token, 23'040ULL);

  MF_CHECK_EQ(qwen.value().source_weight_bytes, 988'065'536ULL);
  MF_CHECK_EQ(qwen.value().materialized_weight_bytes, 1'976'131'072ULL);
  MF_CHECK_EQ(qwen.value().peak_cpu_weight_bytes, 2'964'196'608ULL);
  MF_CHECK_EQ(qwen.value().kv_bytes_per_token, 12'288ULL);

  MF_CHECK(smol.value().source_weight_bytes <
           marketforge::smollm2_135m_profile().max_checkpoint_bytes);
  MF_CHECK(qwen.value().source_weight_bytes <
           marketforge::qwen2_5_0_5b_profile().max_checkpoint_bytes);
}

MF_TEST(both_models_fit_the_local_36_gib_development_profile) {
  const auto smol = marketforge::estimate_model_footprint(
                        marketforge::smollm2_135m_profile().spec, DType::bf16,
                        DType::f32, DType::f16)
                        .value();
  const auto qwen = marketforge::estimate_model_footprint(
                        marketforge::qwen2_5_0_5b_profile().spec, DType::bf16,
                        DType::f32, DType::f16)
                        .value();

  const RuntimeMemoryPlan ten_thousand_short_agents{
      10'000ULL * 40ULL,
      1ULL * gib,
      4ULL * gib,
  };
  MF_CHECK(marketforge::fits_memory_budget(smol, ten_thousand_short_agents,
                                           36ULL * gib)
               .value());
  MF_CHECK(marketforge::fits_memory_budget(qwen, ten_thousand_short_agents,
                                           36ULL * gib)
               .value());

  const RuntimeMemoryPlan ten_thousand_long_contexts{
      10'000ULL * 1'024ULL,
      1ULL * gib,
      4ULL * gib,
  };
  MF_CHECK(!marketforge::fits_memory_budget(smol, ten_thousand_long_contexts,
                                            36ULL * gib)
                .value());
  MF_CHECK(!marketforge::fits_memory_budget(qwen, ten_thousand_long_contexts,
                                            36ULL * gib)
                .value());
}

MF_TEST(model_validation_rejects_invalid_dimensions) {
  const ModelSpec valid = marketforge::smollm2_135m_profile().spec;

  auto changed = valid;
  changed.layers = 0;
  MF_CHECK_EQ(marketforge::validate(changed).code, ErrorCode::invalid_model);

  changed = valid;
  changed.query_heads = 8;
  MF_CHECK_EQ(marketforge::validate(changed).code, ErrorCode::invalid_model);
  MF_CHECK(!marketforge::estimate_parameter_count(changed));

  changed = valid;
  changed.kv_heads = 2;
  MF_CHECK_EQ(marketforge::validate(changed).code, ErrorCode::invalid_model);

  changed = valid;
  changed.rms_norm_epsilon = 0.0F;
  MF_CHECK_EQ(marketforge::validate(changed).code, ErrorCode::invalid_model);

  changed = valid;
  changed.rope_theta = std::numeric_limits<float>::infinity();
  MF_CHECK_EQ(marketforge::validate(changed).code, ErrorCode::invalid_model);

  changed = valid;
  changed.architecture = static_cast<Architecture>(255);
  MF_CHECK_EQ(marketforge::validate(changed).code, ErrorCode::invalid_model);
}

MF_TEST(untied_embeddings_add_a_second_vocabulary_matrix) {
  auto spec = marketforge::smollm2_135m_profile().spec;
  const auto tied = marketforge::estimate_parameter_count(spec).value();
  spec.tied_embeddings = false;
  const auto untied = marketforge::estimate_parameter_count(spec).value();
  MF_CHECK_EQ(untied - tied, static_cast<std::uint64_t>(spec.vocabulary_size) *
                                 spec.hidden_size);
}

MF_TEST(memory_estimates_report_overflow) {
  const auto footprint = marketforge::estimate_model_footprint(
                             marketforge::smollm2_135m_profile().spec,
                             DType::bf16, DType::f32, DType::f16)
                             .value();
  const RuntimeMemoryPlan impossible{
      std::numeric_limits<std::uint64_t>::max(),
      0,
      0,
  };
  const auto result =
      marketforge::estimate_peak_runtime_bytes(footprint, impossible);
  MF_CHECK(!result);
  MF_CHECK_EQ(result.status().code, ErrorCode::arithmetic_overflow);
}

MF_TEST(checkpoint_budgets_are_small_relative_to_local_disk) {
  MF_CHECK(marketforge::smollm2_135m_profile().max_checkpoint_bytes <
           512ULL * mib);
  MF_CHECK(marketforge::qwen2_5_0_5b_profile().max_checkpoint_bytes <
           2ULL * gib);
}

} // namespace
