#pragma once

#include <cstdint>
#include <string_view>

#include "marketforge/core/dtype.hpp"

namespace marketforge {

enum class Architecture : std::uint8_t {
  smollm2_llama,
  qwen2,
};

struct ModelSpec {
  Architecture architecture{Architecture::smollm2_llama};
  std::uint32_t layers{0};
  std::uint32_t hidden_size{0};
  std::uint32_t intermediate_size{0};
  std::uint32_t query_heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t vocabulary_size{0};
  std::uint32_t max_positions{0};
  float rms_norm_epsilon{0.0F};
  float rope_theta{0.0F};
  bool qkv_bias{false};
  bool tied_embeddings{true};

  friend constexpr bool operator==(const ModelSpec&,
                                   const ModelSpec&) = default;
};

struct ModelProfile {
  std::string_view name;
  std::string_view repository;
  std::string_view revision;
  ModelSpec spec;
  std::uint64_t max_checkpoint_bytes;
};

struct ModelFootprint {
  std::uint64_t parameter_count;
  std::uint64_t source_weight_bytes;
  std::uint64_t materialized_weight_bytes;
  std::uint64_t peak_cpu_weight_bytes;
  std::uint64_t kv_bytes_per_token;
};

struct RuntimeMemoryPlan {
  std::uint64_t resident_kv_tokens{0};
  std::uint64_t workspace_bytes{0};
  std::uint64_t reserve_bytes{0};
};

[[nodiscard]] constexpr ModelProfile smollm2_135m_profile() noexcept {
  return ModelProfile{
      "SmolLM2-135M",
      "HuggingFaceTB/SmolLM2-135M",
      "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
      ModelSpec{
          Architecture::smollm2_llama,
          30,
          576,
          1'536,
          9,
          3,
          64,
          49'152,
          8'192,
          1.0e-5F,
          100'000.0F,
          false,
          true,
      },
      300ULL * 1024ULL * 1024ULL,
  };
}

[[nodiscard]] constexpr ModelProfile qwen2_5_0_5b_profile() noexcept {
  return ModelProfile{
      "Qwen2.5-0.5B-Instruct",
      "Qwen/Qwen2.5-0.5B-Instruct",
      "7ae557604adf67be50417f59c2c2f167def9a775",
      ModelSpec{
          Architecture::qwen2,
          24,
          896,
          4'864,
          14,
          2,
          64,
          151'936,
          32'768,
          1.0e-6F,
          1'000'000.0F,
          true,
          true,
      },
      1'100ULL * 1024ULL * 1024ULL,
  };
}

[[nodiscard]] Status validate(const ModelSpec& spec) noexcept;

[[nodiscard]] Result<std::uint64_t>
estimate_parameter_count(const ModelSpec& spec) noexcept;

[[nodiscard]] Result<ModelFootprint>
estimate_model_footprint(const ModelSpec& spec, DType source_weight_dtype,
                         DType materialized_weight_dtype,
                         DType kv_dtype) noexcept;

[[nodiscard]] Result<std::uint64_t>
estimate_peak_runtime_bytes(const ModelFootprint& footprint,
                            const RuntimeMemoryPlan& plan) noexcept;

[[nodiscard]] Result<bool>
fits_memory_budget(const ModelFootprint& footprint,
                   const RuntimeMemoryPlan& plan,
                   std::uint64_t available_bytes) noexcept;

} // namespace marketforge
