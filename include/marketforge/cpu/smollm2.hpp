#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "marketforge/core/host_buffer.hpp"
#include "marketforge/cpu/operators.hpp"
#include "marketforge/model/loaded_weights.hpp"
#include "marketforge/model/model_spec.hpp"

namespace marketforge {

struct GreedyToken {
  std::uint32_t token_id{0};
  float logit{0.0F};
  float runner_up_logit{0.0F};
};

struct CpuSmolLm2Memory {
  std::uint64_t fp32_weight_bytes{0};
  std::uint64_t kv_bytes{0};
  std::uint64_t execution_bytes{0};
  std::uint64_t workspace_bytes{0};
  std::uint64_t position_bytes{0};
  std::uint64_t layer_view_bytes{0};
  std::uint64_t total_owned_bytes{0};

  friend constexpr bool operator==(const CpuSmolLm2Memory&,
                                   const CpuSmolLm2Memory&) = default;
};

class CpuSmolLm2 {
public:
  [[nodiscard]] static Result<CpuSmolLm2>
  load(const std::filesystem::path& checkpoint, std::uint32_t maximum_context,
       std::uint32_t maximum_prefill_tokens);

  CpuSmolLm2(CpuSmolLm2&&) noexcept = default;
  CpuSmolLm2& operator=(CpuSmolLm2&&) noexcept = default;
  CpuSmolLm2(const CpuSmolLm2&) = delete;
  CpuSmolLm2& operator=(const CpuSmolLm2&) = delete;

  [[nodiscard]] Status reset() noexcept;
  [[nodiscard]] Result<GreedyToken>
  prefill(std::span<const std::uint32_t> token_ids) noexcept;
  [[nodiscard]] Result<GreedyToken> decode(std::uint32_t token_id) noexcept;

  [[nodiscard]] std::span<const float> logits() const noexcept;
  [[nodiscard]] std::uint32_t context_length() const noexcept {
    return context_length_;
  }
  [[nodiscard]] std::uint32_t maximum_context() const noexcept {
    return maximum_context_;
  }
  [[nodiscard]] std::uint32_t maximum_prefill_tokens() const noexcept {
    return maximum_prefill_tokens_;
  }
  [[nodiscard]] CpuSmolLm2Memory memory() const noexcept { return memory_; }

private:
  CpuSmolLm2(ModelSpec spec, HostBuffer fp32_weights, ConstTensorView embedding,
             ConstTensorView final_norm, ConstTensorView output_head,
             std::vector<LayerWeights> layers, HostBuffer kv,
             HostBuffer execution, CpuWorkspace workspace,
             std::vector<std::uint32_t> positions,
             std::uint32_t maximum_context,
             std::uint32_t maximum_prefill_tokens,
             CpuSmolLm2Memory memory) noexcept;

  [[nodiscard]] Result<GreedyToken>
  forward(std::span<const std::uint32_t> token_ids) noexcept;

  ModelSpec spec_;
  HostBuffer fp32_weights_;
  ConstTensorView embedding_;
  ConstTensorView final_norm_;
  ConstTensorView output_head_;
  std::vector<LayerWeights> layers_;
  HostBuffer kv_;
  HostBuffer execution_;
  CpuWorkspace workspace_;
  std::vector<std::uint32_t> positions_;
  std::uint32_t maximum_context_{0};
  std::uint32_t maximum_prefill_tokens_{0};
  std::uint32_t context_length_{0};
  CpuSmolLm2Memory memory_;
};

} // namespace marketforge
