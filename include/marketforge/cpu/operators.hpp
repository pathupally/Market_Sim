#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "marketforge/core/host_buffer.hpp"
#include "marketforge/core/tensor_view.hpp"
#include "marketforge/model/loaded_weights.hpp"

namespace marketforge {

struct RopeSpec {
  float theta{10'000.0F};
  std::uint32_t max_positions{0};
};

struct CpuKvView {
  TensorView key;
  TensorView value;
};

struct DecoderLayerSpec {
  std::uint32_t hidden_size{0};
  std::uint32_t intermediate_size{0};
  std::uint32_t query_heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{0};
  float rms_norm_epsilon{0.0F};
  RopeSpec rope;
};

struct DecoderLayerTrace {
  TensorView after_attention;
  TensorView after_mlp;
};

class CpuWorkspace {
public:
  [[nodiscard]] static Result<CpuWorkspace>
  allocate(std::uint64_t float_capacity) noexcept;

  CpuWorkspace(CpuWorkspace&&) noexcept = default;
  CpuWorkspace& operator=(CpuWorkspace&&) noexcept = default;
  CpuWorkspace(const CpuWorkspace&) = delete;
  CpuWorkspace& operator=(const CpuWorkspace&) = delete;

  [[nodiscard]] std::span<float> floats() noexcept {
    return {reinterpret_cast<float*>(storage_.bytes().data()), capacity_};
  }

  [[nodiscard]] std::span<const float> floats() const noexcept {
    return {reinterpret_cast<const float*>(storage_.bytes().data()), capacity_};
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
  CpuWorkspace(HostBuffer storage, std::size_t capacity) noexcept
      : storage_(std::move(storage)), capacity_(capacity) {}

  HostBuffer storage_;
  std::size_t capacity_{0};
};

[[nodiscard]] Status rms_norm_f32(ConstTensorView input, ConstTensorView weight,
                                  float epsilon, TensorView output) noexcept;

[[nodiscard]] Status linear_f32(ConstTensorView input, ConstTensorView weight,
                                TensorView output) noexcept;

[[nodiscard]] Status softmax_f32(ConstTensorView input,
                                 std::span<const std::uint32_t> valid_lengths,
                                 TensorView output) noexcept;

[[nodiscard]] Status apply_rope_f32(TensorView query, TensorView key,
                                    std::span<const std::uint32_t> positions,
                                    const RopeSpec& spec) noexcept;

[[nodiscard]] Status append_kv_f32(ConstTensorView key, ConstTensorView value,
                                   std::span<const std::uint32_t> positions,
                                   CpuKvView cache) noexcept;

[[nodiscard]] Result<std::uint64_t>
required_attention_workspace_floats(ConstTensorView query,
                                    ConstTensorView key_cache) noexcept;

[[nodiscard]] Status
attention_f32(ConstTensorView query, ConstTensorView key_cache,
              ConstTensorView value_cache,
              std::span<const std::uint32_t> context_lengths, TensorView output,
              CpuWorkspace& workspace) noexcept;

[[nodiscard]] Result<std::uint64_t> required_decoder_layer_workspace_floats(
    const DecoderLayerSpec& spec, std::uint64_t batch_size,
    std::uint64_t query_tokens, std::uint64_t maximum_context) noexcept;

[[nodiscard]] Status smollm2_decoder_layer_f32(
    const LayerWeights& weights, const DecoderLayerSpec& spec,
    TensorView hidden, CpuKvView cache,
    std::span<const std::uint32_t> positions,
    std::span<const std::uint32_t> context_lengths, CpuWorkspace& workspace,
    DecoderLayerTrace* trace = nullptr) noexcept;

} // namespace marketforge
