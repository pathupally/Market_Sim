#include "marketforge/cpu/smollm2.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "marketforge/core/checked_math.hpp"
#include "marketforge/core/shape.hpp"

namespace marketforge {
namespace {

struct MaterializedWeights {
  HostBuffer storage;
  ConstTensorView embedding;
  ConstTensorView final_norm;
  ConstTensorView output_head;
  std::vector<LayerWeights> layers;
};

Result<std::uint64_t>
checked_sum(const std::initializer_list<std::uint64_t> values) noexcept {
  std::uint64_t total = 0;
  for (const auto value : values) {
    const auto next = checked_add(total, value);
    if (!next) {
      return next;
    }
    total = next.value();
  }
  return Result<std::uint64_t>::success(total);
}

Result<std::uint64_t>
multiply(const std::initializer_list<std::uint64_t> factors) noexcept {
  std::uint64_t value = 1;
  for (const auto factor : factors) {
    const auto next = checked_multiply(value, factor);
    if (!next) {
      return next;
    }
    value = next.value();
  }
  return Result<std::uint64_t>::success(value);
}

float decode_bf16(const std::byte* const bytes) noexcept {
  const auto low =
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[0]));
  const auto high =
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[1]));
  const std::uint16_t bf16 =
      static_cast<std::uint16_t>(low | static_cast<std::uint16_t>(high << 8U));
  return std::bit_cast<float>(static_cast<std::uint32_t>(bf16) << 16U);
}

Result<MaterializedWeights> materialize_smollm2(const LoadedWeights& source,
                                                const ModelSpec& spec) {
  if (!source.output_head_aliases_embedding() ||
      source.layers().size() != spec.layers) {
    return Result<MaterializedWeights>::failure(
        Status::failure(ErrorCode::invalid_model));
  }
  const auto parameter_count = estimate_parameter_count(spec);
  if (!parameter_count) {
    return Result<MaterializedWeights>::failure(parameter_count.status());
  }
  const auto storage_bytes =
      checked_multiply(parameter_count.value(), sizeof(float));
  if (!storage_bytes) {
    return Result<MaterializedWeights>::failure(storage_bytes.status());
  }
  auto storage_result = HostBuffer::allocate(storage_bytes.value(), 64);
  if (!storage_result) {
    return Result<MaterializedWeights>::failure(storage_result.status());
  }
  HostBuffer storage = std::move(storage_result).value();
  auto* const destination = reinterpret_cast<float*>(storage.bytes().data());
  std::uint64_t cursor = 0;

  const auto append =
      [&](const ConstTensorView tensor) -> Result<ConstTensorView> {
    if (tensor.dtype != DType::bf16 || tensor.memory != MemoryKind::host) {
      return Result<ConstTensorView>::failure(
          Status::failure(ErrorCode::invalid_tensor));
    }
    const auto count = checked_numel(tensor.shape);
    if (!count || cursor > parameter_count.value() ||
        count.value() > parameter_count.value() - cursor) {
      return Result<ConstTensorView>::failure(
          Status::failure(ErrorCode::invalid_tensor));
    }
    const auto* const source_bytes = static_cast<const std::byte*>(tensor.data);
    if (count.value() != 0 && source_bytes == nullptr) {
      return Result<ConstTensorView>::failure(
          Status::failure(ErrorCode::invalid_tensor));
    }
    const std::uint64_t begin = cursor;
    for (std::uint64_t index = 0; index < count.value(); ++index) {
      const float value =
          decode_bf16(source_bytes + static_cast<std::size_t>(index * 2));
      if (!std::isfinite(value)) {
        return Result<ConstTensorView>::failure(
            Status::failure(ErrorCode::invalid_tensor));
      }
      destination[static_cast<std::size_t>(cursor++)] = value;
    }
    return Result<ConstTensorView>::success(ConstTensorView{
        destination + static_cast<std::size_t>(begin),
        tensor.shape,
        DType::f32,
        MemoryKind::host,
    });
  };

  auto embedding = append(source.embedding());
  if (!embedding) {
    return Result<MaterializedWeights>::failure(embedding.status());
  }
  std::vector<LayerWeights> layers;
  layers.reserve(spec.layers);
  for (const auto& source_layer : source.layers()) {
    auto input_norm = append(source_layer.input_norm);
    auto query = append(source_layer.attention.query);
    auto key = append(source_layer.attention.key);
    auto value = append(source_layer.attention.value);
    auto output = append(source_layer.attention.output);
    auto post_norm = append(source_layer.post_attention_norm);
    auto gate = append(source_layer.mlp.gate);
    auto up = append(source_layer.mlp.up);
    auto down = append(source_layer.mlp.down);
    if (!input_norm || !query || !key || !value || !output || !post_norm ||
        !gate || !up || !down) {
      return Result<MaterializedWeights>::failure(
          Status::failure(ErrorCode::invalid_tensor));
    }
    layers.push_back(LayerWeights{
        input_norm.value(),
        AttentionWeights{
            query.value(),
            key.value(),
            value.value(),
            output.value(),
        },
        post_norm.value(),
        MlpWeights{
            gate.value(),
            up.value(),
            down.value(),
        },
    });
  }
  auto final_norm = append(source.final_norm());
  if (!final_norm || cursor != parameter_count.value()) {
    return Result<MaterializedWeights>::failure(
        Status::failure(ErrorCode::invalid_tensor));
  }

  return Result<MaterializedWeights>::success(MaterializedWeights{
      std::move(storage),
      embedding.value(),
      final_norm.value(),
      embedding.value(),
      std::move(layers),
  });
}

TensorView f32_view(float* const data,
                    const std::span<const std::uint64_t> extents) {
  return TensorView{
      data,
      make_shape(extents).value(),
      DType::f32,
      MemoryKind::host,
  };
}

ConstTensorView const_f32_view(const float* const data,
                               const std::span<const std::uint64_t> extents) {
  return ConstTensorView{
      data,
      make_shape(extents).value(),
      DType::f32,
      MemoryKind::host,
  };
}

} // namespace

CpuSmolLm2::CpuSmolLm2(ModelSpec spec, HostBuffer fp32_weights,
                       const ConstTensorView embedding,
                       const ConstTensorView final_norm,
                       const ConstTensorView output_head,
                       std::vector<LayerWeights> layers, HostBuffer kv,
                       HostBuffer execution, CpuWorkspace workspace,
                       std::vector<std::uint32_t> positions,
                       const std::uint32_t maximum_context,
                       const std::uint32_t maximum_prefill_tokens,
                       const CpuSmolLm2Memory memory) noexcept
    : spec_(spec), fp32_weights_(std::move(fp32_weights)),
      embedding_(embedding), final_norm_(final_norm), output_head_(output_head),
      layers_(std::move(layers)), kv_(std::move(kv)),
      execution_(std::move(execution)), workspace_(std::move(workspace)),
      positions_(std::move(positions)), maximum_context_(maximum_context),
      maximum_prefill_tokens_(maximum_prefill_tokens), memory_(memory) {}

Result<CpuSmolLm2>
CpuSmolLm2::load(const std::filesystem::path& checkpoint,
                 const std::uint32_t maximum_context,
                 const std::uint32_t maximum_prefill_tokens) {
  const ModelSpec spec = smollm2_135m_profile().spec;
  if (maximum_context == 0 || maximum_context > spec.max_positions ||
      maximum_prefill_tokens == 0 || maximum_prefill_tokens > maximum_context) {
    return Result<CpuSmolLm2>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }

  auto mapped = LoadedWeights::open_and_bind(checkpoint, spec);
  if (!mapped) {
    return Result<CpuSmolLm2>::failure(mapped.status());
  }
  auto materialized = materialize_smollm2(mapped.value(), spec);
  if (!materialized) {
    return Result<CpuSmolLm2>::failure(materialized.status());
  }

  const auto kv_floats =
      multiply({2, spec.layers, maximum_context, spec.kv_heads, spec.head_dim});
  const auto kv_bytes =
      kv_floats ? checked_multiply(kv_floats.value(), sizeof(float))
                : Result<std::uint64_t>::failure(kv_floats.status());
  const auto hidden_floats =
      checked_multiply(maximum_prefill_tokens, spec.hidden_size);
  const auto execution_floats =
      hidden_floats ? checked_sum({hidden_floats.value(), spec.hidden_size,
                                   spec.vocabulary_size})
                    : Result<std::uint64_t>::failure(hidden_floats.status());
  const auto execution_bytes =
      execution_floats
          ? checked_multiply(execution_floats.value(), sizeof(float))
          : Result<std::uint64_t>::failure(execution_floats.status());
  const auto required_workspace = required_decoder_layer_workspace_floats(
      DecoderLayerSpec{
          spec.hidden_size,
          spec.intermediate_size,
          spec.query_heads,
          spec.kv_heads,
          spec.head_dim,
          spec.rms_norm_epsilon,
          RopeSpec{spec.rope_theta, spec.max_positions},
      },
      1, maximum_prefill_tokens, maximum_context);
  const auto workspace_bytes =
      required_workspace
          ? checked_multiply(required_workspace.value(), sizeof(float))
          : Result<std::uint64_t>::failure(required_workspace.status());
  if (!kv_bytes || !execution_bytes || !required_workspace ||
      !workspace_bytes) {
    return Result<CpuSmolLm2>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }

  auto kv = HostBuffer::allocate(kv_bytes.value(), 64);
  auto execution = HostBuffer::allocate(execution_bytes.value(), 64);
  auto workspace = CpuWorkspace::allocate(required_workspace.value());
  if (!kv || !execution || !workspace) {
    const Status status =
        !kv ? kv.status()
            : (!execution ? execution.status() : workspace.status());
    return Result<CpuSmolLm2>::failure(status);
  }
  std::memset(kv.value().bytes().data(), 0, kv.value().bytes().size());
  std::memset(execution.value().bytes().data(), 0,
              execution.value().bytes().size());
  std::vector<std::uint32_t> positions(maximum_prefill_tokens);

  const std::uint64_t weight_bytes =
      materialized.value().storage.bytes().size();
  const auto position_bytes =
      checked_multiply(positions.capacity(), sizeof(std::uint32_t));
  const auto layer_view_bytes = checked_multiply(
      materialized.value().layers.capacity(), sizeof(LayerWeights));
  if (!position_bytes || !layer_view_bytes) {
    return Result<CpuSmolLm2>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  const auto total_owned =
      checked_sum({weight_bytes, kv_bytes.value(), execution_bytes.value(),
                   workspace_bytes.value(), position_bytes.value(),
                   layer_view_bytes.value()});
  if (!total_owned) {
    return Result<CpuSmolLm2>::failure(total_owned.status());
  }
  const CpuSmolLm2Memory memory{
      weight_bytes,
      kv_bytes.value(),
      execution_bytes.value(),
      workspace_bytes.value(),
      position_bytes.value(),
      layer_view_bytes.value(),
      total_owned.value(),
  };

  auto weights = std::move(materialized).value();
  return Result<CpuSmolLm2>::success(CpuSmolLm2{
      spec,
      std::move(weights.storage),
      weights.embedding,
      weights.final_norm,
      weights.output_head,
      std::move(weights.layers),
      std::move(kv).value(),
      std::move(execution).value(),
      std::move(workspace).value(),
      std::move(positions),
      maximum_context,
      maximum_prefill_tokens,
      memory,
  });
}

Status CpuSmolLm2::reset() noexcept {
  std::memset(kv_.bytes().data(), 0, kv_.bytes().size());
  context_length_ = 0;
  return Status::success();
}

Result<GreedyToken>
CpuSmolLm2::prefill(const std::span<const std::uint32_t> token_ids) noexcept {
  if (context_length_ != 0 || token_ids.empty() ||
      token_ids.size() > maximum_prefill_tokens_) {
    return Result<GreedyToken>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return forward(token_ids);
}

Result<GreedyToken> CpuSmolLm2::decode(const std::uint32_t token_id) noexcept {
  if (context_length_ == 0) {
    return Result<GreedyToken>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return forward(std::span<const std::uint32_t>(&token_id, 1));
}

std::span<const float> CpuSmolLm2::logits() const noexcept {
  const auto hidden_capacity =
      static_cast<std::uint64_t>(maximum_prefill_tokens_) * spec_.hidden_size;
  const auto* const execution =
      reinterpret_cast<const float*>(execution_.bytes().data());
  return {execution + hidden_capacity + spec_.hidden_size,
          spec_.vocabulary_size};
}

Result<GreedyToken>
CpuSmolLm2::forward(const std::span<const std::uint32_t> token_ids) noexcept {
  if (token_ids.empty() || token_ids.size() > maximum_prefill_tokens_ ||
      token_ids.size() > maximum_context_ - context_length_) {
    return Result<GreedyToken>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  for (const auto token : token_ids) {
    if (token >= spec_.vocabulary_size) {
      return Result<GreedyToken>::failure(
          Status::failure(ErrorCode::invalid_argument, token));
    }
  }

  const std::uint64_t token_count = token_ids.size();
  const std::uint64_t hidden_size = spec_.hidden_size;
  const std::uint64_t hidden_capacity =
      static_cast<std::uint64_t>(maximum_prefill_tokens_) * hidden_size;
  auto* const execution = reinterpret_cast<float*>(execution_.bytes().data());
  float* const hidden = execution;
  float* const normalized_last = execution + hidden_capacity;
  float* const logits_data = normalized_last + hidden_size;
  const auto* const embedding = static_cast<const float*>(embedding_.data);

  for (std::uint64_t token = 0; token < token_count; ++token) {
    const auto token_id = token_ids[static_cast<std::size_t>(token)];
    std::memcpy(hidden + token * hidden_size,
                embedding + static_cast<std::uint64_t>(token_id) * hidden_size,
                static_cast<std::size_t>(hidden_size * sizeof(float)));
    positions_[static_cast<std::size_t>(token)] =
        context_length_ + static_cast<std::uint32_t>(token);
  }
  const std::array<std::uint32_t, 1> context_lengths{
      context_length_ + static_cast<std::uint32_t>(token_count)};
  const std::array<std::uint64_t, 3> hidden_shape{1, token_count, hidden_size};
  const std::array<std::uint64_t, 4> cache_shape{
      1, maximum_context_, spec_.kv_heads, spec_.head_dim};
  const DecoderLayerSpec layer_spec{
      spec_.hidden_size,
      spec_.intermediate_size,
      spec_.query_heads,
      spec_.kv_heads,
      spec_.head_dim,
      spec_.rms_norm_epsilon,
      RopeSpec{spec_.rope_theta, spec_.max_positions},
  };
  const std::uint64_t cache_floats =
      static_cast<std::uint64_t>(maximum_context_) * spec_.kv_heads *
      spec_.head_dim;
  auto* const kv = reinterpret_cast<float*>(kv_.bytes().data());
  auto hidden_view = f32_view(hidden, hidden_shape);
  for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
    const std::uint64_t layer_offset =
        static_cast<std::uint64_t>(layer) * 2 * cache_floats;
    const CpuKvView cache{
        f32_view(kv + layer_offset, cache_shape),
        f32_view(kv + layer_offset + cache_floats, cache_shape),
    };
    const auto status = smollm2_decoder_layer_f32(
        layers_[layer], layer_spec, hidden_view, cache,
        std::span<const std::uint32_t>(positions_.data(), token_count),
        context_lengths, workspace_);
    if (!status.ok()) {
      return Result<GreedyToken>::failure(status);
    }
  }

  const std::array<std::uint64_t, 2> one_hidden{1, hidden_size};
  const std::array<std::uint64_t, 2> one_vocabulary{1, spec_.vocabulary_size};
  const float* const last_hidden = hidden + (token_count - 1) * hidden_size;
  auto status = rms_norm_f32(const_f32_view(last_hidden, one_hidden),
                             final_norm_, spec_.rms_norm_epsilon,
                             f32_view(normalized_last, one_hidden));
  if (status.ok()) {
    status = linear_f32(const_f32_view(normalized_last, one_hidden),
                        output_head_, f32_view(logits_data, one_vocabulary));
  }
  if (!status.ok()) {
    return Result<GreedyToken>::failure(status);
  }

  std::uint32_t best_token = 0;
  float best = -std::numeric_limits<float>::infinity();
  float runner_up = -std::numeric_limits<float>::infinity();
  for (std::uint32_t token = 0; token < spec_.vocabulary_size; ++token) {
    const float value = logits_data[token];
    if (!std::isfinite(value)) {
      return Result<GreedyToken>::failure(
          Status::failure(ErrorCode::invalid_argument, token));
    }
    if (value > best) {
      runner_up = best;
      best = value;
      best_token = token;
    } else if (value > runner_up) {
      runner_up = value;
    }
  }
  context_length_ += static_cast<std::uint32_t>(token_count);
  return Result<GreedyToken>::success(GreedyToken{best_token, best, runner_up});
}

} // namespace marketforge
