#include "marketforge/cpu/operators.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>

#include "marketforge/core/checked_math.hpp"
#include "marketforge/core/shape.hpp"

#if defined(MARKETFORGE_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace marketforge {
namespace {

struct MemoryRange {
  std::uintptr_t begin{0};
  std::uintptr_t end{0};
};

template <typename View>
Result<MemoryRange> memory_range(const View& view) noexcept {
  const auto bytes = checked_nbytes(view.shape, view.dtype);
  if (!bytes) {
    return Result<MemoryRange>::failure(bytes.status());
  }
  if (bytes.value() == 0) {
    return Result<MemoryRange>::success(MemoryRange{});
  }
  if (view.data == nullptr ||
      bytes.value() > std::numeric_limits<std::uintptr_t>::max()) {
    return Result<MemoryRange>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(view.data);
  if (begin > std::numeric_limits<std::uintptr_t>::max() -
                  static_cast<std::uintptr_t>(bytes.value())) {
    return Result<MemoryRange>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  return Result<MemoryRange>::success(
      MemoryRange{begin, begin + static_cast<std::uintptr_t>(bytes.value())});
}

bool overlaps(const MemoryRange left, const MemoryRange right) noexcept {
  if (left.begin == left.end || right.begin == right.end) {
    return false;
  }
  return left.begin < right.end && right.begin < left.end;
}

bool exactly_aliases(const MemoryRange left, const MemoryRange right) noexcept {
  return left.begin == right.begin && left.end == right.end;
}

template <typename View> Status validate_f32_host(const View& view) noexcept {
  if (view.dtype != DType::f32 || view.memory != MemoryKind::host) {
    return Status::failure(view.dtype != DType::f32
                               ? ErrorCode::unsupported_dtype
                               : ErrorCode::invalid_argument);
  }
  return validate_tensor_view(view);
}

Status require_shape(const Shape& shape, const std::uint8_t rank,
                     const std::span<const std::uint64_t> extents) noexcept {
  if (shape.rank != rank || extents.size() != rank) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  for (std::size_t axis = 0; axis < extents.size(); ++axis) {
    if (shape.extents[axis] != extents[axis]) {
      return Status::failure(ErrorCode::invalid_shape,
                             static_cast<std::uint32_t>(axis));
    }
  }
  return Status::success();
}

Status validate_output_alias(ConstTensorView input, TensorView output,
                             const bool allow_exact_alias) noexcept {
  const auto input_range = memory_range(input);
  const auto output_range = memory_range(output);
  if (!input_range || !output_range) {
    return !input_range ? input_range.status() : output_range.status();
  }
  if (overlaps(input_range.value(), output_range.value()) &&
      !(allow_exact_alias &&
        exactly_aliases(input_range.value(), output_range.value()))) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  return Status::success();
}

Status validate_no_overlap(ConstTensorView left, TensorView right) noexcept {
  return validate_output_alias(left, right, false);
}

Result<Shape> shape_from(const std::span<const std::uint64_t> extents) {
  return make_shape(extents);
}

ConstTensorView as_const(const TensorView view) noexcept {
  return ConstTensorView{view.data, view.shape, view.dtype, view.memory};
}

TensorView make_f32_view(float* data,
                         const std::span<const std::uint64_t> extents) {
  return TensorView{
      data,
      shape_from(extents).value(),
      DType::f32,
      MemoryKind::host,
  };
}

ConstTensorView
make_const_f32_view(const float* data,
                    const std::span<const std::uint64_t> extents) {
  return ConstTensorView{
      data,
      shape_from(extents).value(),
      DType::f32,
      MemoryKind::host,
  };
}

Result<std::uint64_t>
product(const std::initializer_list<std::uint64_t> factors) noexcept {
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

Status validate_rope_tensor(const TensorView view) noexcept {
  const auto status = validate_f32_host(view);
  if (!status.ok()) {
    return status;
  }
  if (view.shape.rank != 4 || view.shape.extents[0] == 0 ||
      view.shape.extents[1] == 0 || view.shape.extents[2] == 0 ||
      view.shape.extents[3] == 0 || view.shape.extents[3] % 2 != 0) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  return Status::success();
}

Status validate_attention_shapes(
    const ConstTensorView query, const ConstTensorView key_cache,
    const ConstTensorView value_cache, const TensorView output,
    const std::span<const std::uint32_t> context_lengths) noexcept {
  for (const auto status :
       {validate_f32_host(query), validate_f32_host(key_cache),
        validate_f32_host(value_cache), validate_f32_host(output)}) {
    if (!status.ok()) {
      return status;
    }
  }
  if (query.shape.rank != 4 || key_cache.shape.rank != 4 ||
      value_cache.shape != key_cache.shape || output.shape != query.shape) {
    return Status::failure(ErrorCode::invalid_shape);
  }

  const std::uint64_t batch = query.shape.extents[0];
  const std::uint64_t tokens = query.shape.extents[1];
  const std::uint64_t query_heads = query.shape.extents[2];
  const std::uint64_t head_dim = query.shape.extents[3];
  const std::uint64_t maximum_context = key_cache.shape.extents[1];
  const std::uint64_t kv_heads = key_cache.shape.extents[2];
  if (batch == 0 || tokens == 0 || query_heads == 0 || head_dim == 0 ||
      maximum_context == 0 || kv_heads == 0 ||
      key_cache.shape.extents[0] != batch ||
      key_cache.shape.extents[3] != head_dim || query_heads % kv_heads != 0 ||
      context_lengths.size() != batch) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  for (const auto context : context_lengths) {
    if (context < tokens || context > maximum_context) {
      return Status::failure(ErrorCode::invalid_argument);
    }
  }

  const auto output_vs_query = validate_no_overlap(query, output);
  const auto output_vs_key = validate_no_overlap(key_cache, output);
  const auto output_vs_value = validate_no_overlap(value_cache, output);
  if (!output_vs_query.ok()) {
    return output_vs_query;
  }
  if (!output_vs_key.ok()) {
    return output_vs_key;
  }
  return output_vs_value;
}

Status attention_impl(const ConstTensorView query,
                      const ConstTensorView key_cache,
                      const ConstTensorView value_cache,
                      const std::span<const std::uint32_t> context_lengths,
                      const TensorView output,
                      const std::span<float> scratch) noexcept {
  const std::uint64_t batch = query.shape.extents[0];
  const std::uint64_t tokens = query.shape.extents[1];
  const std::uint64_t query_heads = query.shape.extents[2];
  const std::uint64_t head_dim = query.shape.extents[3];
  const std::uint64_t maximum_context = key_cache.shape.extents[1];
  const std::uint64_t kv_heads = key_cache.shape.extents[2];
  const std::uint64_t groups = query_heads / kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

  const auto* const query_data = static_cast<const float*>(query.data);
  const auto* const key_data = static_cast<const float*>(key_cache.data);
  const auto* const value_data = static_cast<const float*>(value_cache.data);
  auto* const output_data = static_cast<float*>(output.data);

  for (std::uint64_t batch_index = 0; batch_index < batch; ++batch_index) {
    const std::uint64_t context = context_lengths[batch_index];
    const std::uint64_t query_start = context - tokens;
    for (std::uint64_t token = 0; token < tokens; ++token) {
      const std::uint64_t valid_keys = query_start + token + 1;
      for (std::uint64_t query_head = 0; query_head < query_heads;
           ++query_head) {
        const std::uint64_t kv_head = query_head / groups;
        const std::uint64_t scratch_offset =
            ((batch_index * tokens + token) * query_heads + query_head) *
            maximum_context;
        auto logits =
            scratch.subspan(static_cast<std::size_t>(scratch_offset),
                            static_cast<std::size_t>(maximum_context));
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint64_t key_token = 0; key_token < valid_keys; ++key_token) {
          float dot = 0.0F;
          for (std::uint64_t dimension = 0; dimension < head_dim; ++dimension) {
            const std::uint64_t query_offset =
                (((batch_index * tokens + token) * query_heads + query_head) *
                     head_dim +
                 dimension);
            const std::uint64_t key_offset =
                (((batch_index * maximum_context + key_token) * kv_heads +
                  kv_head) *
                     head_dim +
                 dimension);
            dot += query_data[query_offset] * key_data[key_offset];
          }
          logits[static_cast<std::size_t>(key_token)] = dot * scale;
          maximum =
              std::max(maximum, logits[static_cast<std::size_t>(key_token)]);
        }

        float denominator = 0.0F;
        for (std::uint64_t key_token = 0; key_token < valid_keys; ++key_token) {
          auto& probability = logits[static_cast<std::size_t>(key_token)];
          probability = std::exp(probability - maximum);
          denominator += probability;
        }
        const float inverse_denominator = 1.0F / denominator;
        for (std::uint64_t key_token = 0; key_token < valid_keys; ++key_token) {
          logits[static_cast<std::size_t>(key_token)] *= inverse_denominator;
        }

        for (std::uint64_t dimension = 0; dimension < head_dim; ++dimension) {
          float weighted_value = 0.0F;
          for (std::uint64_t key_token = 0; key_token < valid_keys;
               ++key_token) {
            const std::uint64_t value_offset =
                (((batch_index * maximum_context + key_token) * kv_heads +
                  kv_head) *
                     head_dim +
                 dimension);
            weighted_value += logits[static_cast<std::size_t>(key_token)] *
                              value_data[value_offset];
          }
          const std::uint64_t output_offset =
              (((batch_index * tokens + token) * query_heads + query_head) *
                   head_dim +
               dimension);
          output_data[output_offset] = weighted_value;
        }
      }
    }
  }
  return Status::success();
}

Status validate_weight(const ConstTensorView weight,
                       const std::span<const std::uint64_t> extents) noexcept {
  const auto status = validate_f32_host(weight);
  if (!status.ok()) {
    return status;
  }
  return require_shape(weight.shape, static_cast<std::uint8_t>(extents.size()),
                       extents);
}

Status validate_layer(const LayerWeights& weights, const DecoderLayerSpec& spec,
                      const TensorView hidden, const CpuKvView cache,
                      const std::span<const std::uint32_t> positions,
                      const std::span<const std::uint32_t> context_lengths,
                      const CpuWorkspace& workspace,
                      const DecoderLayerTrace* const trace,
                      std::uint64_t& batch, std::uint64_t& tokens,
                      std::uint64_t& maximum_context) noexcept {
  if (spec.hidden_size == 0 || spec.intermediate_size == 0 ||
      spec.query_heads == 0 || spec.kv_heads == 0 || spec.head_dim == 0 ||
      spec.query_heads % spec.kv_heads != 0 ||
      static_cast<std::uint64_t>(spec.query_heads) * spec.head_dim !=
          spec.hidden_size ||
      !std::isfinite(spec.rms_norm_epsilon) || spec.rms_norm_epsilon <= 0.0F ||
      !std::isfinite(spec.rope.theta) || spec.rope.theta <= 0.0F ||
      spec.rope.max_positions == 0) {
    return Status::failure(ErrorCode::invalid_model);
  }
  const auto hidden_status = validate_f32_host(hidden);
  if (!hidden_status.ok()) {
    return hidden_status;
  }
  if (hidden.shape.rank != 3 || hidden.shape.extents[0] == 0 ||
      hidden.shape.extents[1] == 0 ||
      hidden.shape.extents[2] != spec.hidden_size) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  batch = hidden.shape.extents[0];
  tokens = hidden.shape.extents[1];

  for (const auto status :
       {validate_f32_host(cache.key), validate_f32_host(cache.value)}) {
    if (!status.ok()) {
      return status;
    }
  }
  if (cache.key.shape.rank != 4 || cache.value.shape != cache.key.shape ||
      cache.key.shape.extents[0] != batch ||
      cache.key.shape.extents[2] != spec.kv_heads ||
      cache.key.shape.extents[3] != spec.head_dim) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  maximum_context = cache.key.shape.extents[1];
  if (maximum_context == 0 || positions.size() != batch * tokens ||
      context_lengths.size() != batch) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  for (std::uint64_t batch_index = 0; batch_index < batch; ++batch_index) {
    const std::uint64_t context = context_lengths[batch_index];
    if (context < tokens || context > maximum_context) {
      return Status::failure(ErrorCode::invalid_argument);
    }
    const std::uint64_t start = context - tokens;
    for (std::uint64_t token = 0; token < tokens; ++token) {
      const auto position =
          positions[static_cast<std::size_t>(batch_index * tokens + token)];
      if (position != start + token || position >= spec.rope.max_positions) {
        return Status::failure(ErrorCode::invalid_argument);
      }
    }
  }

  const std::uint64_t hidden_size = spec.hidden_size;
  const std::uint64_t intermediate = spec.intermediate_size;
  const std::uint64_t kv_width =
      static_cast<std::uint64_t>(spec.kv_heads) * spec.head_dim;
  const std::array<std::uint64_t, 1> norm_shape{hidden_size};
  const std::array<std::uint64_t, 2> query_shape{hidden_size, hidden_size};
  const std::array<std::uint64_t, 2> key_value_shape{kv_width, hidden_size};
  const std::array<std::uint64_t, 2> gate_up_shape{intermediate, hidden_size};
  const std::array<std::uint64_t, 2> down_shape{hidden_size, intermediate};
  for (const auto status :
       {validate_weight(weights.input_norm, norm_shape),
        validate_weight(weights.attention.query, query_shape),
        validate_weight(weights.attention.key, key_value_shape),
        validate_weight(weights.attention.value, key_value_shape),
        validate_weight(weights.attention.output, query_shape),
        validate_weight(weights.post_attention_norm, norm_shape),
        validate_weight(weights.mlp.gate, gate_up_shape),
        validate_weight(weights.mlp.up, gate_up_shape),
        validate_weight(weights.mlp.down, down_shape)}) {
    if (!status.ok()) {
      return status;
    }
  }

  const auto required = required_decoder_layer_workspace_floats(
      spec, batch, tokens, maximum_context);
  if (!required || required.value() > workspace.capacity()) {
    return Status::failure(ErrorCode::insufficient_memory);
  }

  const std::array<std::uint64_t, 1> workspace_shape{workspace.capacity()};
  const auto workspace_view = make_f32_view(
      const_cast<float*>(workspace.floats().data()), workspace_shape);
  for (const auto status :
       {validate_no_overlap(as_const(hidden), cache.key),
        validate_no_overlap(as_const(hidden), cache.value),
        validate_no_overlap(as_const(hidden), workspace_view),
        validate_no_overlap(as_const(cache.key), cache.value),
        validate_no_overlap(as_const(cache.key), workspace_view),
        validate_no_overlap(as_const(cache.value), workspace_view)}) {
    if (!status.ok()) {
      return status;
    }
  }

  const std::array<ConstTensorView, 9> all_weights{
      weights.input_norm,       weights.attention.query,
      weights.attention.key,    weights.attention.value,
      weights.attention.output, weights.post_attention_norm,
      weights.mlp.gate,         weights.mlp.up,
      weights.mlp.down,
  };
  for (const auto& weight : all_weights) {
    for (const auto status : {validate_no_overlap(weight, hidden),
                              validate_no_overlap(weight, cache.key),
                              validate_no_overlap(weight, cache.value),
                              validate_no_overlap(weight, workspace_view)}) {
      if (!status.ok()) {
        return status;
      }
    }
  }

  if (trace != nullptr) {
    for (const auto& trace_view : {trace->after_attention, trace->after_mlp}) {
      const auto trace_status = validate_f32_host(trace_view);
      if (!trace_status.ok()) {
        return trace_status;
      }
      if (trace_view.shape != hidden.shape) {
        return Status::failure(ErrorCode::invalid_shape);
      }
      const auto hidden_overlap =
          validate_no_overlap(as_const(hidden), trace_view);
      if (!hidden_overlap.ok()) {
        return hidden_overlap;
      }
      for (const auto overlap_status :
           {validate_no_overlap(as_const(cache.key), trace_view),
            validate_no_overlap(as_const(cache.value), trace_view),
            validate_no_overlap(as_const(workspace_view), trace_view)}) {
        if (!overlap_status.ok()) {
          return overlap_status;
        }
      }
      for (const auto& weight : all_weights) {
        const auto overlap_status = validate_no_overlap(weight, trace_view);
        if (!overlap_status.ok()) {
          return overlap_status;
        }
      }
    }
    const auto traces_overlap =
        validate_no_overlap(as_const(trace->after_attention), trace->after_mlp);
    if (!traces_overlap.ok()) {
      return traces_overlap;
    }
  }
  return Status::success();
}

} // namespace

Result<CpuWorkspace>
CpuWorkspace::allocate(const std::uint64_t float_capacity) noexcept {
  if (float_capacity > std::numeric_limits<std::size_t>::max()) {
    return Result<CpuWorkspace>::failure(
        Status::failure(ErrorCode::resource_limit));
  }
  const auto bytes = checked_multiply(float_capacity, sizeof(float));
  if (!bytes) {
    return Result<CpuWorkspace>::failure(bytes.status());
  }
  auto storage = HostBuffer::allocate(bytes.value(), 64);
  if (!storage) {
    return Result<CpuWorkspace>::failure(storage.status());
  }
  return Result<CpuWorkspace>::success(CpuWorkspace{
      std::move(storage).value(), static_cast<std::size_t>(float_capacity)});
}

Status rms_norm_f32(const ConstTensorView input, const ConstTensorView weight,
                    const float epsilon, const TensorView output) noexcept {
  for (const auto status : {validate_f32_host(input), validate_f32_host(weight),
                            validate_f32_host(output)}) {
    if (!status.ok()) {
      return status;
    }
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0F || input.shape.rank == 0 ||
      weight.shape.rank != 1 || output.shape != input.shape ||
      input.shape.extents[input.shape.rank - 1] != weight.shape.extents[0]) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  const auto alias_status = validate_output_alias(input, output, true);
  const auto weight_alias_status = validate_no_overlap(weight, output);
  if (!alias_status.ok()) {
    return alias_status;
  }
  if (!weight_alias_status.ok()) {
    return weight_alias_status;
  }

  const auto element_count = checked_numel(input.shape);
  if (!element_count) {
    return element_count.status();
  }
  const std::uint64_t hidden = weight.shape.extents[0];
  if (hidden == 0 || element_count.value() % hidden != 0) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  const std::uint64_t rows = element_count.value() / hidden;
  const auto* const input_data = static_cast<const float*>(input.data);
  const auto* const weight_data = static_cast<const float*>(weight.data);
  auto* const output_data = static_cast<float*>(output.data);
  for (std::uint64_t row = 0; row < rows; ++row) {
    float sum_squares = 0.0F;
    for (std::uint64_t column = 0; column < hidden; ++column) {
      const float value = input_data[row * hidden + column];
      sum_squares += value * value;
    }
    const float inverse_rms =
        1.0F / std::sqrt(sum_squares / static_cast<float>(hidden) + epsilon);
    for (std::uint64_t column = 0; column < hidden; ++column) {
      output_data[row * hidden + column] =
          input_data[row * hidden + column] * inverse_rms * weight_data[column];
    }
  }
  return Status::success();
}

Status linear_f32(const ConstTensorView input, const ConstTensorView weight,
                  const TensorView output) noexcept {
  for (const auto status : {validate_f32_host(input), validate_f32_host(weight),
                            validate_f32_host(output)}) {
    if (!status.ok()) {
      return status;
    }
  }
  if (input.shape.rank != 2 || weight.shape.rank != 2 ||
      output.shape.rank != 2 || input.shape.extents[0] == 0 ||
      input.shape.extents[1] == 0 || weight.shape.extents[0] == 0 ||
      weight.shape.extents[1] != input.shape.extents[1] ||
      output.shape.extents[0] != input.shape.extents[0] ||
      output.shape.extents[1] != weight.shape.extents[0]) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  const auto input_alias = validate_no_overlap(input, output);
  const auto weight_alias = validate_no_overlap(weight, output);
  if (!input_alias.ok()) {
    return input_alias;
  }
  if (!weight_alias.ok()) {
    return weight_alias;
  }

  const std::uint64_t rows = input.shape.extents[0];
  const std::uint64_t input_width = input.shape.extents[1];
  const std::uint64_t output_width = weight.shape.extents[0];
  const auto* const input_data = static_cast<const float*>(input.data);
  const auto* const weight_data = static_cast<const float*>(weight.data);
  auto* const output_data = static_cast<float*>(output.data);
#if defined(MARKETFORGE_USE_ACCELERATE)
  if (rows <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()) &&
      input_width <=
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) &&
      output_width <=
          static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows),
                static_cast<int>(output_width), static_cast<int>(input_width),
                1.0F, input_data, static_cast<int>(input_width), weight_data,
                static_cast<int>(input_width), 0.0F, output_data,
                static_cast<int>(output_width));
    return Status::success();
  }
#endif
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t output_column = 0; output_column < output_width;
         ++output_column) {
      float sum = 0.0F;
      for (std::uint64_t input_column = 0; input_column < input_width;
           ++input_column) {
        sum += input_data[row * input_width + input_column] *
               weight_data[output_column * input_width + input_column];
      }
      output_data[row * output_width + output_column] = sum;
    }
  }
  return Status::success();
}

Status softmax_f32(const ConstTensorView input,
                   const std::span<const std::uint32_t> valid_lengths,
                   const TensorView output) noexcept {
  for (const auto status :
       {validate_f32_host(input), validate_f32_host(output)}) {
    if (!status.ok()) {
      return status;
    }
  }
  if (input.shape.rank != 2 || output.shape != input.shape ||
      input.shape.extents[0] == 0 || input.shape.extents[1] == 0 ||
      valid_lengths.size() != input.shape.extents[0]) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  const auto alias_status = validate_output_alias(input, output, true);
  if (!alias_status.ok()) {
    return alias_status;
  }

  const std::uint64_t rows = input.shape.extents[0];
  const std::uint64_t columns = input.shape.extents[1];
  const auto* const input_data = static_cast<const float*>(input.data);
  for (std::uint64_t row = 0; row < rows; ++row) {
    const std::uint64_t valid = valid_lengths[row];
    if (valid == 0 || valid > columns) {
      return Status::failure(ErrorCode::invalid_argument,
                             static_cast<std::uint32_t>(row));
    }
    for (std::uint64_t column = 0; column < valid; ++column) {
      if (!std::isfinite(input_data[row * columns + column])) {
        return Status::failure(ErrorCode::invalid_argument,
                               static_cast<std::uint32_t>(row));
      }
    }
  }

  auto* const output_data = static_cast<float*>(output.data);
  for (std::uint64_t row = 0; row < rows; ++row) {
    const std::uint64_t valid = valid_lengths[row];
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint64_t column = 0; column < valid; ++column) {
      maximum = std::max(maximum, input_data[row * columns + column]);
    }
    float denominator = 0.0F;
    for (std::uint64_t column = 0; column < valid; ++column) {
      auto& value = output_data[row * columns + column];
      value = std::exp(input_data[row * columns + column] - maximum);
      denominator += value;
    }
    const float inverse_denominator = 1.0F / denominator;
    for (std::uint64_t column = 0; column < valid; ++column) {
      output_data[row * columns + column] *= inverse_denominator;
    }
    for (std::uint64_t column = valid; column < columns; ++column) {
      output_data[row * columns + column] = 0.0F;
    }
  }
  return Status::success();
}

Status apply_rope_f32(const TensorView query, const TensorView key,
                      const std::span<const std::uint32_t> positions,
                      const RopeSpec& spec) noexcept {
  const auto query_status = validate_rope_tensor(query);
  const auto key_status = validate_rope_tensor(key);
  if (!query_status.ok()) {
    return query_status;
  }
  if (!key_status.ok()) {
    return key_status;
  }
  if (query.shape.extents[0] != key.shape.extents[0] ||
      query.shape.extents[1] != key.shape.extents[1] ||
      query.shape.extents[3] != key.shape.extents[3] ||
      positions.size() != query.shape.extents[0] * query.shape.extents[1] ||
      !std::isfinite(spec.theta) || spec.theta <= 0.0F ||
      spec.max_positions == 0) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  const auto query_range = memory_range(query);
  const auto key_range = memory_range(key);
  if (!query_range || !key_range) {
    return !query_range ? query_range.status() : key_range.status();
  }
  if (overlaps(query_range.value(), key_range.value())) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  for (const auto position : positions) {
    if (position >= spec.max_positions) {
      return Status::failure(ErrorCode::invalid_argument);
    }
  }

  const std::uint64_t batch = query.shape.extents[0];
  const std::uint64_t tokens = query.shape.extents[1];
  const std::uint64_t query_heads = query.shape.extents[2];
  const std::uint64_t key_heads = key.shape.extents[2];
  const std::uint64_t head_dim = query.shape.extents[3];
  const std::uint64_t half = head_dim / 2;
  auto* const query_data = static_cast<float*>(query.data);
  auto* const key_data = static_cast<float*>(key.data);

  const auto rotate = [&](float* data, const std::uint64_t heads,
                          const std::uint64_t batch_index,
                          const std::uint64_t token,
                          const std::uint32_t position) {
    for (std::uint64_t head = 0; head < heads; ++head) {
      const std::uint64_t vector_offset =
          ((batch_index * tokens + token) * heads + head) * head_dim;
      for (std::uint64_t dimension = 0; dimension < half; ++dimension) {
        const float exponent =
            static_cast<float>(2 * dimension) / static_cast<float>(head_dim);
        const float angle =
            static_cast<float>(position) / std::pow(spec.theta, exponent);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = data[vector_offset + dimension];
        const float second = data[vector_offset + half + dimension];
        data[vector_offset + dimension] = first * cosine - second * sine;
        data[vector_offset + half + dimension] = second * cosine + first * sine;
      }
    }
  };

  for (std::uint64_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::uint64_t token = 0; token < tokens; ++token) {
      const auto position =
          positions[static_cast<std::size_t>(batch_index * tokens + token)];
      rotate(query_data, query_heads, batch_index, token, position);
      rotate(key_data, key_heads, batch_index, token, position);
    }
  }
  return Status::success();
}

Status append_kv_f32(const ConstTensorView key, const ConstTensorView value,
                     const std::span<const std::uint32_t> positions,
                     const CpuKvView cache) noexcept {
  for (const auto status :
       {validate_f32_host(key), validate_f32_host(value),
        validate_f32_host(cache.key), validate_f32_host(cache.value)}) {
    if (!status.ok()) {
      return status;
    }
  }
  if (key.shape.rank != 4 || value.shape != key.shape ||
      cache.key.shape.rank != 4 || cache.value.shape != cache.key.shape ||
      key.shape.extents[0] != cache.key.shape.extents[0] ||
      key.shape.extents[2] != cache.key.shape.extents[2] ||
      key.shape.extents[3] != cache.key.shape.extents[3] ||
      positions.size() != key.shape.extents[0] * key.shape.extents[1]) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  const std::uint64_t maximum_context = cache.key.shape.extents[1];
  for (const auto position : positions) {
    if (position >= maximum_context) {
      return Status::failure(ErrorCode::invalid_argument);
    }
  }
  for (const auto status : {validate_no_overlap(key, cache.key),
                            validate_no_overlap(key, cache.value),
                            validate_no_overlap(value, cache.key),
                            validate_no_overlap(value, cache.value)}) {
    if (!status.ok()) {
      return status;
    }
  }

  const std::uint64_t batch = key.shape.extents[0];
  const std::uint64_t tokens = key.shape.extents[1];
  const std::uint64_t heads = key.shape.extents[2];
  const std::uint64_t head_dim = key.shape.extents[3];
  const std::uint64_t token_width = heads * head_dim;
  const auto* const key_data = static_cast<const float*>(key.data);
  const auto* const value_data = static_cast<const float*>(value.data);
  auto* const key_cache = static_cast<float*>(cache.key.data);
  auto* const value_cache = static_cast<float*>(cache.value.data);
  for (std::uint64_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::uint64_t token = 0; token < tokens; ++token) {
      const auto position =
          positions[static_cast<std::size_t>(batch_index * tokens + token)];
      const std::uint64_t source_offset =
          (batch_index * tokens + token) * token_width;
      const std::uint64_t destination_offset =
          (batch_index * maximum_context + position) * token_width;
      std::memcpy(key_cache + destination_offset, key_data + source_offset,
                  static_cast<std::size_t>(token_width * sizeof(float)));
      std::memcpy(value_cache + destination_offset, value_data + source_offset,
                  static_cast<std::size_t>(token_width * sizeof(float)));
    }
  }
  return Status::success();
}

Result<std::uint64_t>
required_attention_workspace_floats(const ConstTensorView query,
                                    const ConstTensorView key_cache) noexcept {
  if (query.shape.rank != 4 || key_cache.shape.rank != 4) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::invalid_shape));
  }
  return product({query.shape.extents[0], query.shape.extents[1],
                  query.shape.extents[2], key_cache.shape.extents[1]});
}

Status attention_f32(const ConstTensorView query,
                     const ConstTensorView key_cache,
                     const ConstTensorView value_cache,
                     const std::span<const std::uint32_t> context_lengths,
                     const TensorView output,
                     CpuWorkspace& workspace) noexcept {
  const auto validation = validate_attention_shapes(
      query, key_cache, value_cache, output, context_lengths);
  if (!validation.ok()) {
    return validation;
  }
  const auto required = required_attention_workspace_floats(query, key_cache);
  if (!required || required.value() > workspace.capacity()) {
    return Status::failure(ErrorCode::insufficient_memory);
  }

  const auto query_count = checked_numel(query.shape);
  if (!query_count) {
    return Status::failure(ErrorCode::invalid_shape);
  }
  const auto* const query_data = static_cast<const float*>(query.data);
  const auto* const key_data = static_cast<const float*>(key_cache.data);
  const auto* const value_data = static_cast<const float*>(value_cache.data);
  for (std::uint64_t index = 0; index < query_count.value(); ++index) {
    if (!std::isfinite(query_data[index])) {
      return Status::failure(ErrorCode::invalid_argument);
    }
  }
  const std::uint64_t batch = key_cache.shape.extents[0];
  const std::uint64_t maximum_context = key_cache.shape.extents[1];
  const std::uint64_t kv_heads = key_cache.shape.extents[2];
  const std::uint64_t head_dim = key_cache.shape.extents[3];
  for (std::uint64_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::uint64_t token = 0; token < context_lengths[batch_index];
         ++token) {
      for (std::uint64_t head = 0; head < kv_heads; ++head) {
        for (std::uint64_t dimension = 0; dimension < head_dim; ++dimension) {
          const std::uint64_t index =
              (((batch_index * maximum_context + token) * kv_heads + head) *
                   head_dim +
               dimension);
          if (!std::isfinite(key_data[index]) ||
              !std::isfinite(value_data[index])) {
            return Status::failure(ErrorCode::invalid_argument);
          }
        }
      }
    }
  }
  return attention_impl(
      query, key_cache, value_cache, context_lengths, output,
      workspace.floats().first(static_cast<std::size_t>(required.value())));
}

Result<std::uint64_t> required_decoder_layer_workspace_floats(
    const DecoderLayerSpec& spec, const std::uint64_t batch_size,
    const std::uint64_t query_tokens,
    const std::uint64_t maximum_context) noexcept {
  const auto rows = checked_multiply(batch_size, query_tokens);
  if (!rows) {
    return rows;
  }
  const auto row_hidden = checked_multiply(rows.value(), spec.hidden_size);
  const auto kv_width = checked_multiply(spec.kv_heads, spec.head_dim);
  const auto row_kv = kv_width
                          ? checked_multiply(rows.value(), kv_width.value())
                          : Result<std::uint64_t>::failure(kv_width.status());
  const auto logits =
      product({batch_size, query_tokens, spec.query_heads, maximum_context});
  const auto row_intermediate =
      checked_multiply(rows.value(), spec.intermediate_size);
  if (!row_hidden || !row_kv || !logits || !row_intermediate) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }

  auto attention_phase = checked_multiply(row_hidden.value(), 3);
  if (attention_phase) {
    const auto two_kv = checked_multiply(row_kv.value(), 2);
    if (!two_kv) {
      return two_kv;
    }
    attention_phase = checked_add(attention_phase.value(), two_kv.value());
  }
  if (attention_phase) {
    attention_phase = checked_add(attention_phase.value(), logits.value());
  }
  auto mlp_phase = checked_multiply(row_intermediate.value(), 2);
  if (mlp_phase) {
    mlp_phase = checked_add(mlp_phase.value(), row_hidden.value());
  }
  if (!attention_phase || !mlp_phase) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  return Result<std::uint64_t>::success(
      std::max(attention_phase.value(), mlp_phase.value()));
}

Status smollm2_decoder_layer_f32(
    const LayerWeights& weights, const DecoderLayerSpec& spec,
    const TensorView hidden, const CpuKvView cache,
    const std::span<const std::uint32_t> positions,
    const std::span<const std::uint32_t> context_lengths,
    CpuWorkspace& workspace, DecoderLayerTrace* const trace) noexcept {
  std::uint64_t batch = 0;
  std::uint64_t tokens = 0;
  std::uint64_t maximum_context = 0;
  const auto validation =
      validate_layer(weights, spec, hidden, cache, positions, context_lengths,
                     workspace, trace, batch, tokens, maximum_context);
  if (!validation.ok()) {
    return validation;
  }

  const std::uint64_t rows = batch * tokens;
  const std::uint64_t hidden_size = spec.hidden_size;
  const std::uint64_t kv_width =
      static_cast<std::uint64_t>(spec.kv_heads) * spec.head_dim;
  const std::uint64_t row_hidden = rows * hidden_size;
  const std::uint64_t row_kv = rows * kv_width;
  const std::uint64_t logits_count =
      batch * tokens * spec.query_heads * maximum_context;
  auto scratch = workspace.floats();
  std::size_t cursor = 0;
  const auto take = [&](const std::uint64_t count) {
    auto result = scratch.subspan(cursor, static_cast<std::size_t>(count));
    cursor += static_cast<std::size_t>(count);
    return result;
  };
  auto normalized = take(row_hidden);
  auto query = take(row_hidden);
  auto key = take(row_kv);
  auto value = take(row_kv);
  auto attention = take(row_hidden);
  auto logits = take(logits_count);

  const std::array<std::uint64_t, 3> hidden_shape{batch, tokens, hidden_size};
  const std::array<std::uint64_t, 2> flat_hidden_shape{rows, hidden_size};
  const std::array<std::uint64_t, 2> flat_kv_shape{rows, kv_width};
  const std::array<std::uint64_t, 4> query_shape{
      batch, tokens, spec.query_heads, spec.head_dim};
  const std::array<std::uint64_t, 4> key_value_shape{
      batch, tokens, spec.kv_heads, spec.head_dim};

  auto normalized_hidden = make_f32_view(normalized.data(), hidden_shape);
  auto query_flat = make_f32_view(query.data(), flat_hidden_shape);
  auto key_flat = make_f32_view(key.data(), flat_kv_shape);
  auto value_flat = make_f32_view(value.data(), flat_kv_shape);
  auto query_heads = make_f32_view(query.data(), query_shape);
  auto key_heads = make_f32_view(key.data(), key_value_shape);
  auto value_heads = make_f32_view(value.data(), key_value_shape);
  auto attention_heads = make_f32_view(attention.data(), query_shape);

  auto status = rms_norm_f32(as_const(hidden), weights.input_norm,
                             spec.rms_norm_epsilon, normalized_hidden);
  if (status.ok()) {
    status =
        linear_f32(make_const_f32_view(normalized.data(), flat_hidden_shape),
                   weights.attention.query, query_flat);
  }
  if (status.ok()) {
    status =
        linear_f32(make_const_f32_view(normalized.data(), flat_hidden_shape),
                   weights.attention.key, key_flat);
  }
  if (status.ok()) {
    status =
        linear_f32(make_const_f32_view(normalized.data(), flat_hidden_shape),
                   weights.attention.value, value_flat);
  }
  if (status.ok()) {
    status = apply_rope_f32(query_heads, key_heads, positions, spec.rope);
  }
  if (status.ok()) {
    status = append_kv_f32(as_const(key_heads), as_const(value_heads),
                           positions, cache);
  }
  if (status.ok()) {
    status = attention_impl(as_const(query_heads), as_const(cache.key),
                            as_const(cache.value), context_lengths,
                            attention_heads, logits);
  }
  if (!status.ok()) {
    return status;
  }

  auto projected = make_f32_view(normalized.data(), flat_hidden_shape);
  status = linear_f32(make_const_f32_view(attention.data(), flat_hidden_shape),
                      weights.attention.output, projected);
  if (!status.ok()) {
    return status;
  }
  auto* const hidden_data = static_cast<float*>(hidden.data);
  for (std::uint64_t index = 0; index < row_hidden; ++index) {
    hidden_data[index] += normalized[static_cast<std::size_t>(index)];
  }
  if (trace != nullptr) {
    std::memcpy(trace->after_attention.data, hidden.data,
                static_cast<std::size_t>(row_hidden * sizeof(float)));
  }

  cursor = 0;
  auto post_norm = take(row_hidden);
  const std::uint64_t row_intermediate = rows * spec.intermediate_size;
  auto gate = take(row_intermediate);
  auto up = take(row_intermediate);
  const std::array<std::uint64_t, 2> flat_intermediate_shape{
      rows, spec.intermediate_size};
  auto post_norm_hidden = make_f32_view(post_norm.data(), hidden_shape);
  auto gate_view = make_f32_view(gate.data(), flat_intermediate_shape);
  auto up_view = make_f32_view(up.data(), flat_intermediate_shape);

  status = rms_norm_f32(as_const(hidden), weights.post_attention_norm,
                        spec.rms_norm_epsilon, post_norm_hidden);
  if (status.ok()) {
    status =
        linear_f32(make_const_f32_view(post_norm.data(), flat_hidden_shape),
                   weights.mlp.gate, gate_view);
  }
  if (status.ok()) {
    status =
        linear_f32(make_const_f32_view(post_norm.data(), flat_hidden_shape),
                   weights.mlp.up, up_view);
  }
  if (!status.ok()) {
    return status;
  }
  for (std::uint64_t index = 0; index < row_intermediate; ++index) {
    const float gate_value = gate[static_cast<std::size_t>(index)];
    const float silu = gate_value / (1.0F + std::exp(-gate_value));
    gate[static_cast<std::size_t>(index)] =
        silu * up[static_cast<std::size_t>(index)];
  }
  auto down = make_f32_view(post_norm.data(), flat_hidden_shape);
  status = linear_f32(make_const_f32_view(gate.data(), flat_intermediate_shape),
                      weights.mlp.down, down);
  if (!status.ok()) {
    return status;
  }
  for (std::uint64_t index = 0; index < row_hidden; ++index) {
    hidden_data[index] += post_norm[static_cast<std::size_t>(index)];
  }
  if (trace != nullptr) {
    std::memcpy(trace->after_mlp.data, hidden.data,
                static_cast<std::size_t>(row_hidden * sizeof(float)));
  }
  return Status::success();
}

} // namespace marketforge
