#include "marketforge/model/model_spec.hpp"

#include <cmath>
#include <limits>

#include "marketforge/core/checked_math.hpp"

namespace marketforge {
namespace {

Result<std::uint64_t> add_to(const std::uint64_t total,
                             const Result<std::uint64_t>& value) noexcept {
  if (!value) {
    return Result<std::uint64_t>::failure(value.status());
  }
  return checked_add(total, value.value());
}

Result<std::uint64_t> matrix_parameters(const std::uint64_t rows,
                                        const std::uint64_t columns) noexcept {
  return checked_multiply(rows, columns);
}

} // namespace

Status validate(const ModelSpec& spec) noexcept {
  switch (spec.architecture) {
  case Architecture::smollm2_llama:
  case Architecture::qwen2:
    break;
  default:
    return Status::failure(ErrorCode::invalid_model);
  }

  if (spec.layers == 0 || spec.hidden_size == 0 ||
      spec.intermediate_size == 0 || spec.query_heads == 0 ||
      spec.kv_heads == 0 || spec.head_dim == 0 || spec.vocabulary_size == 0 ||
      spec.max_positions == 0) {
    return Status::failure(ErrorCode::invalid_model);
  }
  if (spec.query_heads % spec.kv_heads != 0) {
    return Status::failure(ErrorCode::invalid_model);
  }

  const auto query_width = checked_multiply(spec.query_heads, spec.head_dim);
  if (!query_width ||
      query_width.value() != static_cast<std::uint64_t>(spec.hidden_size)) {
    return Status::failure(ErrorCode::invalid_model);
  }

  if (!std::isfinite(spec.rms_norm_epsilon) || spec.rms_norm_epsilon <= 0.0F ||
      !std::isfinite(spec.rope_theta) || spec.rope_theta <= 0.0F) {
    return Status::failure(ErrorCode::invalid_model);
  }

  const auto parameters = estimate_parameter_count(spec);
  return parameters ? Status::success() : parameters.status();
}

Result<std::uint64_t> estimate_parameter_count(const ModelSpec& spec) noexcept {
  switch (spec.architecture) {
  case Architecture::smollm2_llama:
  case Architecture::qwen2:
    break;
  default:
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::invalid_model));
  }

  if (spec.layers == 0 || spec.hidden_size == 0 ||
      spec.intermediate_size == 0 || spec.query_heads == 0 ||
      spec.kv_heads == 0 || spec.head_dim == 0 || spec.vocabulary_size == 0) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::invalid_model));
  }
  if (spec.query_heads % spec.kv_heads != 0) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::invalid_model));
  }

  const std::uint64_t hidden = spec.hidden_size;
  const std::uint64_t intermediate = spec.intermediate_size;
  const auto query_width = checked_multiply(spec.query_heads, spec.head_dim);
  const auto kv_width = checked_multiply(spec.kv_heads, spec.head_dim);
  if (!query_width || !kv_width) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  if (query_width.value() != static_cast<std::uint64_t>(spec.hidden_size)) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::invalid_model));
  }

  auto total = matrix_parameters(spec.vocabulary_size, hidden);
  if (!total) {
    return total;
  }

  std::uint64_t per_layer = 0;
  const auto add_matrix = [&per_layer](const std::uint64_t rows,
                                       const std::uint64_t columns) -> Status {
    const auto next = add_to(per_layer, matrix_parameters(rows, columns));
    if (!next) {
      return next.status();
    }
    per_layer = next.value();
    return Status::success();
  };

  if (!add_matrix(hidden, query_width.value()).ok() ||
      !add_matrix(hidden, kv_width.value()).ok() ||
      !add_matrix(hidden, kv_width.value()).ok() ||
      !add_matrix(query_width.value(), hidden).ok() ||
      !add_matrix(hidden, intermediate).ok() ||
      !add_matrix(hidden, intermediate).ok() ||
      !add_matrix(intermediate, hidden).ok()) {
    return Result<std::uint64_t>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }

  const auto two_norms = checked_multiply(hidden, 2);
  auto layer_with_norms = add_to(per_layer, two_norms);
  if (!layer_with_norms) {
    return layer_with_norms;
  }
  per_layer = layer_with_norms.value();

  if (spec.qkv_bias) {
    const auto two_kv_biases = checked_multiply(kv_width.value(), 2);
    if (!two_kv_biases) {
      return two_kv_biases;
    }
    const auto all_biases =
        checked_add(query_width.value(), two_kv_biases.value());
    const auto layer_with_biases = add_to(per_layer, all_biases);
    if (!layer_with_biases) {
      return layer_with_biases;
    }
    per_layer = layer_with_biases.value();
  }

  const auto stacked_layers = checked_multiply(per_layer, spec.layers);
  total = add_to(total.value(), stacked_layers);
  if (!total) {
    return total;
  }

  total = checked_add(total.value(), hidden);
  if (!total) {
    return total;
  }

  if (!spec.tied_embeddings) {
    total =
        add_to(total.value(), matrix_parameters(spec.vocabulary_size, hidden));
  }
  return total;
}

Result<ModelFootprint>
estimate_model_footprint(const ModelSpec& spec, const DType source_weight_dtype,
                         const DType materialized_weight_dtype,
                         const DType kv_dtype) noexcept {
  const auto validation = validate(spec);
  if (!validation.ok()) {
    return Result<ModelFootprint>::failure(validation);
  }

  const auto parameters = estimate_parameter_count(spec);
  const auto source_size = dtype_size(source_weight_dtype);
  const auto materialized_size = dtype_size(materialized_weight_dtype);
  const auto kv_size = dtype_size(kv_dtype);
  if (!parameters || !source_size || !materialized_size || !kv_size) {
    return Result<ModelFootprint>::failure(
        Status::failure(ErrorCode::unsupported_dtype));
  }

  const auto source_bytes =
      checked_multiply(parameters.value(), source_size.value());
  const auto materialized_bytes =
      checked_multiply(parameters.value(), materialized_size.value());
  if (!source_bytes || !materialized_bytes) {
    return Result<ModelFootprint>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  const auto peak_weights =
      checked_add(source_bytes.value(), materialized_bytes.value());
  if (!peak_weights) {
    return Result<ModelFootprint>::failure(peak_weights.status());
  }

  auto kv_bytes = checked_multiply(2, spec.layers);
  if (kv_bytes) {
    kv_bytes = checked_multiply(kv_bytes.value(), spec.kv_heads);
  }
  if (kv_bytes) {
    kv_bytes = checked_multiply(kv_bytes.value(), spec.head_dim);
  }
  if (kv_bytes) {
    kv_bytes = checked_multiply(kv_bytes.value(), kv_size.value());
  }
  if (!kv_bytes) {
    return Result<ModelFootprint>::failure(kv_bytes.status());
  }

  return Result<ModelFootprint>::success(ModelFootprint{
      parameters.value(),
      source_bytes.value(),
      materialized_bytes.value(),
      peak_weights.value(),
      kv_bytes.value(),
  });
}

Result<std::uint64_t>
estimate_peak_runtime_bytes(const ModelFootprint& footprint,
                            const RuntimeMemoryPlan& plan) noexcept {
  const auto kv_bytes =
      checked_multiply(footprint.kv_bytes_per_token, plan.resident_kv_tokens);
  if (!kv_bytes) {
    return kv_bytes;
  }
  auto total = checked_add(footprint.peak_cpu_weight_bytes, kv_bytes.value());
  if (total) {
    total = checked_add(total.value(), plan.workspace_bytes);
  }
  if (total) {
    total = checked_add(total.value(), plan.reserve_bytes);
  }
  return total;
}

Result<bool> fits_memory_budget(const ModelFootprint& footprint,
                                const RuntimeMemoryPlan& plan,
                                const std::uint64_t available_bytes) noexcept {
  const auto required = estimate_peak_runtime_bytes(footprint, plan);
  if (!required) {
    return Result<bool>::failure(required.status());
  }
  return Result<bool>::success(required.value() <= available_bytes);
}

} // namespace marketforge
